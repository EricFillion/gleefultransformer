import json
import logging
import math
from os import PathLike
from pathlib import Path

import torch
from accelerate.data_loader import IterableDatasetShard
from accelerate.utils import broadcast_object_list, parse_flag_from_env
from torch.nn import functional as F
from transformers import Trainer, TrainingArguments

from gleefultransformer._artifact_paths import find_artifact
from gleefultransformer._core.exceptions import GleefulIOError, GleefulInputError, GleefulResumeError
from gleefultransformer._core.utils.train.confirm_optimizer import get_optim
from gleefultransformer._core.utils.train.get_precision import get_precision
from gleefultransformer._core.utils.train.get_tok_data import TokenizedDataset


def resolve_training_precision(requested, device):
    if requested == "auto":
        return get_precision(device)
    if requested == "fp16" and device.type == "cpu":
        raise GleefulInputError("FP16 mixed precision requires an accelerator device.")
    if requested == "bf16" and device.type == "cuda" and get_precision(device) != "bf16":
        raise GleefulInputError(f"BF16 mixed precision requires native BF16 support on {device}.")
    return requested


def make_training_arguments(args, precision, *, has_eval, use_cpu):
    if parse_flag_from_env("ACCELERATE_USE_DEEPSPEED"):
        raise GleefulInputError("Launch distributed training with torchrun. Use fsdp_config to enable FSDP2.")
    bf16 = precision == "bf16"
    fp16 = precision == "fp16"
    eval_enabled = has_eval and bool(args.eval_steps)
    save_strategy = "no"
    if args.checkpoint_steps > 0:
        save_strategy = "steps"
    elif args.save_best and eval_enabled:
        save_strategy = "best"

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_torch" if args.optim == "adamw" else "sgd",
        weight_decay=0.05 if args.optim == "adamw" else 1e-4,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs or args.bs,
        gradient_accumulation_steps=args.gas,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=args.eval_steps if eval_enabled else None,
        logging_strategy="steps" if args.log_steps else "no",
        logging_steps=args.log_steps or 1,
        save_strategy=save_strategy,
        save_steps=max(1, args.checkpoint_steps),
        save_total_limit=2,
        metric_for_best_model="loss" if args.save_best and eval_enabled else None,
        greater_is_better=False if args.save_best and eval_enabled else None,
        bf16=bf16,
        fp16=fp16,
        tf32=False if args.precision == "fp32" and torch.cuda.is_available() else None,
        use_cpu=use_cpu,
        max_grad_norm=1.0 if fp16 else 0.0,
        seed=args.seed,
        label_names=["labels"],
        prediction_loss_only=True,
        accelerator_config={"dispatch_batches": False},
        dataloader_pin_memory=not use_cpu and torch.cuda.is_available(),
        include_num_input_tokens_seen="non_padding",
        report_to="none",
        disable_tqdm=True,
        fsdp=args.fsdp_config is not None,
        fsdp_config=args.fsdp_config,
    )
    training_args.mixed_precision = "no" if precision == "fp32" else precision
    return training_args


def resolve_fsdp_config(args):
    config = args.fsdp_config
    if config is None:
        return
    if isinstance(config, (str, PathLike)):
        try:
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GleefulInputError(f"Could not read fsdp_config configuration: {exc}") from exc
        if not isinstance(config, dict):
            raise GleefulInputError("`fsdp_config` must contain a JSON object.")

    config = {"version": 2, "state_dict_type": "SHARDED_STATE_DICT", **config}
    if str(config["version"]) != "2":
        raise GleefulInputError("Gleeful Transformer supports FSDP version 2.")
    args.fsdp_config = config


def read_trainer_checkpoint(path):
    path = Path(path).resolve()
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        if (path / "lr_sched.pt").is_file():
            raise GleefulResumeError(
                "This checkpoint uses the previous training engine. Resume with the previous "
                "Gleeful Transformer version, or load the checkpoint as a model for a new run."
            )
        raise GleefulResumeError(f"No Trainer state found in {path}. Pass a checkpoint directory.")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        tracker_state = json.loads(find_artifact(path, "gleeful_tracker_state.json").read_text(encoding="utf-8"))
        gleeful_args = json.loads((path / "train_args.json").read_text(encoding="utf-8"))
        if gleeful_args.pop("deepspeed", None) is not None:
            raise GleefulResumeError(
                "Resume this DeepSpeed checkpoint with the previous Gleeful Transformer version, "
                "or load a consolidated model export for a new run."
            )
        metadata = json.loads(find_artifact(path, "gleeful_resume.json").read_text(encoding="utf-8"))
        if state["global_step"] != tracker_state["last_checkpoint_step"]:
            raise ValueError("Trainer and GleefulTracker record different checkpoint steps")
        if state["global_step"] >= state["max_steps"]:
            raise GleefulResumeError("The checkpoint is from a completed training run.")
    except (OSError, ValueError, KeyError) as exc:
        raise GleefulResumeError(f"Could not read checkpoint in {path}: {exc}") from exc
    return str(path), state, tracker_state, gleeful_args, metadata


def on_main_process(training_args, operation):
    if training_args.world_size == 1:
        return operation()

    result = [None, None]
    if training_args.process_index == 0:
        try:
            result[0] = operation()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(result)
    if result[1] is not None:
        raise GleefulIOError(result[1])
    return result[0]


class GleefulTrainer(Trainer):
    def __init__(self, *, gleeful_args=None, causal=False, **kwargs):
        self.gleeful_args = gleeful_args
        self.causal = causal
        self.last_update_loss = None
        self._fsdp_eval_totals = None
        self._fsdp_eval_remaining = 0
        super().__init__(**kwargs)

    def precision_report(self):
        mixed_precision = self.accelerator.mixed_precision
        dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in self.model.parameters()}
        report = {
            "requested_precision": self.gleeful_args.precision,
            "training_precision": "fp32" if mixed_precision == "no" else mixed_precision,
            "weight_dtypes": sorted(dtypes),
        }
        quantization = getattr(self.model.config, "quantization_config", None)
        compute_dtype = getattr(quantization, "bnb_4bit_compute_dtype", None)
        if compute_dtype is not None:
            report["quantization_compute_dtype"] = str(compute_dtype).removeprefix("torch.")
        return report

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = get_optim(
                self.gleeful_args, self.model if model is None else model, logging.getLogger(__name__)
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None and self.gleeful_args.lr_sched == "warmup_then_decay" and num_training_steps >= 8:
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = math.ceil(num_training_steps / 10)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps - warmup_steps, eta_min=0.1 * self.gleeful_args.lr
            )
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, decay], milestones=[warmup_steps]
            )
            self._created_lr_scheduler = True
        return super().create_scheduler(num_training_steps, optimizer)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        if batches:
            self.last_update_loss = None
        return batches, num_items

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.last_update_loss = (
            loss if self.last_update_loss is None else self.last_update_loss + loss
        )
        return loss

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
        metric_key_prefix="eval",
    ):
        dataset = dataloader.dataset
        if isinstance(dataset, IterableDatasetShard):
            dataset = dataset.dataset
        track_token_loss = (
            self.is_fsdp_enabled and self.causal and isinstance(dataset, TokenizedDataset)
            and (self.args.prediction_loss_only if prediction_loss_only is None else prediction_loss_only)
        )
        if track_token_loss:
            self._fsdp_eval_remaining = len(dataset)
            self._fsdp_eval_totals = torch.zeros(2, device=self.args.device, dtype=torch.float64)
        try:
            output = super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix,
            )
            if track_token_loss:
                loss, tokens = self._fsdp_eval_totals.unbind()
                output.metrics[f"{metric_key_prefix}_loss"] = (loss / tokens).item()
                output = output._replace(num_samples=len(dataset))
            return output
        finally:
            self._fsdp_eval_totals = None

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self._fsdp_eval_totals is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            logits = model(**inputs, use_cache=False).logits
            targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none",
            ).reshape_as(targets)

        # Accelerate repeats early rows to fill the last shard. Gather one pair
        # per example so repeated rows can be excluded before averaging tokens.
        stats = torch.stack((losses.sum(dim=-1, dtype=torch.float64), targets.ne(-100).sum(dim=-1)), dim=-1)
        real_examples = self.accelerator.gather(stats)[:self._fsdp_eval_remaining]
        self._fsdp_eval_remaining -= real_examples.shape[0]
        self._fsdp_eval_totals += real_examples.sum(dim=0)
        return None, None, None

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if getattr(self.model, "peft_config", None) and getattr(self.model, "is_loaded_in_4bit", False):
            # PEFT saves the adapter config. Reloading QLoRA also needs the base quantization config.
            self.model.config.save_pretrained(output_dir or self.args.output_dir)

    def export_model(self, path):
        if self.is_fsdp_enabled:
            plugin = self.accelerator.state.fsdp_plugin
            checkpoint_type = plugin.state_dict_type
            plugin.set_state_dict_type("FULL_STATE_DICT")
            try:
                self.save_model(path)
            finally:
                plugin.set_state_dict_type(checkpoint_type)
        else:
            self.save_model(path)
        self.accelerator.wait_for_everyone()
