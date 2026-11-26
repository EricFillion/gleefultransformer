import os
import textwrap
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, Literal

import torch
from accelerate.utils import parse_flag_from_env
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    TextIteratorStreamer,
    default_data_collator,
)

from gleefultransformer._core.args import GleefulEvalArgs, GleefulTrainArgs
from gleefultransformer._core.exceptions import (
    GleefulDatasetError,
    GleefulDeviceError,
    GleefulIOError,
    GleefulInputError,
    GleefulNoModelError,
    GleefulPushError,
    GleefulResumeError,
    GleefulSaveError,
)
from gleefultransformer._core.gleeful_tasks.args import CHATCallArgs, CHATTokArgs
from gleefultransformer._core.gleeful_tasks.chat_adapters import is_adapter_directory, load_chat_adapter
from gleefultransformer._core.gleeful_tasks.chat_stream_handlers import (
    CHATStreamResult,
    get_response_parser,
    parsed_stream_events,
    stream_events,
    stream_protocol,
)
from gleefultransformer._core.gleeful_tasks.misc import generate_gen_kwargs, get_pad_eos
from gleefultransformer._core.gleeful_tasks.misc.call_utils import GenerationThread, format_messages
from gleefultransformer._core.gleeful_tasks.results import CHATResult
from gleefultransformer._core.gleeful_tasks.tok.tok_functions import get_max_in_len, tokenize_chat_template
from gleefultransformer._core.gleeful_tracker.trainer_callback import (
    GleefulTrackerCallback,
    TrainingObserver,
)
from gleefultransformer._core.loops import EvalResult, TrainResult
from gleefultransformer._core.trainer import (
    GleefulTrainer,
    make_training_arguments,
    on_main_process,
    read_trainer_checkpoint,
    resolve_fsdp_config,
    resolve_training_precision,
)
from gleefultransformer._core.utils import (
    create_tracker_dir,
    get_device,
    get_logger,
    get_model_components,
    get_num_training_steps,
    get_precision,
    get_procs,
    prepare_output_locations,
    resolve_input_files,
    save_json_tok_data,
    tok_dir_to_dataset,
    write_details_file,
)
from gleefultransformer._core.utils.train.get_tok_data import TokenizedDataset
from gleefultransformer._core.validator import EvalValidator, TrainValidator
from gleefultransformer.gleefulsearch import GleefulSearch

if TYPE_CHECKING:
    from peft import LoraConfig


class GleefulChat:
    """Run inference, train, and evaluate causal chat models."""

    def __init__(
        self,
        model_name: str | os.PathLike[str] | PreTrainedModel | None = "openai/gpt-oss-20b",
        *,
        trust_remote_code: bool = False,
        tokenizer: str | os.PathLike[str] | PreTrainedTokenizerBase | None = None,
        gleeful_search: GleefulSearch | None = None,
        adapter: "LoraConfig | None" = None,
        quantization: Literal["nf4"] | BitsAndBytesConfig | None = None,
    ):
        self.gleeful_search = gleeful_search
        self.logger = get_logger()
        if tokenizer is None and not isinstance(model_name, (str, os.PathLike)):
            raise ValueError("Pass a tokenizer when providing a loaded model or no model.")
        if isinstance(tokenizer, os.PathLike):
            tokenizer = os.fspath(tokenizer)

        self.device = get_device()
        self.precision_type = get_precision(device=self.device)
        self.trust_remote_code = trust_remote_code
        self._train_just_happened = True
        self._distributed_trainer = None

        self.config, self.tokenizer, self.model = self._load_chat_components(
            model_name, tokenizer, adapter, quantization=quantization
        )
        self._validate_chat_template()

        if self.model is not None:
            if not getattr(self.model, "peft_config", None):
                embeddings = self.model.get_input_embeddings()
                if embeddings.num_embeddings < len(self.tokenizer):
                    self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.logger.info("Using device: %s", self.device)

    def _validate_chat_template(self):
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The tokenizer must include a chat template.")

    def __call__(
        self, text: str | list[dict], args: CHATCallArgs | None = None
    ) -> CHATResult:
        """Return the final reply to a prompt or a list of chat messages."""
        return CHATResult(
            text="".join(event.text for event in self.stream(text, args) if event.marker == "text")
        )

    def stream(
        self, text: str | list[dict], args: CHATCallArgs | None = None
    ) -> Iterator[CHATStreamResult]:
        """Yield reply text, reasoning markers, and optional search results."""
        args = args if args is not None else CHATCallArgs()
        messages = format_messages(text)
        if self.gleeful_search is not None:
            query = messages[-1].get("content")
            if not isinstance(query, str):
                raise GleefulInputError("Search requires a text message as the query.")
            yield CHATStreamResult(text="", marker="search", payload={"query": query})
            passages = []
            for result in self.gleeful_search(query, args=args.search_args) or ():
                yield CHATStreamResult(
                    text="",
                    marker="search_result",
                    payload={
                        "text": result.text,
                        "best_sentence": result.best_sentence,
                        "metadata": result.metadata,
                    },
                )
                passages.append(result.text)

            if passages:
                context = "\n\n".join(passages)
                messages[-1] = {
                    **messages[-1],
                    "content": (
                        f"Based on the search query: '{query}'. "
                        f"The following data may be relevant: ' {context} '\n\n{query}"
                    ),
                }

        self._get_model_ready_inference()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        chat_template = self.tokenizer.get_chat_template()
        parser = get_response_parser(self.tokenizer, chat_template, inputs["input_ids"][0])
        inputs = inputs.to(self.model.device)
        protocol = stream_protocol(chat_template) if parser is None else None
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=parser is None and protocol is None,
            clean_up_tokenization_spaces=False,
        )
        pad_token_id, eos_token_id = get_pad_eos(self.tokenizer, self.model)
        kwargs = generate_gen_kwargs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            streamer=streamer,
            args=args,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            context_limit=getattr(self.config, "max_position_embeddings", None),
        )
        generator = GenerationThread(self.model, kwargs)
        generator.start()
        try:
            if parser is None:
                yield from stream_events(streamer, protocol, self.tokenizer.all_special_tokens)
            else:
                yield from parsed_stream_events(streamer, parser)
        finally:
            generator.join()

    def _load_resume_model_components(self, checkpoint_path):
        return self._load_chat_components(checkpoint_path, None, None, weight_precision="auto")

    def _load_chat_components(self, model_name, tokenizer, adapter, *, quantization=None, weight_precision=None):
        if isinstance(model_name, os.PathLike):
            model_name = os.fspath(model_name)
        if adapter is not None or quantization is not None or is_adapter_directory(model_name):
            return load_chat_adapter(
                model_name,
                adapter,
                quantization=quantization,
                trust_remote_code=self.trust_remote_code,
                tokenizer_path=tokenizer,
                device=self.device,
                precision=self.precision_type,
            )
        if weight_precision is None:
            # Local exports should reload the dtype recorded in their checkpoint.
            weight_precision = (
                "auto" if isinstance(model_name, str) and os.path.isdir(model_name)
                else self.precision_type
            )
        return get_model_components(
            model_name_path=model_name,
            trust_remote_code=self.trust_remote_code,
            tokenizer_path=tokenizer,
            precision=weight_precision,
        )

    def _move_model_to_device(self):
        if not (
            getattr(self.model, "is_loaded_in_4bit", False)
            and getattr(self.model, "hf_device_map", None)
        ):
            self.model.to(self.device)

    @staticmethod
    def _format_tokenized_example(example: dict) -> dict:
        return {key: example[key] for key in ("input_ids", "attention_mask", "labels")}

    def train(
        self,
        train_path: str = "",
        args: GleefulTrainArgs | None = None,
        eval_path: str = "",
        *,
        resume_path: str = "",
        tok_args: CHATTokArgs | None = None,
    ) -> TrainResult:
        """Train on a chat dataset."""
        return self._train(train_path, args, eval_path, resume_path=resume_path, tok_args=tok_args)

    def _train(
        self,
        train_path: str,
        args: GleefulTrainArgs | None,
        eval_path: str,
        *,
        resume_path: str = "",
        tok_args: CHATTokArgs | None = None,
        observer: TrainingObserver | None = None,
    ) -> TrainResult:
        if self._distributed_trainer is not None:
            raise GleefulInputError("Create a new model object before starting another distributed run.")
        if args is None:
            args = GleefulTrainArgs()
        if not isinstance(args, GleefulTrainArgs):
            raise TypeError("args must be a GleefulTrainArgs instance.")

        tracker_state = None
        checkpoint_state = None
        metadata = None
        use_cpu = self.device.type == "cpu" or parse_flag_from_env("ACCELERATE_USE_CPU")
        training_device = torch.device("cpu") if use_cpu else self.device
        train_path = os.path.abspath(train_path) if train_path else ""
        eval_path = os.path.abspath(eval_path) if eval_path else ""
        if resume_path:
            resume_path, checkpoint_state, tracker_state, saved_args, metadata = (
                read_trainer_checkpoint(resume_path)
            )
            train_path = self._resume_data_path(train_path, metadata, "train")
            eval_path = self._resume_data_path(eval_path, metadata, "eval")
            if args.out_dir != GleefulTrainArgs.out_dir:
                saved_args["out_dir"] = args.out_dir
            if args.run_name:
                saved_args["run_name"] = args.run_name
            if "bf16" in saved_args:
                legacy_bf16 = saved_args.pop("bf16")
                if legacy_bf16 is True:
                    saved_args["precision"] = "bf16"
                elif legacy_bf16 is False:
                    saved_args["precision"] = "fp16" if get_precision(training_device) == "fp16" else "fp32"
            if saved_args.pop("fsdp", False) and saved_args.get("fsdp_config") is None:
                saved_args["fsdp_config"] = {}
            args = GleefulTrainArgs(**saved_args)

        validator = TrainValidator(
            train_path=train_path,
            args=args,
            eval_path=eval_path,
            resume_path=resume_path,
            model=self.model,
        )
        args = validator.args
        resolve_fsdp_config(args)

        precision = resolve_training_precision(args.precision, training_device)
        if metadata and "precision" in metadata:
            saved_precision = metadata["precision"]["training_precision"]
            if precision != saved_precision:
                raise GleefulResumeError(f"Resume with the original training precision of {saved_precision}.")
        training_args = make_training_arguments(
            deepcopy(args), precision, has_eval=bool(eval_path), use_cpu=use_cpu,
        )
        self.device = training_args.device
        world_size = training_args.world_size
        if observer is not None and world_size > 1:
            raise GleefulInputError("A training observer requires single-process training.")
        use_fsdp = args.fsdp_config is not None
        if use_fsdp and world_size < 2:
            raise GleefulInputError("Launch FSDP2 with torchrun and at least two processes.")
        if metadata and world_size != metadata["world_size"]:
            raise GleefulResumeError(f"Resume with the original world size of {metadata['world_size']}.")

        self._prepare_training_model(resume_path, training_args, sharded=use_fsdp)
        if use_fsdp and (
            getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_loaded_in_8bit", False)
        ):
            raise GleefulInputError("Quantized adapters currently support single-device and DDP training.")

        def prepare_data():
            run_dir = os.path.abspath(create_tracker_dir(args.out_dir, "train", args.run_name))
            data = {"world_size": world_size}
            for label, source in (("train", train_path), ("eval", eval_path)):
                tok_dir = source
                if source and os.path.isfile(source):
                    tok_dir = self._tokenize_input_file(source, run_dir, label, tok_args=tok_args)
                data[f"{label}_path"] = metadata[f"{label}_path"] if metadata else source
                data[f"{label}_data"] = os.fspath(tok_dir)
            return run_dir, data

        run_dir, resume_metadata = on_main_process(training_args, prepare_data)
        train_dataset = TokenizedDataset(resume_metadata["train_data"], args.seed)
        eval_dataset = (
            TokenizedDataset(resume_metadata["eval_data"], args.seed)
            if eval_path and args.eval_steps else None
        )
        training_args.output_dir = run_dir
        training_args.max_steps = args.max_steps or get_num_training_steps(
            train_cases=len(train_dataset), num_devices=world_size, epochs=args.epochs,
            gas=args.gas, bs=args.bs,
        )
        if checkpoint_state and training_args.max_steps != checkpoint_state["max_steps"]:
            raise GleefulResumeError("The data or training configuration changed the number of training steps.")

        callback = GleefulTrackerCallback(
            args, run_dir, tracker_state,
            resume_metadata=resume_metadata, resume_path=resume_path or None,
            observer=observer,
        )
        trainer = self._make_trainer(
            training_args, gleeful_args=args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, callbacks=[callback], causal=True,
        )
        if use_fsdp and not trainer.accelerator.is_fsdp2:
            raise GleefulInputError("FSDP2 did not activate. Use a supported accelerator with torchrun.")
        callback.precision_report = trainer.precision_report
        try:
            result = trainer.train(resume_from_checkpoint=resume_path or None)
        finally:
            callback.close()
            self.model = trainer.model
            self.model.eval()

        if world_size > 1:
            self._distributed_trainer = trainer
        self._train_just_happened = not use_fsdp
        final_train_loss = result.training_loss
        if trainer.last_update_loss is not None:
            final_train_loss = trainer.accelerator.reduce(trainer.last_update_loss, reduction="mean").item()
        return TrainResult(
            final_train_loss=final_train_loss,
            final_eval_loss=callback.state.eval_loss if callback.state.last_eval_step else None,
            best_eval_loss=callback.state.best_eval_loss,
        )

    @staticmethod
    def _resume_data_path(requested, metadata, label):
        source = metadata[f"{label}_path"]
        tokenized = metadata[f"{label}_data"]
        if requested and requested not in (source, tokenized):
            raise GleefulResumeError(f"Resume with the original {label} data: {source}")
        return tokenized

    def _prepare_training_model(self, resume_path, training_args, *, sharded):
        if resume_path and not sharded:
            self.config, self.tokenizer, self.model = self._load_resume_model_components(resume_path)
            self._validate_chat_template()
        elif self.model is None:
            raise GleefulResumeError("A distributed resume needs the original chat model.")

        if training_args.fp16:
            # AMP must unscale gradients from FP32 trainable parameters.
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                        parameter.data = parameter.data.float()
        elif not training_args.bf16 and not getattr(self.model, "is_quantized", False):
            self.model.float()

    def _make_trainer(self, training_args, **kwargs):
        trainer = GleefulTrainer(
            model=self.model, args=training_args, processing_class=self.tokenizer,
            data_collator=default_data_collator, **kwargs,
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    def eval(
        self, eval_path: str = "", args: GleefulEvalArgs | None = None, *,
        tok_args: CHATTokArgs | None = None,
    ) -> EvalResult:
        if args is None:
            args = GleefulEvalArgs()
        validator = EvalValidator(eval_path=eval_path, args=args, out_dir=args.out_dir)
        if self.model is None:
            raise GleefulNoModelError("Provide a model before evaluating.")

        trainer = self._distributed_trainer
        if trainer is None:
            eval_args = GleefulTrainArgs(bs=args.bs, out_dir=args.out_dir, seed=args.seed)
            training_args = make_training_arguments(
                eval_args, self.precision_type, has_eval=False,
                use_cpu=self.device.type == "cpu",
            )
            trainer = self._make_trainer(training_args)
            self.device = trainer.args.device

        def prepare_eval_data():
            if validator.eval_source == "file":
                out_dir = create_tracker_dir(args.out_dir, "eval", args.run_name)
                return self._tokenize_input_file(
                    eval_path, out_dir, "eval", in_eval=True, tok_args=tok_args
                )
            return eval_path

        tok_dir = on_main_process(trainer.args, prepare_eval_data)
        dataset = TokenizedDataset(tok_dir, args.seed)
        previous_bs = trainer.args.per_device_eval_batch_size
        trainer.args.per_device_eval_batch_size = args.bs
        try:
            metrics = trainer.evaluate(eval_dataset=dataset)
        finally:
            trainer.args.per_device_eval_batch_size = previous_bs
        return EvalResult(loss=metrics["eval_loss"])

    def tok(self, path: str, out_dir: str, args: CHATTokArgs | None = None):
        if args is None:
            args = CHATTokArgs()
        if not isinstance(args, CHATTokArgs):
            raise TypeError("args must be a CHATTokArgs instance.")
        if args.bs < 1 or args.shards < 1 or args.procs < -1:
            raise ValueError("bs and shards must be positive, and procs must be at least -1.")

        file_infos = resolve_input_files(os.fspath(path))

        output_paths, output_files = prepare_output_locations(out_dir, args.shards)
        num_cases = 0
        procs = get_procs(args.procs)
        try:
            with tqdm(file_infos, desc="Tokenizing files") as files:
                for file, file_type in files:
                    if args.max_cases > 0 and num_cases >= args.max_cases:
                        break
                    try:
                        dataset = load_dataset(file_type, data_files=[file], split="train")
                    except Exception as exc:
                        raise GleefulDatasetError(f"Error loading {file}: {exc}") from exc

                    if args.max_cases > 0:
                        remaining = min(len(dataset), args.max_cases - num_cases)
                        dataset = dataset.select(range(remaining))
                    tokenized = tokenize_chat_template(
                        tokenizer=self.tokenizer,
                        dataset=dataset,
                        max_len=get_max_in_len(args.max_len, self.tokenizer),
                        bs=args.bs,
                        procs=procs,
                    )
                    save_json_tok_data(
                        tokenized, output_files, args.shards, self._format_tokenized_example
                    )
                    num_cases += len(tokenized)
        finally:
            for output_file in output_files:
                output_file.close()

        write_details_file(out_dir, num_cases, output_paths)

    def save(self, path: str):
        if self.model is None:
            raise GleefulNoModelError(
                f"No model found. Provide a model_name or PreTrainedModel when instantiating {self.__class__.__name__}."
            )
        if self._distributed_trainer is not None:
            try:
                self._distributed_trainer.export_model(os.fspath(path))
            except Exception as e:
                raise GleefulSaveError(f"Error exporting distributed model to {path}: {e}") from e
            return

        self._save_components(path)

    def _save_components(self, path):
        try:
            self.model.save_pretrained(path)
        except Exception as e:
            raise GleefulSaveError(f"Error saving model to {path}: {e}")
        try:
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise GleefulSaveError(f"Error saving tokenizer to {path}: {e}")
        try:
            self.config.save_pretrained(path)
        except Exception as e:
            raise GleefulSaveError(f"Error saving config to {path}: {e}")

    def push(self, repo_id: str, private: bool = True):
        if self.model is None:
            raise GleefulNoModelError(
                f"No model found. Provide a model_name or PreTrainedModel when instantiating {self.__class__.__name__}."
            )
        trainer = self._distributed_trainer
        if trainer is not None:
            if trainer.is_fsdp_enabled:
                raise GleefulPushError("Export with save() on all ranks, then reopen the export to push it.")
            if not trainer.is_world_process_zero():
                return

        api = HfApi()
        try:
            api.create_repo(repo_id, exist_ok=True, private=private)
        except Exception as e:
            self.logger.warning(f"Could not create repo {e}")
            return

        try:
            self.logger.info("Pushing model...")
            with TemporaryDirectory() as path:
                self._save_components(path)
                api.upload_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="model",
                    ignore_patterns="README.md",
                    commit_message="Uploaded model from Gleeful Transformer",
                )
        except Exception as e:
            raise GleefulPushError(f"Error pushing model: {e}") from e

        try:
            if not api.file_exists(repo_id, "README.md", repo_type="model"):
                self.logger.info("Pushing README...")
                api.upload_file(
                    path_or_fileobj=self._get_readme(repo_id).encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )
        except Exception as e:
            self.logger.warning("Error ensuring README exists, but model uploaded OK: %s", e)

    @staticmethod
    def _get_readme(repo_id: str) -> str:
        return textwrap.dedent(f"""\
        ---
        tags:
        - gleefultransformer
        ---
        # {repo_id}

        [Gleeful Transformer on GitHub](https://github.com/EricFillion/gleefultransformer)

        ## Usage

        ```python
        from gleefultransformer import GleefulChat

        chat = GleefulChat(model_name="{repo_id}")
        print(chat("Hello!").text)
        ```
        """)

    def _tokenize_input_file(
        self, train_path: str, out_dir: str, label: str, in_eval: bool = False, *, tok_args=None
    ) -> str:
        dir_name = f"tok_{label}_data" if in_eval else f"data/tok_{label}_data"
        tok_dir = os.path.join(out_dir, dir_name)
        try:
            os.makedirs(tok_dir, exist_ok=True)
        except Exception as e:
            raise GleefulIOError(f"error making directory ({tok_dir}): {e}")

        if tok_args is None:
            self.tok(os.fspath(train_path), tok_dir)
        else:
            self.tok(os.fspath(train_path), tok_dir, tok_args)
        self.logger.info(
            f"{label.capitalize()} data has been tokenized and saved to {tok_dir}"
        )
        return tok_dir

    def _get_model_ready_inference(self):
        if self.model is None:
            raise GleefulNoModelError("Load a chat model before generating a reply.")
        if self._distributed_trainer is not None:
            raise GleefulInputError("Export the distributed model and reload it before generating replies.")
        if self._train_just_happened:
            self.logger.info(f"Moving model to {self.device}")
            try:
                self._move_model_to_device()
            except Exception as e:
                raise GleefulDeviceError(f"Failed to move model to device: {e}")

            self._train_just_happened = False

        self.model.eval()

    @staticmethod
    def tok_dir_to_dataset(tok_dir: str) -> Dataset:
        return tok_dir_to_dataset(tok_dir)
