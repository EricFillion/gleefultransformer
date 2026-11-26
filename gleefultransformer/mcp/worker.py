"""Independent worker for the local Gleeful Transformer MCP queue."""

import hashlib
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from tempfile import gettempdir
from time import monotonic, sleep
from uuid import uuid4

import torch
from accelerate.utils import parse_flag_from_env
from filelock import FileLock, Timeout

from gleefultransformer import CHATCallArgs, CHATTokArgs, GleefulChat, GleefulEvalArgs, GleefulTrainArgs
from gleefultransformer._core.utils import get_device, get_precision

from .errors import ServiceError
from .registry import model_digest, tokenizer_digest
from .store import RunStore

_LOGGER = logging.getLogger(__name__)
_PROGRESS_INTERVAL = 0.25


class _Cancelled(Exception):
    pass


def _device():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ServiceError("invalid_worker_configuration", "Start the MCP worker as a single process.")
    if parse_flag_from_env("ACCELERATE_USE_CPU"):
        return torch.device("cpu")
    return get_device()


def _device_lock(device) -> FileLock:
    identity = str(device)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        identity = f"cuda:{getattr(properties, 'uuid', None) or device.index}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    # Distinct workspaces on this host still share a physical accelerator.
    account = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
    private_dir = Path(gettempdir()) / f"gleeful-mcp-{account}"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    info = private_dir.stat()
    if private_dir.is_symlink() or (
        hasattr(os, "getuid") and (info.st_uid != os.getuid() or info.st_mode & 0o077)
    ):
        raise ServiceError("invalid_worker_configuration", "The local device lock directory is not private.")
    lock_path = private_dir / f"device-{suffix}.lock"
    if lock_path.is_symlink():
        raise ServiceError("invalid_worker_configuration", "The device lock must not be a symlink.")
    return FileLock(lock_path, timeout=0)


def _metric(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _loss_report(loss) -> dict:
    value = _metric(loss)
    result = {"loss": value}
    if value is None:
        result["warnings"] = ["Non-finite evaluation loss."]
    return result


class _Worker:
    def __init__(self, workspace: Path, device):
        self.workspace = workspace
        self.store = RunStore(workspace)
        self.device = device
        self.id = f"worker_{uuid4().hex}"
        self.artifacts = workspace / ".gleeful" / "artifacts"
        self._reply_model = None
        if self.artifacts.is_symlink():
            raise ServiceError("invalid_workspace", "The worker artifact directory must not be a symlink.")
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def recover(self):
        self.store.interrupt_abandoned(self.id)
        for directory in self.artifacts.iterdir():
            if not directory.name.startswith("run_") or not directory.is_dir() or directory.is_symlink():
                continue
            try:
                snapshot = self.store.read(directory.name)
            except ServiceError:
                continue
            if snapshot["state"] in {"cancelled", "failed", "interrupted"}:
                self._discard_export(directory)

    @staticmethod
    def _discard_export(directory: Path):
        if directory.is_symlink():
            return
        for name in ("export.staging", "model"):
            path = directory / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)

    def _cancelled(self, run_id):
        return self.store.read(run_id)["state"] == "cancel_requested"

    def _check_cancel(self, run_id):
        if self._cancelled(run_id):
            raise _Cancelled

    def _phase(self, run_id, phase):
        snapshot = self.store.update(run_id, self.id, phase=phase)
        if snapshot["state"] == "cancel_requested":
            raise _Cancelled

    def _local_path(self, record):
        try:
            path = (self.workspace / record["path"]).resolve(strict=True)
        except OSError as exc:
            raise ServiceError("snapshot_changed", "A registered snapshot is no longer available.") from exc
        if not path.is_relative_to(self.workspace):
            raise ServiceError("snapshot_changed", "The registered snapshot no longer resolves inside the workspace.")
        return path

    def _pin_dataset(self, record, destination: Path, limits):
        source = self._local_path(record)
        max_bytes = limits.get("max_dataset_bytes", 32 * 1024 ** 2)
        max_row_bytes = limits.get("max_row_bytes", 256 * 1024)
        digest = hashlib.sha256()
        row_count = size = 0
        with source.open("rb") as original, destination.open("xb") as snapshot:
            while line := original.readline(max_row_bytes + 1):
                size += len(line)
                if size > max_bytes or len(line) > max_row_bytes:
                    raise ServiceError("snapshot_changed", "A registered dataset no longer fits its limits.")
                digest.update(line)
                snapshot.write(line)
                row_count += bool(line.strip())
        if digest.hexdigest() != record["sha256"]:
            raise ServiceError("snapshot_changed", "A registered dataset has changed. Register the new revision.")
        return row_count

    def _load_model(self, record, limits, *, cache_reply=False):
        source = self._local_path(record)
        digest = model_digest(
            source, workspace=self.workspace,
            max_bytes=limits.get("max_model_bytes", 64 * 1024 ** 3),
            max_files=limits.get("max_model_files", 2048),
        )
        if digest != record["sha256"]:
            raise ServiceError("snapshot_changed", "A registered model has changed. Register the new revision.")
        if (source / "adapter_config.json").is_file():
            raise ServiceError("invalid_model", "This worker requires a consolidated local model.")
        key = (source, digest)
        if cache_reply and self._reply_model is not None and self._reply_model[0] == key:
            return self._reply_model[1]
        self._reply_model = None
        chat = GleefulChat(model_name=source, trust_remote_code=False)
        chat.device = self.device
        chat.precision_type = get_precision(self.device)
        if cache_reply:
            self._reply_model = (key, chat)
        return chat

    @staticmethod
    def _provenance(model, dataset, max_sequence_tokens, row_count):
        return {
            "model_id": model["id"],
            "model_revision": model["revision"],
            "tokenizer_revision": model.get("tokenizer_revision") or model.get("tokenizer_sha256"),
            "dataset_id": dataset["id"],
            "dataset_revision": dataset.get("revision") or dataset["sha256"],
            "max_sequence_tokens": max_sequence_tokens,
            "row_count": row_count,
        }

    def _observer(self, run_id, provenance):
        last_update = None

        def observe(event, fields):
            nonlocal last_update
            current_step = fields["current_step"]
            total_steps = fields["total_steps"]
            now = monotonic()
            if (
                event == "step" and last_update is not None and current_step < total_steps
                and now - last_update < _PROGRESS_INTERVAL
                and not self._cancelled(run_id)
            ):
                return False

            update = {
                "phase": "training",
                "progress": {"current_step": current_step, "total_steps": total_steps},
            }
            if event == "evaluation" and provenance is not None and fields["loss"] is not None:
                update["metrics"] = {
                    "validation": {
                        **provenance,
                        **_loss_report(fields["loss"]),
                        "step": current_step,
                    },
                }
            snapshot = self.store.update(run_id, self.id, **update)
            last_update = now
            if event == "train_begin" and snapshot["state"] == "cancel_requested":
                # Trainer enters its first epoch even if a start callback sets the stop flag.
                raise _Cancelled
            return snapshot["state"] == "cancel_requested"

        return observe

    def _finetune(self, run_id, payload, directory):
        model = payload["model"]
        settings = payload["settings"]
        limits = payload.get("limits", {})
        train_path = directory / "train.jsonl"
        eval_path = directory / "validation.jsonl"
        validation = payload.get("validation_dataset")
        self._phase(run_id, "preparing")
        self._pin_dataset(payload["train_dataset"], train_path, limits)
        provenance = None
        if validation is not None:
            row_count = self._pin_dataset(validation, eval_path, limits)
            provenance = self._provenance(model, validation, settings["max_sequence_tokens"], row_count)
            provenance["input_model_id"] = provenance.pop("model_id")
            provenance["input_model_revision"] = provenance.pop("model_revision")

        self._phase(run_id, "loading_model")
        chat = self._load_model(model, limits)
        if getattr(chat.model, "is_quantized", False):
            raise ServiceError("invalid_model", "Fine-tuning requires an unquantized model.")
        self._phase(run_id, "tokenizing")
        args = GleefulTrainArgs(
            out_dir=str(directory), run_name="training",
            max_steps=settings["max_steps"],
            bs=settings["batch_size"], eval_bs=settings["eval_batch_size"],
            gas=settings["gradient_accumulation_steps"], lr=settings["learning_rate"],
            precision=settings["precision"], seed=settings["seed"],
            log_steps=settings["log_steps"], eval_steps=settings["eval_steps"],
            checkpoint_steps=settings["checkpoint_steps"], save_best=settings["save_best"],
        )
        output = chat._train(
            str(train_path), args, str(eval_path) if validation is not None else "",
            tok_args=CHATTokArgs(max_len=settings["max_sequence_tokens"]),
            observer=self._observer(run_id, provenance),
        )
        self._phase(run_id, "exporting")
        staging = directory / "export.staging"
        chat.save(staging)
        del chat
        self._check_cancel(run_id)

        # Register a model only after the same loader can reopen its export.
        verified = GleefulChat(model_name=staging, trust_remote_code=False)
        tokenizer_sha = tokenizer_digest(verified.tokenizer)
        del verified
        digest = model_digest(
            staging, workspace=self.workspace,
            max_bytes=min(limits.get("max_output_bytes", 64 * 1024 ** 3),
                          limits.get("max_model_bytes", 64 * 1024 ** 3)),
            max_files=limits.get("max_model_files", 2048),
        )
        self._check_cancel(run_id)
        final = directory / "model"
        staging.rename(final)

        artifact_id = f"mdl_{run_id.removeprefix('run_')}"
        published = {
            **model,
            "id": artifact_id, "model_id": artifact_id,
            "path": str(final),
            "origin": "export",
            "sha256": digest, "revision": f"sha256:{digest}",
            "tokenizer_sha256": tokenizer_sha, "tokenizer_revision": f"sha256:{tokenizer_sha}",
        }
        result = {
            "final_model_id": artifact_id,
            "final_train_loss": _metric(output.final_train_loss),
            "best_eval_loss": _metric(output.best_eval_loss),
        }
        if provenance is not None and output.final_eval_loss is not None:
            result["validation"] = {**provenance, **_loss_report(output.final_eval_loss)}
        return result, published

    def _evaluate(self, run_id, payload, directory):
        model = payload["model"]
        settings = payload["settings"]
        limits = payload.get("limits", {})
        dataset = payload["dataset"]
        path = directory / "evaluation.jsonl"
        self._phase(run_id, "preparing")
        row_count = self._pin_dataset(dataset, path, limits)
        self._phase(run_id, "loading_model")
        chat = self._load_model(model, limits)
        self._phase(run_id, "evaluating")
        result = chat.eval(
            str(path),
            GleefulEvalArgs(
                bs=settings["eval_batch_size"], out_dir=str(directory),
                run_name="evaluation", seed=settings["seed"],
            ),
            tok_args=CHATTokArgs(max_len=settings["max_sequence_tokens"]),
        )
        self._check_cancel(run_id)
        return {
            **self._provenance(model, dataset, settings["max_sequence_tokens"], row_count),
            **_loss_report(result.loss),
        }

    def _reply(self, run_id, payload):
        model = payload["model"]
        self._phase(run_id, "loading_model")
        limits = payload.get("limits", {})
        chat = self._load_model(model, limits, cache_reply=True)
        self._phase(run_id, "generating")
        messages = payload["messages"]
        requested = payload["max_new_tokens"]
        try:
            prompt = chat.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            )
            prefix_tokens = len(prompt["input_ids"])
        except Exception as exc:
            raise ServiceError("invalid_messages", "The messages do not match the model's chat template.") from exc
        context_limit = model.get("max_sequence_tokens") or limits.get("max_sequence_tokens", 4096)
        available = context_limit - prefix_tokens
        if available <= 0:
            raise ServiceError("invalid_messages", "The messages fill the model's context window.")
        result = chat(messages, CHATCallArgs(max_len=min(requested, available)))
        self._check_cancel(run_id)
        return {"text": result.text, "model_id": model["id"], "model_revision": model["revision"]}

    def process_one(self):
        job = self.store.claim(self.id)
        if job is None:
            return False
        run_id = job["run_id"]
        directory = self.artifacts / run_id
        try:
            self._check_cancel(run_id)
            kind = job["kind"]
            payload = job["payload"]
            if kind == "reply":
                result, published = self._reply(run_id, payload), None
            elif kind in {"finetune", "evaluation"}:
                self._reply_model = None
                directory.mkdir()
                if kind == "finetune":
                    result, published = self._finetune(run_id, payload, directory)
                else:
                    result, published = self._evaluate(run_id, payload, directory), None
            else:
                raise ServiceError("unknown_run_kind", "The worker does not support this run kind.")
        except _Cancelled:
            snapshot = self.store.finish(run_id, self.id)
        except ServiceError as exc:
            snapshot = self.store.finish(run_id, self.id, error=exc.to_dict())
        except torch.OutOfMemoryError:
            _LOGGER.exception("Run %s ran out of device memory.", run_id)
            snapshot = self.store.finish(run_id, self.id, error={
                "code": "device_out_of_memory", "message": "The run exceeded available device memory.",
                "retryable": False,
            })
        except Exception:
            _LOGGER.exception("Worker failed while running %s.", run_id)
            snapshot = self.store.finish(run_id, self.id, error={
                "code": "worker_error", "message": "Execution failed. Inspect the local worker logs.",
                "retryable": False,
            })
        else:
            snapshot = self.store.finish(run_id, self.id, result=result, model=published)
        if snapshot["state"] != "succeeded":
            self._discard_export(directory)
        return True


def run_forever(workspace: str | Path, *, poll_interval: float = 0.1):
    """Process queued runs until the worker is stopped by its operator."""
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    workspace = Path(workspace).resolve(strict=True)
    device = _device()
    work_directory = workspace / ".gleeful"
    if work_directory.is_symlink():
        raise ServiceError("invalid_workspace", "The worker control directory must not be a symlink.")
    work_directory.mkdir(mode=0o700, exist_ok=True)
    lock_path = work_directory / "worker.lock"
    if lock_path.is_symlink():
        raise ServiceError("invalid_workspace", "The workspace lock must not be a symlink.")
    workspace_lock = FileLock(lock_path, timeout=0)
    try:
        with workspace_lock, _device_lock(device):
            worker = _Worker(workspace, device)
            worker.recover()
            print(f"Gleeful Transformer MCP worker ready on {device} for {workspace}", file=sys.stderr, flush=True)
            while True:
                if not worker.process_one():
                    sleep(poll_interval)
    except Timeout as exc:
        raise ServiceError("device_busy", "A worker already holds this workspace or device.", retryable=True) from exc
