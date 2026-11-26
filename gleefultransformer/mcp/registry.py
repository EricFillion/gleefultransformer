"""Workspace-scoped model and dataset references for MCP recipes."""

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from jinja2 import TemplateError

from .errors import ServiceError


_MIB = 1024 ** 2
_GIB = 1024 ** 3
_REGISTRY_BYTES = 512 * 1024
_MAX_RECORDS = 256
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
_DATASET_ROLES = {"train", "validation", "test"}
_MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}
_MODEL_OPERATIONS = ("chat", "supervised", "evaluation")
_PUBLIC_OPERATIONS = {"chat": "reply", "supervised": "finetune", "evaluation": "evaluation"}
_LIMITS = {
    # Each pair contains the default and the largest workspace override.
    "max_dataset_bytes": (32 * _MIB, 256 * _MIB),
    "max_dataset_rows": (20_000, 100_000),
    "max_row_bytes": (256 * 1024, _MIB),
    "max_messages_per_row": (64, 128),
    "max_sequence_tokens": (4096, 32_768),
    "max_steps": (128, 10_000),
    "max_model_bytes": (64 * _GIB, 256 * _GIB),
    "max_model_files": (2048, 8192),
    "max_output_bytes": (64 * _GIB, 256 * _GIB),
}
_COMMON_FIELDS = {"mode", "model_id", "validation_dataset_id", "max_sequence_tokens"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bounded_int(value, name, minimum, maximum, *, code="invalid_recipe"):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ServiceError(code, f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _confined_path(workspace, value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ServiceError("invalid_registry", "Registered paths must be nonempty file paths.")
    try:
        path = (workspace / value).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ServiceError("asset_unavailable", "A registered path could not be opened.") from exc
    if not path.is_relative_to(workspace):
        raise ServiceError("path_outside_workspace", "Registered paths must stay inside the workspace.")
    return path


def _file_digest(path, byte_limit):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while data := handle.read(min(_MIB, byte_limit - size + 1)):
            size += len(data)
            if size > byte_limit:
                raise ServiceError("asset_too_large", "A registered asset exceeds its byte limit.")
            digest.update(data)
    return digest.hexdigest(), size


def dataset_digest(path, *, max_bytes=_LIMITS["max_dataset_bytes"][0]):
    """Hash the exact JSONL bytes through a bounded, streaming read."""
    try:
        return _file_digest(Path(path), max_bytes)[0]
    except OSError as exc:
        raise ServiceError("asset_unavailable", "Could not read the registered dataset.") from exc


def _file_signature(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=8)
def _model_content_digest(manifest):
    digest = hashlib.sha256()
    for relative_name, path, signature in manifest:
        file_sha, size = _file_digest(path, signature[2])
        if size != signature[2] or _file_signature(path) != signature:
            raise ServiceError("asset_changed", "A model file changed while its identity was being checked.")
        digest.update(_json_bytes([relative_name, size, file_sha]))
        digest.update(b"\n")
    return digest.hexdigest()


def model_digest(
    path, *, workspace=None, max_bytes=_LIMITS["max_model_bytes"][0],
    max_files=_LIMITS["max_model_files"][0],
):
    """Content identity of a local checkpoint, including weights and tokenizer files."""
    try:
        root = Path(path).resolve(strict=True)
        workspace = Path(workspace).resolve(strict=True) if workspace is not None else root
        if not root.is_dir() or not root.is_relative_to(workspace):
            raise ServiceError("path_outside_workspace", "The model must be a workspace directory.")

        def on_walk_error(error):
            raise error

        manifest = []
        remaining = max_bytes
        for directory, dirs, filenames in os.walk(root, onerror=on_walk_error):
            folder = Path(directory)
            dirs.sort()
            if any((folder / name).is_symlink() for name in dirs):
                raise ServiceError("invalid_registry", "Model snapshots cannot contain symlinked directories.")
            for filename in sorted(filenames):
                if len(manifest) >= max_files:
                    raise ServiceError("asset_too_large", "The model snapshot contains too many files.")
                entry = folder / filename
                resolved = _confined_path(workspace, entry)
                if not resolved.is_file():
                    raise ServiceError("invalid_registry", "A model snapshot entry is not a regular file.")
                signature = _file_signature(resolved)
                remaining -= signature[2]
                if remaining < 0:
                    raise ServiceError("asset_too_large", "The model snapshot exceeds its byte limit.")
                manifest.append((entry.relative_to(root).as_posix(), resolved, signature))
        if not manifest:
            raise ServiceError("invalid_registry", "The model snapshot directory is empty.")
        return _model_content_digest(tuple(manifest))
    except OSError as exc:
        raise ServiceError("asset_unavailable", "Could not read the registered model snapshot.") from exc


def tokenizer_digest(tokenizer):
    """Identify the tokenizer contents and the default template used for chat."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    options = {
        "template": tokenizer.get_chat_template(),
        "special_tokens": {
            role: [str(token) for token in value] if isinstance(value, (list, tuple)) else str(value)
            for role, value in tokenizer.special_tokens_map.items()
        },
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
    }
    if backend is not None:
        backend_data = json.loads(backend.to_str())
        # These mutable request settings are represented by the options above.
        backend_data.pop("padding", None)
        backend_data.pop("truncation", None)
        options["backend"] = backend_data
    else:
        options["vocabulary"] = tokenizer.get_vocab()
    return hashlib.sha256(_json_bytes(options)).hexdigest()


@lru_cache(maxsize=4)
def _load_tokenizer(path, snapshot_digest):
    from transformers import AutoConfig, AutoTokenizer

    from gleefultransformer._core.utils.init.get_components import get_tokenizer

    try:
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        tokenizer = get_tokenizer(path, tokenizer, trust_remote_code=False)
        if not tokenizer.get_chat_template():
            raise ValueError("no chat template")
    except (OSError, ValueError, TypeError, ImportError, TemplateError) as exc:
        raise ServiceError(
            "invalid_model", "The model must have a local configuration and tokenizer with a chat template."
        ) from exc

    model_config = config.get_text_config()
    caps = [getattr(model_config, "max_position_embeddings", None), tokenizer.model_max_length]
    caps = [cap for cap in caps if type(cap) is int and 0 < cap < 10_000_000]
    return tokenizer, min(caps, default=None), tokenizer_digest(tokenizer)


class Registry:
    """A process-local view of the operator's workspace registry.json."""

    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        manifest = self.workspace / "registry.json"
        if manifest.exists() or manifest.is_symlink():
            manifest = _confined_path(self.workspace, manifest)
            if not manifest.is_file():
                raise ServiceError("invalid_registry", "registry.json must be a file.")
            try:
                with manifest.open("rb") as handle:
                    raw = handle.read(_REGISTRY_BYTES + 1)
                if len(raw) > _REGISTRY_BYTES:
                    raise ServiceError("invalid_registry", "registry.json exceeds 512 KiB.")
                document = json.loads(raw)
            except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                raise ServiceError("invalid_registry", "Could not read registry.json as JSON.") from exc
        else:
            document = {}
        if not isinstance(document, dict) or document.keys() - {"version", "models", "datasets", "limits"}:
            raise ServiceError("invalid_registry", "registry.json must contain models, datasets, or limits.")
        if type(document.get("version", 1)) is not int or document.get("version", 1) != 1:
            raise ServiceError("invalid_registry", "Only registry version 1 is supported.")
        limits = document.get("limits", {})
        retired_limits = {"max_batch_size", "max_gradient_accumulation_steps"}
        if not isinstance(limits, dict) or limits.keys() - _LIMITS.keys() - retired_limits:
            raise ServiceError("invalid_registry", "Unknown registry limit.")
        self.limits = {
            name: _bounded_int(limits.get(name, default), name, 1, maximum, code="invalid_registry")
            for name, (default, maximum) in _LIMITS.items()
        }
        self._models = self._read_records(document.get("models", []), "model")
        self._datasets = self._read_records(document.get("datasets", []), "dataset")

    @staticmethod
    def _read_records(records, kind):
        if not isinstance(records, list) or len(records) > _MAX_RECORDS:
            raise ServiceError("invalid_registry", f"The {kind} list may contain up to {_MAX_RECORDS} records.")
        result = {}
        for record in records:
            if not isinstance(record, dict):
                raise ServiceError("invalid_registry", f"Every {kind} record must be an object.")
            required = {"id", "path", "revision"} if kind == "model" else {"id", "path"}
            optional = {"operations"} if kind == "model" else {"role"}
            if required - record.keys() or record.keys() - (required | optional):
                raise ServiceError("invalid_registry", f"Invalid fields in a {kind} record.")
            asset_id = record["id"]
            if not isinstance(asset_id, str) or _ID.fullmatch(asset_id) is None or asset_id in result:
                raise ServiceError("invalid_registry", f"Every {kind} ID must be valid and unique.")
            if not isinstance(record["path"], str) or not record["path"] or len(record["path"]) > 4096:
                raise ServiceError("invalid_registry", f"Invalid path for {kind} {asset_id}.")
            normalized = dict(record)
            if kind == "model":
                revision = record["revision"]
                if not isinstance(revision, str) or not revision.strip() or len(revision) > 256:
                    raise ServiceError("invalid_registry", "Model revision must be a nonempty pin label.")
                operations = record.get("operations", list(_MODEL_OPERATIONS))
                if (not isinstance(operations, list) or not operations
                        or not all(isinstance(op, str) and op in _MODEL_OPERATIONS for op in operations)):
                    raise ServiceError("invalid_registry", f"Invalid operations for model {asset_id}.")
                normalized["operations"] = sorted(set(operations))
                normalized["capabilities"] = [_PUBLIC_OPERATIONS[op] for op in normalized["operations"]]
            elif "role" in record and (
                not isinstance(record["role"], str) or record["role"] not in _DATASET_ROLES
            ):
                raise ServiceError("invalid_registry", "Dataset role must be train, validation, or test.")
            result[asset_id] = normalized
        return result

    def _resolved_record(self, record, kind):
        path = _confined_path(self.workspace, record["path"])
        if kind == "model" and (not path.is_dir() or path == self.workspace):
            raise ServiceError("invalid_registry", "A model must be in its own workspace directory.")
        if kind == "dataset" and (not path.is_file() or path.suffix.lower() != ".jsonl"):
            raise ServiceError("invalid_registry", "A dataset must be a workspace JSONL file.")
        return {**record, "path": str(path)}

    @staticmethod
    def _lookup(records, asset_id, kind):
        if not isinstance(asset_id, str) or asset_id not in records:
            raise ServiceError("not_found", f"No registered {kind} has the requested ID.")
        return records[asset_id]

    def list_models(self):
        return [self._resolved_record(record, "model") for record in self._models.values()]

    def list_datasets(self):
        return [self._resolved_record(record, "dataset") for record in self._datasets.values()]

    def _pinned_model(self, record):
        record = self._resolved_record(record, "model")
        actual = model_digest(record["path"], workspace=self.workspace,
                              max_bytes=self.limits["max_model_bytes"], max_files=self.limits["max_model_files"])
        if record.get("sha256", actual) != actual:
            raise ServiceError("asset_changed", "The model snapshot changed after it was recorded.")
        _, context_limit, tokenizer_sha = _load_tokenizer(record["path"], actual)
        return {
            **record, "sha256": actual, "tokenizer_sha256": tokenizer_sha,
            "tokenizer_revision": f"sha256:{tokenizer_sha}", "max_sequence_tokens": context_limit,
        }

    def model(self, model_id):
        return self._pinned_model(self._lookup(self._models, model_id, "model"))

    def dataset(self, dataset_id):
        record = self._resolved_record(self._lookup(self._datasets, dataset_id, "dataset"), "dataset")
        return {**record, "sha256": dataset_digest(record["path"], max_bytes=self.limits["max_dataset_bytes"])}

    def check_dataset(self, dataset, model, sequence_cap=None, *, require_assistant=True):
        """Validate every bounded JSONL row against the pinned model's chat template."""
        dataset = self._resolved_record(dataset, "dataset") if isinstance(dataset, dict) else self.dataset(dataset)
        model = self._pinned_model(model) if isinstance(model, dict) else self.model(model)
        return self._check_pinned_dataset(dataset, model, sequence_cap, require_assistant)

    def _check_pinned_dataset(self, dataset, model, sequence_cap, require_assistant):
        cap = self._sequence_cap(model, sequence_cap)
        tokenizer, _, _ = _load_tokenizer(model["path"], model["sha256"])
        data_sha = hashlib.sha256()
        count = byte_count = total = longest = too_long = 0
        shortest = None
        try:
            with Path(dataset["path"]).open("rb") as handle:
                for row_number in range(1, self.limits["max_dataset_rows"] + 2):
                    row = handle.readline(self.limits["max_row_bytes"] + 1)
                    if not row:
                        break
                    byte_count += len(row)
                    if len(row) > self.limits["max_row_bytes"]:
                        raise ServiceError("dataset_too_large", f"Dataset row {row_number} exceeds the byte limit.")
                    if byte_count > self.limits["max_dataset_bytes"] or row_number > self.limits["max_dataset_rows"]:
                        raise ServiceError("dataset_too_large", "The dataset exceeds the configured size or row limit.")
                    data_sha.update(row)
                    messages = self._messages(row, row_number, require_assistant)
                    try:
                        tokens = tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
                        )["input_ids"]
                    except (ValueError, TypeError, KeyError, RuntimeError, TemplateError) as exc:
                        raise ServiceError("invalid_dataset", f"Dataset row {row_number} is incompatible with the chat template.") from exc
                    size = len(tokens)
                    if size < 2:
                        raise ServiceError("invalid_dataset", f"Dataset row {row_number} has fewer than two tokens.")
                    count += 1
                    total += size
                    shortest = size if shortest is None else min(shortest, size)
                    longest = max(longest, size)
                    too_long += size > cap
        except OSError as exc:
            raise ServiceError("asset_unavailable", "Could not read the registered dataset.") from exc
        if not count:
            raise ServiceError("invalid_dataset", "The dataset contains no conversations.")
        actual = data_sha.hexdigest()
        if dataset.get("sha256", actual) != actual:
            raise ServiceError("asset_changed", "The dataset changed after it was recorded.")
        return {"dataset_id": dataset["id"], "sha256": actual, "rows": count,
                "bytes": byte_count, "min_tokens": shortest, "max_tokens": longest,
                "total_tokens": total, "over_limit": too_long, "max_sequence_tokens": cap}

    def _messages(self, row, row_number, require_assistant):
        try:
            value = json.loads(row.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ServiceError("invalid_dataset", f"Dataset row {row_number} is not valid UTF-8 JSON.") from exc
        messages = value.get("messages") if isinstance(value, dict) else None
        if not isinstance(messages, list) or not 1 <= len(messages) <= self.limits["max_messages_per_row"]:
            raise ServiceError("invalid_dataset", f"Dataset row {row_number} requires a bounded messages list.")
        if not all(isinstance(message, dict) and isinstance(message.get("role"), str)
                   and message["role"] in _MESSAGE_ROLES and isinstance(message.get("content"), str)
                   and message["content"].strip() for message in messages):
            raise ServiceError("invalid_dataset", f"Dataset row {row_number} has an invalid message role or content.")
        if require_assistant and not any(message["role"] == "assistant" for message in messages):
            raise ServiceError("invalid_dataset", f"Training row {row_number} requires an assistant message.")
        return messages

    def _sequence_cap(self, model, requested):
        limit = self.limits["max_sequence_tokens"]
        limit = min(limit, model["max_sequence_tokens"] or limit)
        return _bounded_int(min(512, limit) if requested is None else requested,
                            "max_sequence_tokens", 2, limit)

    def plan(self, recipe, *, model=None):
        """Normalize a supervised or evaluation request and pin its inputs by content."""
        if not isinstance(recipe, dict):
            raise ServiceError("invalid_recipe", "A recipe must be an object.")
        mode = recipe.get("mode", "supervised")
        if mode not in ("supervised", "evaluation"):
            raise ServiceError("invalid_recipe", "Supported recipe modes are supervised and evaluation.")
        fields = _COMMON_FIELDS | ({"train_dataset_id", "max_steps"} if mode == "supervised" else set())
        if recipe.keys() - fields:
            raise ServiceError("invalid_recipe", "The selected recipe contains unsupported settings.")
        model_id = recipe.get("model_id")
        if model is not None and model.get("id") != model_id:
            raise ServiceError("invalid_recipe", "The selected model does not match the recipe model ID.")
        model = self._pinned_model(model) if model is not None else self.model(model_id)
        if mode not in model.get("operations", _MODEL_OPERATIONS):
            raise ServiceError("invalid_recipe", "The selected model does not permit this operation.")
        settings = {
            "max_sequence_tokens": self._sequence_cap(model, recipe.get("max_sequence_tokens")),
            "eval_batch_size": 1,
            "seed": 42,
        }
        if mode == "supervised":
            max_steps = _bounded_int(recipe.get("max_steps"), "max_steps", 1, self.limits["max_steps"])
            interval = min(8, max_steps)
            settings.update(
                max_steps=max_steps, batch_size=1, gradient_accumulation_steps=1,
                learning_rate=2e-5, precision="auto", log_steps=1, checkpoint_steps=interval,
                eval_steps=interval if recipe.get("validation_dataset_id") is not None else 0,
                save_best=False,
            )

        validation_id = recipe.get("validation_dataset_id")
        if mode == "evaluation" and validation_id is None:
            raise ServiceError("invalid_recipe", "Evaluation requires validation_dataset_id.")
        validation = self.dataset(validation_id) if validation_id is not None else None
        allowed_roles = {None, "validation", "test"} if mode == "evaluation" else {None, "validation"}
        if validation is not None and validation.get("role") not in allowed_roles:
            raise ServiceError("invalid_recipe", "The dataset role is incompatible with evaluation in this recipe.")
        train_id = recipe.get("train_dataset_id")
        train = self.dataset(train_id) if mode == "supervised" else None
        if train is not None and train.get("role") not in {None, "train"}:
            raise ServiceError("invalid_recipe", "The training dataset must have the train role.")
        if train is not None and validation is not None and train["sha256"] == validation["sha256"]:
            raise ServiceError("invalid_recipe", "Use a separate held-out dataset for validation.")

        for name, dataset in (("train", train), ("validation", validation)):
            if dataset is None:
                continue
            stats = self._check_pinned_dataset(dataset, model, settings["max_sequence_tokens"], name == "train")
            if stats["over_limit"]:
                raise ServiceError("sequence_limit", f"{stats['over_limit']} {name} rows exceed max_sequence_tokens.")

        return {
            "model": model,
            "train_dataset": train,
            "validation_dataset": validation,
            "settings": settings,
            "limits": dict(self.limits),
        }
