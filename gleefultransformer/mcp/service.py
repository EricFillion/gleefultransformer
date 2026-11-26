import json
import time
from operator import itemgetter
from pathlib import Path
from uuid import uuid4

from .errors import ServiceError
from .registry import Registry
from .store import RunStore

_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_REPLY_TIMEOUT_SECONDS = 30


def _public_model(record):
    return {
        "model_id": record["id"],
        "revision": record["revision"],
        "origin": record.get("origin", "registered"),
        "capabilities": record.get("capabilities", ["finetune", "evaluation", "reply"]),
        "tokenizer_revision": record.get("tokenizer_revision") or record.get("tokenizer_sha256"),
        "max_sequence_tokens": record.get("max_sequence_tokens"),
    }


def _public_dataset(record):
    return {
        "dataset_id": record["id"],
        "revision": record.get("revision") or record.get("sha256"),
        "display_name": record.get("display_name") or record["id"],
        "role": record.get("role"),
        "format": "chat_jsonl",
        "row_count": record.get("row_count"),
    }


class RunService:
    def __init__(self, workspace: Path):
        self.registry = Registry(workspace)
        self.store = RunStore(workspace)

    def _exported_models(self):
        return [
            {**record, "id": record.get("id") or record["model_id"], "origin": "export"}
            for record in self.store.published_models()
        ]

    def _exported_model(self, model_id):
        for record in self._exported_models():
            if record["id"] == model_id:
                return record
        return None

    def _model(self, model_id):
        return self._exported_model(model_id) or self.registry.model(model_id)

    def find_models(self):
        models = self.registry.list_models() + self._exported_models()
        return {"items": [_public_model(model) for model in sorted(models, key=itemgetter("id"))]}

    def find_datasets(self):
        datasets = self.registry.list_datasets()
        return {"items": [_public_dataset(dataset) for dataset in sorted(datasets, key=itemgetter("id"))]}

    def check_dataset(self, dataset_id, model_id, max_sequence_tokens=None):
        public = {"dataset_id": dataset_id, "model_id": model_id}
        try:
            dataset = self.registry.dataset(dataset_id)
        except ServiceError as exc:
            if exc.code != "asset_too_large":
                raise
            return {**public, "valid": False, "issues": [exc.to_dict()]}
        model = self._exported_model(model_id) or model_id
        try:
            checked = self.registry.check_dataset(
                dataset, model, max_sequence_tokens,
                require_assistant=dataset.get("role") == "train",
            )
        except ServiceError as exc:
            if exc.code not in {"invalid_dataset", "dataset_too_large"}:
                raise
            return {**public, "valid": False, "issues": [exc.to_dict()]}
        issues = []
        if checked["over_limit"]:
            issues.append({
                "code": "sequence_limit",
                "message": f"{checked['over_limit']} rows exceed max_sequence_tokens.",
                "retryable": False,
            })
        return {**public, **checked, "row_count": checked["rows"], "valid": not issues, "issues": issues}

    def submit_finetune(self, recipe, request_id):
        previous = self.store.lookup_request("finetune", request_id, recipe)
        if previous is not None:
            return previous
        model_id = recipe["model_id"]
        plan = self.registry.plan(
            {**recipe, "mode": "supervised"}, model=self._exported_model(model_id)
        )
        return self.store.submit(
            "finetune", request_id, {"model_id": model_id, **plan}, request_payload=recipe
        )

    def submit_evaluation(self, model_id, dataset_id, request_id, *, max_sequence_tokens=None):
        request = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "batch_size": 1,  # Preserve request IDs from evaluations submitted with the old default.
            "max_sequence_tokens": max_sequence_tokens,
        }
        previous = self.store.lookup_request("evaluation", request_id, request)
        if previous is not None:
            return previous
        plan = self.registry.plan({
            "mode": "evaluation",
            "model_id": model_id,
            "validation_dataset_id": dataset_id,
            **({"max_sequence_tokens": max_sequence_tokens} if max_sequence_tokens is not None else {}),
        }, model=self._exported_model(model_id))
        payload = {
            "model_id": model_id,
            "model": plan["model"],
            "dataset": plan["validation_dataset"],
            "settings": plan["settings"],
            "limits": plan["limits"],
        }
        return self.store.submit(
            "evaluation", request_id, payload, request_payload=request
        )

    def read_run(self, run_id):
        return self.store.read(run_id)

    def find_runs(self, kind=None, state=None, model_id=None, cursor=None, limit=20):
        return self.store.list_runs(kind=kind, state=state, model_id=model_id, cursor=cursor, limit=limit)

    def request_stop(self, run_id):
        return self.store.stop(run_id)

    def reply(self, model_id, messages, max_new_tokens):
        if len(json.dumps(messages, ensure_ascii=False).encode()) > 64_000:
            raise ServiceError("budget_exceeded", "Message history exceeds 64 KB.")
        model = self._model(model_id)
        if "reply" not in _public_model(model)["capabilities"]:
            raise ServiceError("invalid_model", "The selected model does not permit replies.")
        snapshot = self.store.submit(
            "reply", uuid4().hex, {
                "model_id": model_id,
                "model": model,
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "limits": self.registry.limits,
            }, idle_only=True,
        )
        run_id = snapshot["run_id"]
        deadline = time.monotonic() + _REPLY_TIMEOUT_SECONDS
        while snapshot["state"] not in _TERMINAL and time.monotonic() < deadline:
            time.sleep(0.1)
            snapshot = self.store.read(run_id)

        if snapshot["state"] not in _TERMINAL:
            snapshot = self.store.stop(run_id)
            if snapshot.get("stop_action") != "already_terminal":
                raise ServiceError(
                    "reply_timeout",
                    "The worker did not reply in time. Check that it is running and ready, then retry.",
                    True,
                )

        if snapshot["state"] == "succeeded":
            return snapshot["result"]
        error = snapshot.get("error") or {}
        raise ServiceError(
            error.get("code", "reply_unavailable"),
            error.get("message", "The worker could not complete this reply."),
            error.get("retryable", True),
        )
