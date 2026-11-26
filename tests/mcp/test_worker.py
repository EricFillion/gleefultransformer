import json

import torch

from gleefultransformer.mcp.service import RunService
from gleefultransformer.mcp.store import RunStore
from gleefultransformer.mcp.worker import _Worker
from tests.test_gleeful_chat import make_chat_model, tokenizer


def test_worker_stops_after_an_optimizer_step(tmp_path, tokenizer, monkeypatch):
    model_path = tmp_path / "base"
    make_chat_model(tokenizer).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps({"messages": messages}) for _ in range(4)) + "\n"
    )
    (tmp_path / "registry.json").write_text(json.dumps({
        "models": [{"id": "base", "path": "base", "revision": "v1"}],
        "datasets": [{"id": "train", "path": "train.jsonl", "role": "train"}],
    }))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    service = RunService(tmp_path)
    recipe = {
        "model_id": "base", "train_dataset_id": "train", "max_steps": 3,
        "max_sequence_tokens": 8,
    }
    run_id = service.submit_finetune(recipe, "stop-one-step")["run_id"]
    worker = _Worker(tmp_path, torch.device("cpu"))
    observer_factory = worker._observer

    def stop_after_first_step(current_run_id, provenance):
        observe = observer_factory(current_run_id, provenance)

        def request_stop(event, fields):
            if event == "step" and fields["current_step"] == 1:
                worker.store.stop(current_run_id)
            return observe(event, fields)

        return request_stop

    monkeypatch.setattr(worker, "_observer", stop_after_first_step)
    assert worker.process_one()
    snapshot = service.read_run(run_id)
    assert snapshot["state"] == "cancelled"
    assert snapshot["progress"] == {"current_step": 1, "total_steps": 3}
    assert service.store.published_models() == []
    assert not (worker.artifacts / run_id / "model").exists()


def test_restart_removes_unpublished_export_directories(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.submit("finetune", "crash-during-export", {})["run_id"]
    assert store.claim("interrupted-worker")["run_id"] == run_id
    worker = _Worker(tmp_path, torch.device("cpu"))
    run_directory = worker.artifacts / run_id
    for name in ("export.staging", "model", "training"):
        (run_directory / name).mkdir(parents=True)

    worker.recover()

    assert store.read(run_id)["state"] == "interrupted"
    assert not (run_directory / "export.staging").exists()
    assert not (run_directory / "model").exists()
    assert (run_directory / "training").is_dir()
