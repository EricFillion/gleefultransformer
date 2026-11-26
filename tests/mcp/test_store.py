from concurrent.futures import ThreadPoolExecutor

import pytest

from gleefultransformer.mcp.errors import ServiceError
from gleefultransformer.mcp.store import RunStore


def test_idempotency_uses_the_caller_request_under_concurrent_submission(tmp_path):
    store = RunStore(tmp_path)
    request = {"model_id": "base", "max_steps": 2}

    def submit_once(index):
        return store.submit(
            "finetune",
            "repeat",
            {"model_id": "base", "private_path": f"/local/snapshot-{index}"},
            request_payload=request,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(submit_once, range(8)))
    assert len({run["run_id"] for run in runs}) == 1
    assert store.lookup_request("finetune", "repeat", {"max_steps": 2, "model_id": "base"}) == runs[0]
    assert "payload" not in runs[0]
    assert store.lookup_request("evaluation", "repeat", request) is None
    with pytest.raises(ServiceError) as exc:
        store.lookup_request("finetune", "repeat", {"model_id": "base", "max_steps": 3})
    assert exc.value.code == "request_id_conflict"


def test_claim_fencing_and_stop_settle_before_export_registration(tmp_path):
    store = RunStore(tmp_path)
    current = store.submit("finetune", "first", {"model_id": "base"})
    queued = store.submit("evaluation", "second", {"model_id": "base"})
    run_id = current["run_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(store.claim, ("first-worker", "second-worker")))
    assert sum(claim is not None for claim in claims) == 1
    worker = "first-worker" if claims[0] is not None else "second-worker"

    assert store.stop(queued["run_id"])["stop_action"] == "cancelled_queued"
    assert store.stop(run_id)["stop_action"] == "requested"
    assert store.stop(run_id)["stop_action"] == "already_requested"
    assert store.update(run_id, worker, progress={"current_step": 1})["state"] == "cancel_requested"
    assert store.finish(run_id, worker, model={"model_id": "discarded"})["state"] == "cancelled"
    assert store.published_models() == []
    assert store.stop(run_id)["stop_action"] == "already_terminal"


def test_workspace_restart_and_model_publication_persist(tmp_path):
    store = RunStore(tmp_path)
    abandoned = store.submit("finetune", "crash", {"model_id": "base"})["run_id"]
    store.claim("crashed-worker")
    next_run = store.submit("finetune", "next", {"model_id": "base"})["run_id"]
    reopened = RunStore(tmp_path)

    assert reopened.interrupt_abandoned("new-worker") == 1
    assert reopened.read(abandoned)["state"] == "interrupted"
    assert reopened.read(abandoned)["error"]["retryable"] is True
    with pytest.raises(ServiceError) as exc:
        store.finish(abandoned, "crashed-worker", model={"model_id": "orphan"})
    assert exc.value.code == "run_not_owned"
    assert reopened.claim("new-worker")["run_id"] == next_run

    model = {"model_id": "valid-export", "path": "private/model"}
    with pytest.raises(ValueError):
        reopened.finish(next_run, "new-worker", result={"loss": float("nan")}, model=model)
    assert reopened.published_models() == []
    assert reopened.read(next_run)["state"] == "running"

    finished = reopened.finish(next_run, "new-worker", result={"loss": 1.0}, model=model)
    assert finished["state"] == "succeeded"
    assert finished["artifacts"] == {"final_model_id": model["model_id"]}
    assert reopened.published_models() == [model]
    assert store.read(next_run) == finished
    assert store.list_runs(model_id="valid-export")["items"] == [finished]


def test_pagination_filter_and_idle_admission(tmp_path):
    store = RunStore(tmp_path)
    oldest = store.submit("finetune", "oldest", {"model_id": "base"})
    newest = store.submit("finetune", "newest", {"model_id": "base"})
    page = store.list_runs(kind="finetune", limit=1)
    assert page["items"] == [newest]

    with pytest.raises(ServiceError) as exc:
        store.submit("reply", "busy", {"model_id": "base"}, idle_only=True)
    assert exc.value.code == "device_busy"
    assert exc.value.retryable is True
    assert store.submit("finetune", "oldest", {"model_id": "base"}, idle_only=True) == oldest

    store.submit("finetune", "more-recent", {"model_id": "base"})
    assert store.list_runs(kind="finetune", cursor=page["next_cursor"]) == {
        "items": [oldest],
        "next_cursor": None,
    }
    with pytest.raises(ServiceError) as exc:
        store.list_runs(kind="evaluation", cursor=page["next_cursor"])
    assert exc.value.code == "invalid_cursor"


def test_run_store_refuses_a_symlinked_control_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (workspace / ".gleeful").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ServiceError) as exc:
        RunStore(workspace)
    assert exc.value.code == "path_outside_workspace"
    assert list(elsewhere.iterdir()) == []
