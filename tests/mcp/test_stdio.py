import asyncio
import json
import math
import os
import subprocess
import sys

import mcp
import pytest

from gleefultransformer.mcp.store import RunStore
from tests.test_gleeful_chat import make_chat_model, tokenizer


@pytest.fixture
def workspace(tmp_path, tokenizer):
    model_path = tmp_path / "base"
    make_chat_model(tokenizer).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    for name, pairs in (
        ("train", [("hello", "world"), ("world", "hello"), ("hello hello", "world")]),
        ("valid", [("canada", "hello world"), ("world", "canada")]),
        ("bad", []),
    ):
        rows = [
            {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}
            for user, answer in pairs
        ]
        content = "".join(json.dumps(row) + "\n" for row in rows)
        (tmp_path / f"{name}.jsonl").write_text(content or '{"messages": [42]}\n', encoding="utf-8")
    (tmp_path / "registry.json").write_text(json.dumps({
        "models": [{"id": "base", "path": "base", "revision": "local-v1"}],
        "datasets": [
            {"id": "train", "path": "train.jsonl", "role": "train"},
            {"id": "valid", "path": "valid.jsonl", "role": "validation"},
            {"id": "bad", "path": "bad.jsonl", "role": "train"},
        ],
    }), encoding="utf-8")
    return tmp_path


async def _call(client, name, arguments, *, error=None):
    reply = await client.call_tool(f"gleeful_{name}", arguments)
    if error:
        assert reply.is_error
        assert error in " ".join(block.text for block in reply.content)
        return None
    assert not reply.is_error, reply.content
    return reply.structured_content


async def _wait(client, run_id):
    async with asyncio.timeout(45):
        while True:
            run = await _call(client, "read_run_snapshot", {"run_id": run_id})
            if run["state"] not in {"queued", "running", "cancel_requested"}:
                assert run["state"] == "succeeded", run
                return run
            await asyncio.sleep(0.1)


def _worker(workspace, logs):
    return subprocess.Popen(
        [sys.executable, "-m", "gleefultransformer.mcp", "worker", "--workspace", str(workspace)],
        env={
            **os.environ, "ACCELERATE_USE_CPU": "1", "OMP_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            # Give the test worker its own device lock directory.
            "TMPDIR": str(workspace),
        },
        stdout=logs, stderr=subprocess.STDOUT,
    )


def test_real_stdio_training_reconnect_evaluate_and_reply(workspace):
    async def scenario():
        parameters = mcp.StdioServerParameters(
            command=sys.executable,
            args=["-m", "gleefultransformer.mcp", "serve", "--workspace", str(workspace)],
            env={
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", str(workspace / "mpl")),
                "HF_HUB_OFFLINE": "1",
            },
        )
        evaluation = {
            "model_id": "base", "dataset_id": "valid", "request_id": "baseline",
            "max_sequence_tokens": 16,
        }
        recipe = {
            "model_id": "base", "train_dataset_id": "train", "validation_dataset_id": "valid",
            "max_steps": 2, "max_sequence_tokens": 16,
        }

        # Exercise the contract over stdio before a worker is started.
        async with mcp.Client(parameters) as client:
            catalog = await client.list_tools()
            assert len(catalog.tools) == 9 and all(tool.output_schema for tool in catalog.tools)
            profiles = await _call(client, "find_model_profiles", {})
            assert profiles["items"][0]["model_id"] == "base"
            assert str(workspace) not in json.dumps(profiles)
            splits = await _call(client, "find_dataset_splits", {})
            assert [(split["dataset_id"], split["role"]) for split in splits["items"]] == [
                ("bad", "train"), ("train", "train"), ("valid", "validation"),
            ]
            checked = await _call(client, "check_dataset_contract", {"dataset_id": "train", "model_id": "base"})
            assert checked["valid"] and checked["row_count"] == 3
            invalid = await _call(client, "check_dataset_contract", {"dataset_id": "bad", "model_id": "base"})
            assert not invalid["valid"] and invalid["issues"]
            await _call(client, "submit_finetune", {
                "recipe": {**recipe, "max_sequence_tokens": 2}, "request_id": "over-limit",
            }, error="sequence_limit")
            cancelled = await _call(client, "submit_dataset_evaluation", {**evaluation, "request_id": "cancel"})
            stopped = await _call(client, "request_run_stop", {"run_id": cancelled["run_id"]})
            assert stopped["state"] == "cancelled" and stopped["stop_action"] == "cancelled_queued"
            base_run = await _call(client, "submit_dataset_evaluation", evaluation)
            assert (await _call(client, "submit_dataset_evaluation", evaluation))["run_id"] == base_run["run_id"]

        log_path = workspace / "worker.log"
        with log_path.open("wb") as logs:
            worker = _worker(workspace, logs)
            try:
                async with mcp.Client(parameters) as client:
                    base = await _wait(client, base_run["run_id"])
                    assert math.isfinite(base["result"]["loss"])
                    ready_message = f"Gleeful Transformer MCP worker ready on cpu for {workspace.resolve()}"
                    assert ready_message in log_path.read_text(encoding="utf-8").splitlines()
                    request = {"recipe": recipe, "request_id": "tune"}
                    run = await _call(client, "submit_finetune", request)
                    assert (await _call(client, "submit_finetune", request))["run_id"] == run["run_id"]
                    await _call(client, "submit_finetune", {
                        **request, "recipe": {**request["recipe"], "max_steps": 3},
                    }, error="request_id_conflict")

                # Observe completion directly while the submitting stdio server is closed.
                local_store = RunStore(workspace)
                async with asyncio.timeout(45):
                    while local_store.read(run["run_id"])["state"] in {"queued", "running"}:
                        await asyncio.sleep(0.1)
                assert local_store.read(run["run_id"])["state"] == "succeeded"

                async with mcp.Client(parameters) as client:
                    trained = await _wait(client, run["run_id"])
                    model_id = trained["artifacts"]["final_model_id"]
                    assert trained["result"]["validation"]["input_model_id"] == "base"
                    assert str(workspace) not in json.dumps(trained)
                    tuned_run = await _call(client, "submit_dataset_evaluation", {
                        **evaluation, "model_id": model_id, "request_id": "tuned",
                    })
                    tuned = await _wait(client, tuned_run["run_id"])
                    assert tuned["result"]["dataset_revision"] == base["result"]["dataset_revision"]
                    assert math.isfinite(tuned["result"]["loss"])
                    history = await _call(client, "find_experiment_runs", {"kind": "finetune"})
                    assert history["items"][0]["run_id"] == run["run_id"]
                    result = await _call(client, "reply_to_messages", {
                        "model_id": model_id, "messages": [{"role": "user", "content": "hello world"}],
                        "max_new_tokens": 2,
                    })
                    assert isinstance(result["text"], str) and result["model_revision"]
            finally:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()

    asyncio.run(scenario())
