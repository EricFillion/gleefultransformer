import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, PreTrainedTokenizerFast

from gleefultransformer.mcp import registry as registry_module
from gleefultransformer.mcp.errors import ServiceError
from gleefultransformer.mcp.registry import Registry, model_digest


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    model = root / "tiny"
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "question": 1, "answer": 2, "extra": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", model_max_length=16,
        chat_template="{% for msg in messages %}{{msg.content}} {% endfor %}",
    ).save_pretrained(model)
    GPT2Config(vocab_size=4, n_layer=1, n_head=1, n_embd=8, n_positions=16).save_pretrained(model)
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer extra"}]
    (root / "train.jsonl").write_text(json.dumps({"messages": messages}) + "\n", encoding="utf-8")
    (root / "validation.jsonl").write_text(json.dumps({"messages": messages[:1] + [
        {"role": "assistant", "content": "answer"},
    ]}) + "\n", encoding="utf-8")
    (root / "registry.json").write_text(json.dumps({
        "models": [{"id": "small", "path": "tiny", "revision": "local-v1"}],
        "datasets": [
            {"id": "train-set", "path": "train.jsonl", "role": "train"},
            {"id": "held-out", "path": "validation.jsonl", "role": "validation"},
        ],
    }), encoding="utf-8")
    return root


def test_plan_pins_inputs_and_caps_sequence_to_model(workspace):
    registry = Registry(workspace)
    recipe = {"model_id": "small", "train_dataset_id": "train-set", "validation_dataset_id": "held-out", "max_steps": 2}
    plan = registry.plan(recipe)
    assert registry.plan({**recipe, "max_sequence_tokens": 16}) == plan
    assert plan["settings"]["max_sequence_tokens"] == 16
    assert plan["settings"]["precision"] == "auto"
    assert plan["settings"]["eval_steps"] == 2
    assert registry.check_dataset(plan["train_dataset"], "small")["total_tokens"] == 3

    data = workspace / "train.jsonl"
    old_ref = plan["train_dataset"]
    data.write_bytes(data.read_bytes() * 2)
    assert registry.plan(recipe)["train_dataset"]["sha256"] != old_ref["sha256"]
    with pytest.raises(ServiceError, match="changed after it was recorded"):
        registry.check_dataset(old_ref, "small")


def test_rows_over_cap_reported_but_cannot_be_submitted(workspace):
    registry = Registry(workspace)
    assert registry.check_dataset("train-set", "small", 2)["over_limit"] == 1
    with pytest.raises(ServiceError, match="exceed max_sequence_tokens"):
        registry.plan({"model_id": "small", "train_dataset_id": "train-set", "max_steps": 1, "max_sequence_tokens": 2})


def test_model_cache_rechecks_file_stats_and_path_confinement(workspace, monkeypatch):
    model_dir = workspace / "tiny"
    original = registry_module._file_digest
    calls = []

    def record_read(path, byte_limit):
        calls.append(path)
        return original(path, byte_limit)

    monkeypatch.setattr(registry_module, "_file_digest", record_read)
    fingerprint = model_digest(model_dir, workspace=workspace)
    assert calls
    calls.clear()
    assert model_digest(model_dir, workspace=workspace) == fingerprint
    assert not calls

    extra = model_dir / "metadata.txt"
    extra.write_text("a", encoding="utf-8")
    new_fingerprint = model_digest(model_dir, workspace=workspace)
    extra.write_text("b", encoding="utf-8")
    assert model_digest(model_dir, workspace=workspace) != new_fingerprint

    outside = workspace.parent / "outside.txt"
    outside.write_text("c", encoding="utf-8")
    extra.unlink()
    extra.symlink_to(outside)
    with pytest.raises(ServiceError, match="inside the workspace"):
        model_digest(model_dir, workspace=workspace)
