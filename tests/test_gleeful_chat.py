"""Exercise the public chat API with small, offline models."""

import json
import math
from dataclasses import replace
from inspect import signature

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BitsAndBytesConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPTQConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from gleefultransformer import (
    CHATCallArgs,
    CHATResult,
    CHATTokArgs,
    GleefulChat,
    GleefulEvalArgs,
    GleefulTrainArgs,
    RankerResult,
    SearchCallArgs,
)
from gleefultransformer._core.gleeful_tasks import chat_adapters
from gleefultransformer._core.gleeful_tasks.chat_stream_handlers import stream_events
from gleefultransformer._core.loops.train_loop import debug_hook_post_checkpoint


def make_chat_model(tokenizer):
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=32, n_layer=1, n_head=1, n_embd=8,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))


@pytest.fixture
def tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "canada": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="[EOS]", model_max_length=32,
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )


def test_auto_cpu_training_upcasts_local_fp16_checkpoint(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(
        "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_device",
        lambda: torch.device("cpu"),
    )
    checkpoint = tmp_path / "fp16-model"
    model = make_chat_model(tokenizer)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.02)
    model.half().save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    chat = GleefulChat(model_name=checkpoint)
    assert chat.model.dtype == torch.float16
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    for path, answer in [(train_path, "world"), (eval_path, "canada")]:
        path.write_text(json.dumps({"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": answer},
        ]}) + "\n", encoding="utf-8")

    result = chat.train(
        train_path,
        eval_path=eval_path,
        args=GleefulTrainArgs(
            out_dir=str(tmp_path), run_name="cpu-auto", precision="auto",
            max_steps=1, bs=1, eval_steps=1, log_steps=1,
        ),
        tok_args=CHATTokArgs(max_len=8),
    )

    assert chat.model.dtype == torch.float32
    assert math.isfinite(result.final_eval_loss)
    report = json.loads((tmp_path / "cpu-auto" / "precision.json").read_text())
    assert report["training_precision"] == "fp32"
    assert report["weight_dtypes"] == ["float32"]


def test_chat_passes_roles_through_to_the_tokenizer(tokenizer, monkeypatch):
    chat = GleefulChat(model_name=make_chat_model(tokenizer), tokenizer=tokenizer)
    assert "model_type" not in signature(GleefulChat).parameters
    assert not hasattr(chat, "model_type")

    messages = [{"role": "developer", "content": "hello"}, {"role": "user", "content": "world"}]
    apply_chat_template = tokenizer.apply_chat_template
    seen = []

    def capture_template(conversation, **kwargs):
        seen.append(conversation)
        return apply_chat_template(conversation, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", capture_template)
    settings = CHATCallArgs(min_len=1, max_len=1, temp=0)
    assert isinstance(chat(messages, args=settings), CHATResult)
    assert seen == [messages]


def test_chat_stream_parser_starts_fresh_for_every_reply(tokenizer, monkeypatch):
    tokenizer.backend_tokenizer.add_special_tokens(["<think>", "</think>"])
    tokenizer.chat_template += "{# <think> </think> #}"
    model = make_chat_model(tokenizer)
    chat = GleefulChat(model_name=model, tokenizer=tokenizer)
    replies = iter([("<think>", "world"), ("hello",), ("hello",)])

    def generate(*, input_ids, streamer, **kwargs):
        streamer.put(input_ids.cpu())
        for token in next(replies):
            streamer.put(torch.tensor([tokenizer.convert_tokens_to_ids(token)]))
        streamer.end()

    monkeypatch.setattr(model, "generate", generate)
    first = list(chat.stream("hello"))
    assert any(event.marker == "thinking" and event.text.strip() == "world" for event in first)
    second = list(chat.stream("hello"))
    rendered = "".join(event.text for event in second if event.marker == "text")
    assert rendered.strip() == "hello"
    assert chat("hello").text == rendered


def test_harmony_stream_keeps_reasoning_out_of_the_final_reply():
    chunks = [
        "<|chan", "nel|>analysis<|message|>draft", "<|end|><|start|>assistant",
        "<|channel|>fi", "nal<|message|>final reply<|return|>",
    ]
    events = list(stream_events(chunks, "harmony", ()))
    assert "".join(event.text for event in events if event.marker == "thinking") == "draft"
    assert "".join(event.text for event in events if event.marker == "text") == "final reply"


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("answer_region", ["RESPONSE", "TEXT"])
def test_cohere_stream_respects_the_thinking_prefix(
    tokenizer, monkeypatch, thinking, answer_region,
):
    start_answer, end_answer = f"<|START_{answer_region}|>", f"<|END_{answer_region}|>"
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|CHATBOT_TOKEN|>", "<|START_THINKING|>", "<|END_THINKING|>", start_answer, end_answer,
    ]})
    tokenizer.chat_template += "<|CHATBOT_TOKEN|><|START_THINKING|>"
    if not thinking:
        tokenizer.chat_template += "<|END_THINKING|>"
    model = make_chat_model(tokenizer)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16, mean_resizing=False)
    chat = GleefulChat(model_name=model, tokenizer=tokenizer)
    assert chat.model.get_input_embeddings().num_embeddings == 16
    generated = ["world", "<|END_THINKING|>"] if thinking else []
    generated += [start_answer, "canada", end_answer]

    def generate(*, input_ids, streamer, **kwargs):
        streamer.put(input_ids.cpu())
        for token in generated:
            streamer.put(torch.tensor([tokenizer.convert_tokens_to_ids(token)]))
        streamer.end()

    monkeypatch.setattr(model, "generate", generate)
    events = list(chat.stream("hello"))
    reasoning = "".join(event.text for event in events if event.marker == "thinking")
    assert reasoning.strip() == ("world" if thinking else "")
    assert chat("hello").text.strip() == "canada"


def test_trainer_resume_replays_remaining_shard_batches(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr(
        "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_device",
        lambda: torch.device("cpu"),
    )
    data = tmp_path / "data.jsonl"
    messages = [
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        for prompt, reply in [("hello", "world"), ("hello", "canada"),
                              ("world", "hello"), ("canada", "hello")]
    ]
    data.write_text("".join(json.dumps({"messages": row}) + "\n" for row in messages * 2),
                    encoding="utf-8")
    first = GleefulChat(model_name=make_chat_model(tokenizer).bfloat16(), tokenizer=tokenizer)
    tokenized = tmp_path / "tokenized"
    first.tok(str(data), str(tokenized), CHATTokArgs(max_len=8))
    first.train(
        tokenized, args=GleefulTrainArgs(
            out_dir=str(tmp_path), run_name="initial", checkpoint_steps=2,
            max_steps=4, gas=2, eval_steps=0, log_steps=0, precision="fp32",
        ),
    )
    checkpoint = tmp_path / "initial" / "checkpoint-2"
    precision = json.loads((tmp_path / "initial" / "precision.json").read_text())
    assert precision["training_precision"] == "fp32"
    assert precision["weight_dtypes"] == ["float32"]
    assert json.loads((checkpoint / "gleeful_resume.json").read_text())["precision"] == precision
    tracker_path = checkpoint / "gleeful_tracker_state.json"
    tracker = json.loads(tracker_path.read_text())
    assert "target_tokens" not in tracker
    assert "target_tokens_per_second" not in tracker
    tracker.update(target_tokens=4, target_tokens_per_second=2.0)
    tracker_path.write_text(json.dumps(tracker), encoding="utf-8")
    for suffix in ("resume.json", "tracker_state.json"):
        (checkpoint / f"gleeful_{suffix}").rename(checkpoint / f"legacy_{suffix}")
    resumed = GleefulChat(model_name=None, tokenizer=str(checkpoint))
    resumed.train(resume_path=checkpoint)

    resumed_weights = resumed.model.state_dict()
    assert all(
        torch.equal(weights, resumed_weights[name])
        for name, weights in first.model.state_dict().items()
    )


def test_gleeful_chat_passes_search_context_without_mutating_messages(tokenizer):
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=6, n_positions=32, n_layer=1, n_head=1, n_embd=8,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))
    calls = []

    def search(text, args):
        calls.append((text, args.limit))
        return [RankerResult(text="hello canada", score=1.0,
                             best_sentence="hello canada", metadata={"id": 1})]

    chat = GleefulChat(model_name=model, tokenizer=tokenizer, gleeful_search=search)
    messages = [{"role": "user", "content": "hello world"}]
    settings = CHATCallArgs(min_len=64, max_len=64, temp=0,
                            search_args=SearchCallArgs(limit=2))
    streamed = list(chat.stream(messages, args=settings))
    from_messages = chat(messages, args=settings)
    from_prompt = chat("hello world", args=settings)

    assert calls == [("hello world", 2)] * 3
    assert streamed[0].marker == "search"
    assert streamed[1].marker == "search_result"
    assert streamed[1].payload["metadata"] == {"id": 1}
    assert isinstance(from_messages, CHATResult)
    assert isinstance(from_prompt, CHATResult)
    assert from_messages.text == from_prompt.text == "".join(
        event.text for event in streamed if event.marker == "text"
    )
    assert messages == [{"role": "user", "content": "hello world"}]


@pytest.fixture
def chat_adapter_data(tmp_path, tokenizer):
    base = tmp_path / "base"
    LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    )).save_pretrained(base)
    tokenizer.save_pretrained(base)
    train_data = tmp_path / "chat.jsonl"
    with train_data.open("w") as file:
        for answer in ("world", "canada"):
            file.write(json.dumps({"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": answer},
            ]}) + "\n")
    return str(base), str(train_data)


def lora_config(peft):
    return peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM, r=4, lora_alpha=12,
        target_modules=["q_proj", "v_proj"], use_rslora=True,
    )


def nf4_config(compute_dtype=torch.bfloat16):
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )


def assert_nf4_layers(model, linear4bit):
    layers = [layer for layer in model.modules() if isinstance(layer, linear4bit)]
    assert layers
    assert all(layer.weight.quant_type == "nf4" for layer in layers)


@pytest.mark.parametrize("quantized", [False, True], ids=["lora", "nf4"])
def test_chat_adapter_checkpoint_and_reload(quantized, chat_adapter_data, tmp_path, monkeypatch):
    if quantized and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        pytest.skip("NF4 adapter round trip requires a CUDA GPU with BF16 support")
    peft = pytest.importorskip("peft")
    quantization = None
    if quantized:
        bitsandbytes = pytest.importorskip("bitsandbytes")
        quantization = nf4_config()
    else:
        monkeypatch.setattr(
            "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_device",
            lambda: torch.device("cpu"),
        )

    base, train_data = chat_adapter_data
    runs = tmp_path / "runs"
    args = GleefulTrainArgs(
        out_dir=str(runs), run_name="first", max_steps=2, checkpoint_steps=1, log_steps=0,
    )
    chat = GleefulChat(
        model_name=base, adapter=lora_config(peft), quantization="nf4" if quantized else None
    )
    assert isinstance(chat.model, peft.PeftModel)
    assert chat.model.get_input_embeddings().num_embeddings == 16
    if quantized:
        assert_nf4_layers(chat.model, bitsandbytes.nn.Linear4bit)
    config = chat.model.peft_config["default"]
    assert (config.r, config.lora_alpha, config.use_rslora) == (4, 12, True)
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    trainable = [name for name, param in chat.model.named_parameters() if param.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    initial = {name: param.detach().cpu().clone() for name, param in chat.model.named_parameters()}

    class Interrupted(Exception):
        pass

    def interrupt():
        raise Interrupted

    with debug_hook_post_checkpoint.set(interrupt), pytest.raises(Interrupted):
        chat.train(train_data, args=args)
    assert any(
        not torch.equal(initial[name], param.detach().cpu())
        for name, param in chat.model.named_parameters() if name in trainable
    )
    for name, param in chat.model.named_parameters():
        if name not in trainable:
            assert torch.equal(initial[name], param.detach().cpu())
    del chat

    checkpoint = runs / "first" / "checkpoint-1"
    assert (checkpoint / "adapter_model.safetensors").is_file()
    if quantized:
        saved = json.loads((checkpoint / "config.json").read_text())
        assert saved["quantization_config"] == quantization.to_dict()

    # The legacy checkpoint alias also contains the current run's native Trainer checkpoint.
    legacy_checkpoint = runs / "first" / "checkpoint"
    assert legacy_checkpoint.resolve() == checkpoint
    resumed = GleefulChat(model_name=None, tokenizer=legacy_checkpoint)
    result = resumed.train(args=replace(args, run_name="resumed"), resume_path=legacy_checkpoint)
    assert math.isfinite(result.final_train_loss)
    if quantized:
        assert_nf4_layers(resumed.model, bitsandbytes.nn.Linear4bit)
    resumed_checkpoint = runs / "resumed" / "checkpoint-2"
    optimizer = torch.load(resumed_checkpoint / "optimizer.pt", weights_only=True)
    assert len(optimizer["param_groups"][0]["params"]) == len(trainable)
    assert {state["step"].item() for state in optimizer["state"].values()} == {2}

    export = tmp_path / "adapter"
    resumed.save(export)
    expected = {
        name: param.detach().cpu().clone()
        for name, param in resumed.model.named_parameters() if param.requires_grad
    }
    assert (export / "adapter_model.safetensors").is_file()
    assert not (export / "model.safetensors").exists()
    if quantized:
        saved = json.loads((export / "config.json").read_text())
        assert saved["quantization_config"] == quantization.to_dict()
    del resumed

    loaded = GleefulChat(model_name=export)
    if quantized:
        assert_nf4_layers(loaded.model, bitsandbytes.nn.Linear4bit)
        assert loaded.config.quantization_config.to_dict() == quantization.to_dict()
    assert loaded.model.peft_config["default"].use_rslora
    actual = {
        name: param.detach().cpu()
        for name, param in loaded.model.named_parameters() if name in expected
    }
    assert actual.keys() == expected.keys()
    assert all(torch.equal(param, actual[name]) for name, param in expected.items())
    assert math.isfinite(loaded.eval(
        train_data, args=GleefulEvalArgs(out_dir=str(tmp_path / "eval"))
    ).loss)
    assert isinstance(loaded("hello", args=CHATCallArgs(min_len=1, max_len=2)), CHATResult)


def test_chat_rejects_unsupported_adapter_and_quantization_configs(chat_adapter_data):
    peft = pytest.importorskip("peft")
    base, _ = chat_adapter_data
    with pytest.raises(TypeError, match="LoraConfig"):
        GleefulChat(model_name=base, adapter=peft.IA3Config(
            task_type=peft.TaskType.CAUSAL_LM,
            target_modules=["q_proj"], feedforward_modules=[],
        ))
    with pytest.raises(TypeError, match="BitsAndBytesConfig"):
        GleefulChat(model_name=base, adapter=lora_config(peft), quantization=GPTQConfig(bits=4))
    with pytest.raises(ValueError, match="nf4"):
        GleefulChat(model_name=base, adapter=lora_config(peft), quantization="int8")


def test_chat_rejects_adapter_overrides_and_unsupported_saved_quantization(chat_adapter_data, tmp_path):
    peft = pytest.importorskip("peft")
    base, _ = chat_adapter_data
    export = tmp_path / "adapter"
    GleefulChat(model_name=base, adapter=lora_config(peft)).save(export)

    with pytest.raises(ValueError, match="saved adapter"):
        GleefulChat(model_name=export, adapter=lora_config(peft))
    with pytest.raises(ValueError, match="saved adapter"):
        GleefulChat(model_name=export, quantization="nf4")
    (export / "config.json").write_text(json.dumps({
        "quantization_config": {"quant_method": "hqq", "quant_config": {"weight_quant_params": {"nbits": 4}}}
    }))
    with pytest.raises(ValueError, match="(?i)bitsandbytes"):
        GleefulChat(model_name=export)


def test_chat_nf4_requires_lora_and_cuda(chat_adapter_data, monkeypatch):
    peft = pytest.importorskip("peft")
    base, _ = chat_adapter_data
    with pytest.raises(ValueError, match="(?i)lora"):
        GleefulChat(model_name=base, quantization="nf4")

    monkeypatch.setattr(
        "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_device",
        lambda: torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="CUDA"):
        GleefulChat(model_name=base, adapter=lora_config(peft), quantization="nf4")


@pytest.mark.parametrize("load_in_8bit", [False, True], ids=["disabled", "int8"])
def test_chat_rejects_non_four_bit_bitsandbytes(chat_adapter_data, load_in_8bit):
    peft = pytest.importorskip("peft")
    base, _ = chat_adapter_data
    quantization = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    with pytest.raises(ValueError, match="(?i)(4.bit|four.bit)"):
        GleefulChat(model_name=base, adapter=lora_config(peft), quantization=quantization)


@pytest.mark.parametrize(
    "precision, compute_dtype", [("bf16", torch.bfloat16), ("fp16", torch.float16)]
)
def test_chat_forwards_preset_native_and_saved_nf4_settings(
    chat_adapter_data, tmp_path, monkeypatch, precision, compute_dtype,
):
    peft = pytest.importorskip("peft")
    base, _ = chat_adapter_data
    device = torch.device("cuda:0")
    monkeypatch.setattr(
        "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_device",
        lambda: device,
    )
    monkeypatch.setattr(
        "gleefultransformer._core.gleeful_tasks.gleeful_chat.get_precision",
        lambda device: precision,
    )
    monkeypatch.setattr(chat_adapters, "is_bitsandbytes_available", lambda: True)

    load_model = chat_adapters.get_model
    forwarded = []

    def load_on_cpu(model_name, trust_remote_code, precision, *, quantization_config, device_map, **kwargs):
        forwarded.append((quantization_config, device_map))
        return load_model(model_name, trust_remote_code, "fp32", **kwargs)

    monkeypatch.setattr(chat_adapters, "get_model", load_on_cpu)
    preset_config = nf4_config(compute_dtype)
    native_config = nf4_config(torch.float32)
    GleefulChat(model_name=base, adapter=lora_config(peft), quantization=native_config)
    chat = GleefulChat(model_name=base, adapter=lora_config(peft), quantization="nf4")
    export = tmp_path / "adapter"
    chat.save(export)
    (export / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": preset_config.to_dict()["bnb_4bit_compute_dtype"],
    }}))

    reloaded = GleefulChat(model_name=export)

    assert len(forwarded) == 3
    for (options, device_map), expected in zip(forwarded, (native_config, preset_config, preset_config)):
        assert options is not expected
        assert options.to_dict() == expected.to_dict()
        assert device_map == {"": device}

    reloaded.model.is_loaded_in_4bit = True
    reloaded.model.hf_device_map = {"": device}
    reloaded._get_model_ready_inference()
    assert reloaded.model.device.type == "cpu"
