import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from transformers import AutoConfig, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

from gleefultransformer._core.utils.init.get_components import get_model, get_tokenizer

if TYPE_CHECKING:
    from peft import LoraConfig


def is_adapter_directory(model_name):
    return (
        isinstance(model_name, str)
        and (Path(model_name) / "adapter_config.json").is_file()
    )


def _saved_quantization(adapter_path):
    config_path = Path(adapter_path) / "config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as config_file:
            return json.load(config_file).get("quantization_config")
    return None


def _resolve_qlora_config(quantization, saved_quantization, device, precision):
    if saved_quantization is not None:
        if not isinstance(saved_quantization, dict) or saved_quantization.get("quant_method") != "bitsandbytes":
            raise ValueError("Saved quantized adapters must use the bitsandbytes backend.")
        config = BitsAndBytesConfig.from_dict(saved_quantization)
    elif isinstance(quantization, BitsAndBytesConfig):
        config = deepcopy(quantization)
    elif isinstance(quantization, str):
        if quantization != "nf4":
            raise ValueError("The supported quantization preset is 'nf4'.")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
        )
    else:
        raise TypeError("quantization must be 'nf4' or a Transformers BitsAndBytesConfig.")
    if not config.load_in_4bit or config.load_in_8bit:
        raise ValueError("QLoRA requires a 4-bit BitsAndBytesConfig.")
    if device.type != "cuda":
        raise ValueError("GleefulChat QLoRA currently requires a CUDA GPU.")
    if not is_bitsandbytes_available():
        raise ImportError("QLoRA requires bitsandbytes. Install gleefultransformer[qlora].")
    if config.bnb_4bit_compute_dtype == torch.bfloat16 and precision != "bf16":
        raise ValueError("The QLoRA config uses BF16 compute, which this GPU does not support.")
    return config


def load_chat_adapter(
    model_name,
    adapter: "LoraConfig | None",
    *,
    quantization: Literal["nf4"] | BitsAndBytesConfig | None,
    tokenizer_path,
    trust_remote_code,
    device,
    precision,
):
    adapter_path = model_name if is_adapter_directory(model_name) else None
    if adapter_path and (adapter is not None or quantization is not None):
        raise ValueError("A saved adapter directory supplies its own adapter and quantization settings.")
    if quantization is not None and adapter is None:
        raise ValueError("quantization requires a PEFT LoraConfig adapter.")
    if not isinstance(model_name, str):
        raise TypeError("Adapter training requires a base model ID or local checkpoint path.")

    try:
        from peft import (
            LoraConfig,
            PeftConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise ImportError(
            "Chat adapters require PEFT. Install gleefultransformer[qlora]."
        ) from exc

    revision = None
    saved_quantization = None
    if adapter_path:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        if not isinstance(peft_config, LoraConfig) or not peft_config.base_model_name_or_path:
            raise ValueError("The saved adapter must contain a LoRA config and a base model ID or path.")
        model_name = peft_config.base_model_name_or_path
        revision = peft_config.revision
        saved_quantization = _saved_quantization(adapter_path)
        if tokenizer_path is None and (Path(adapter_path) / "tokenizer_config.json").is_file():
            tokenizer_path = adapter_path
    else:
        if not isinstance(adapter, LoraConfig):
            raise TypeError("adapter must be a PEFT LoraConfig.")
        peft_config = deepcopy(adapter)
    if peft_config.task_type not in (None, TaskType.CAUSAL_LM):
        raise ValueError("GleefulChat adapters require the CAUSAL_LM task type.")
    peft_config.task_type = TaskType.CAUSAL_LM

    use_qlora = quantization is not None or saved_quantization is not None
    load_options = {}
    if use_qlora:
        load_options = {
            "quantization_config": _resolve_qlora_config(
                quantization, saved_quantization, device, precision
            ),
            "device_map": {"": device},
        }

    if Path(model_name).is_dir():
        model_name = str(Path(model_name).resolve())
    base_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, revision=revision
    )
    revision = revision or getattr(base_config, "_commit_hash", None)
    if getattr(base_config, "quantization_config", None):
        raise ValueError("GleefulChat adapters require an unquantized base checkpoint.")

    tokenizer = get_tokenizer(model_name, tokenizer_path, trust_remote_code, revision=revision)
    model = get_model(
        model_name,
        trust_remote_code,
        precision,
        config=base_config,
        revision=revision,
        **load_options,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        raise ValueError("The tokenizer has tokens outside the frozen base model's vocabulary.")
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, activation_checkpointing_kwargs={"use_reentrant": False}
        )
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=True, config=peft_config
        )
    else:
        model = get_peft_model(model, peft_config, revision=revision)
    return model.config, tokenizer, model
