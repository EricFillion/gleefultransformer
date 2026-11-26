from typing import Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from gleefultransformer._core.exceptions import GleefulLoadModelError, GleefulLoadTokenizerError


def _get_torch_dtype(torch_dtype: str) -> Union[torch.dtype, str]:
    if torch_dtype == "auto":
        return "auto"
    elif torch_dtype == "fp32":
        return torch.float32
    elif torch_dtype == "fp16":
        return torch.float16
    elif torch_dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(
            f"Invalid torch_dtype {torch_dtype}. Provide one of auto, fp32, fp16, or bf16."
        )


def get_model(
    model_name_path: Union[str, PreTrainedModel],
    trust_remote_code: bool,
    precision,
    *,
    config: Optional[PretrainedConfig] = None,
    **model_kwargs,
) -> PreTrainedModel:
    try:
        if isinstance(model_name_path, PreTrainedModel):
            return model_name_path

        if config is None:
            config = AutoConfig.from_pretrained(
                model_name_path, trust_remote_code=trust_remote_code
            )
        torch_dtype = _get_torch_dtype(precision)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            config=config,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            **model_kwargs,
        )

        return model

    except Exception as e:
        raise GleefulLoadModelError(f"Failed to load model from '{model_name_path}': {e}")


def _load_tokenizer(
    tokenizer_path: str, trust_remote_code: bool, *, revision: Optional[str] = None
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        return tokenizer
    except Exception as e:
        raise GleefulLoadTokenizerError(
            f"Failed to load tokenizer from '{tokenizer_path}': {e}"
        )


def get_tokenizer(
    model_name_path: str,
    tokenizer_path: Union[str, PreTrainedTokenizerBase],
    trust_remote_code: bool,
    *,
    revision: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    if (
        tokenizer_path is None
    ):  # by default we use the same value provided to model_name_path
        tokenizer = _load_tokenizer(
            tokenizer_path=model_name_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

    elif isinstance(tokenizer_path, str):
        tokenizer = _load_tokenizer(
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

    elif isinstance(tokenizer_path, PreTrainedTokenizerBase):
        # tokenizer provided directly
        tokenizer = tokenizer_path
        # maybe remove the following code since if an advanced user provides their own tokenizer they might want full control
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            "Invalid tokenizer_path. Provide a None, a string or AutoTokenizer"
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def get_model_components(
    model_name_path: Union[str, PreTrainedModel, None],
    trust_remote_code: bool,
    tokenizer_path: Union[str, PreTrainedTokenizerBase],
    precision: str,
):
    tokenizer = get_tokenizer(
        model_name_path=model_name_path,
        tokenizer_path=tokenizer_path,
        trust_remote_code=trust_remote_code,
    )

    if model_name_path is not None:
        model = get_model(
            model_name_path=model_name_path,
            trust_remote_code=trust_remote_code,
            precision=precision,
        )
        config = model.config
    else:
        model = None
        config = None

    return config, tokenizer, model
