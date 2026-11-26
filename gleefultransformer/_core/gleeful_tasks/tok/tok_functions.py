from typing import Union

from datasets import Dataset
from transformers import PreTrainedTokenizer

from gleefultransformer._core.exceptions import GleefulTokenizationError


def get_max_in_len(
    max_len: Union[int, None], tokenizer: PreTrainedTokenizer
) -> int:
    if max_len >=1:
        return max_len
    else:
        if tokenizer.model_max_length > 10_000_000:
            return 512
        return tokenizer.model_max_length


def tokenize_chat_template(
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_len: int,
    bs: int,
    procs: int = 1,
):

    def tokenize_fn(example):
        try:
            inputs = [msg for msg in example["messages"]]
            tokens = tokenizer.apply_chat_template(
                inputs,
                tokenize=True,
                add_generation_prompt=False,
                max_length=max_len,
                padding="max_length",
                return_dict=True,
                truncation=True,
            )

            tokens["labels"] = [
                [token if attended else -100 for token, attended in zip(ids, mask, strict=True)]
                for ids, mask in zip(tokens["input_ids"], tokens["attention_mask"], strict=True)
            ]

            return tokens
        except Exception as e:
            raise GleefulTokenizationError(
                f"Tokenization failed during chat template application. Error: {e}"
            )

    try:
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["messages"],
            batch_size=bs,
            desc="Tokenizing...",
            num_proc=procs,
        )
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized
    except Exception as e:
        raise GleefulTokenizationError(
            f"Failed during dataset mapping or formatting. Error: {e}"
        )
