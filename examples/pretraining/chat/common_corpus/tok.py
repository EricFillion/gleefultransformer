import argparse
import json
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL = "openai-community/gpt2"
COMPLETION_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}"
    "{% if not loop.last %}{{ eos_token }}{% endif %}{% endfor %}"
)


def get_args():
    parser = argparse.ArgumentParser(description="Pack Common Corpus into causal-LM blocks.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--untok_train", type=Path, default=Path("data/untok/train"))
    parser.add_argument("--untok_eval", type=Path, default=Path("data/untok/eval"))
    parser.add_argument("--tok_train", type=Path, default=Path("data/tok/train"))
    parser.add_argument("--tok_eval", type=Path, default=Path("data/tok/eval"))
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--bs", type=int, default=32, help="Documents per tokenization batch")
    args = parser.parse_args()
    if args.block_size < 2 or min(args.shards, args.bs) < 1:
        parser.error("block_size must be at least 2 and shards and bs must be positive")
    return args


def read_documents(paths):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)["text"]


def iter_blocks(tokenizer, documents, block_size, batch_size):
    documents = iter(documents)
    remaining = []
    while batch := list(islice(documents, batch_size)):
        tokenized = tokenizer(
            batch, add_special_tokens=False, return_attention_mask=False, verbose=False
        )
        for token_ids in tokenized["input_ids"]:
            remaining.extend(token_ids)
            remaining.append(tokenizer.eos_token_id)
            end = len(remaining) // block_size * block_size
            for start in range(0, end, block_size):
                yield remaining[start:start + block_size]
            remaining = remaining[end:]


def pack_split(source_dir, target_dir, tokenizer, args, revision):
    source_manifest = source_dir / "source.json"
    manifest = {}
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_paths = [source_dir / name for name in manifest["paths"]]
    else:
        source_paths = sorted(source_dir.glob("*.jsonl"))
    if not source_paths:
        raise ValueError(f"No JSONL documents found in {source_dir}.")

    target_dir.mkdir(parents=True, exist_ok=True)
    blocks = iter_blocks(
        tokenizer, read_documents(source_paths), args.block_size, args.bs
    )
    count = 0
    paths, handles = [], []
    attention_mask = [1] * args.block_size
    with ExitStack() as stack:
        for index, block in enumerate(blocks):
            shard = index % args.shards
            if shard == len(paths):
                path = (target_dir / f"packed_{shard:04d}.jsonl").resolve()
                paths.append(path)
                handles.append(stack.enter_context(path.open("w", encoding="utf-8")))
            record = {"input_ids": block, "attention_mask": attention_mask, "labels": block}
            handles[shard].write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    if not count:
        raise ValueError(f"{source_dir} did not contain enough tokens for one full block.")

    details = {
        "num_cases": count,
        "paths": [str(path) for path in paths],
        "block_size": args.block_size,
        "model_name": args.model_name,
        "model_revision": revision,
        "dataset": manifest,
        "target_tokens": count * (args.block_size - 1),
    }
    (target_dir / "gleeful_details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packed {count:,} blocks with {details['target_tokens']:,} target tokens in {target_dir}.")


def main(args):
    paths = {path.resolve() for path in (
        args.tok_train, args.tok_eval, args.untok_train, args.untok_eval
    )}
    if len(paths) != 4:
        raise ValueError("Use different input and output directories for each split.")

    revision = None
    if not Path(args.model_name).is_dir():
        revision = HfApi().model_info(args.model_name, revision=args.model_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=revision)
    config = AutoConfig.from_pretrained(args.model_name, revision=revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must provide an end-of-document token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_limit = getattr(config, "max_position_embeddings", None) or tokenizer.model_max_length
    if args.block_size > min(model_limit, tokenizer.model_max_length):
        raise ValueError("block_size exceeds the tokenizer or model context limit.")

    pack_split(args.untok_train, args.tok_train, tokenizer, args, revision)
    pack_split(args.untok_eval, args.tok_eval, tokenizer, args, revision)
    if not tokenizer.chat_template:
        # Plain causal models can use GleefulChat for text continuation.
        tokenizer.chat_template = COMPLETION_TEMPLATE
    snapshot = args.tok_train / "model_source"
    tokenizer.save_pretrained(snapshot)
    config.save_pretrained(snapshot)


if __name__ == "__main__":
    main(get_args())
