import argparse
import hashlib
import json
import random
from contextlib import ExitStack
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm


DATASET = "PleIAs/common_corpus"
SUBSETS = {
    "OpenScience": "Open Science",
    "OpenWeb": "Open Web",
    "OpenCulture": "Open Culture",
    "OpenSource": "Open Source",
    "OpenGovernment": "Open Government",
    "OpenSemantic": "Semantic data",
}


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Common Corpus document splits.")
    parser.add_argument("--train_cases", type=int, default=1024)
    parser.add_argument("--eval_cases", type=int, default=64)
    parser.add_argument("--untok_train", type=Path, default=Path("data/untok/train"))
    parser.add_argument("--untok_eval", type=Path, default=Path("data/untok/eval"))
    parser.add_argument("--dataset_revision", default="main")
    parser.add_argument(
        "--datasets", nargs="+", choices=SUBSETS,
        default=[name for name in SUBSETS if name != "OpenSource"],
    )
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()
    if min(args.train_cases, args.eval_cases, args.shards) < 1:
        parser.error("train_cases, eval_cases, and shards must be positive")
    return args


def write_splits(dataset, args, revision):
    counts = {"train": args.train_cases, "eval": args.eval_cases}
    directories = {"train": args.untok_train, "eval": args.untok_eval}
    if directories["train"].resolve() == directories["eval"].resolve():
        raise ValueError("Use different output directories for train and eval.")
    total = sum(counts.values())
    eval_positions = set(random.Random(args.seed).sample(range(total), args.eval_cases))
    selected_subsets = {SUBSETS[name] for name in args.datasets}
    seen = set()
    written = dict.fromkeys(counts, 0)
    paths = {}

    with ExitStack() as stack, tqdm(total=total, desc="Documents selected") as progress:
        writers = {}
        for split, directory in directories.items():
            directory.mkdir(parents=True, exist_ok=True)
            paths[split] = [
                (directory / f"{split}_shard_{index:04d}.jsonl").resolve()
                for index in range(min(args.shards, counts[split]))
            ]
            writers[split] = [
                stack.enter_context(path.open("w", encoding="utf-8"))
                for path in paths[split]
            ]

        for row in dataset:
            text = row.get("text")
            subset = row.get("open_type")
            if not text or subset not in selected_subsets:
                continue
            if subset != SUBSETS["OpenSource"] and row.get("language") != "English":
                continue
            digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
            if digest in seen:
                continue

            split = "eval" if len(seen) in eval_positions else "train"
            seen.add(digest)
            files = writers[split]
            files[written[split] % len(files)].write(
                json.dumps({"text": text}, ensure_ascii=False) + "\n"
            )
            written[split] += 1
            progress.update()
            if len(seen) == total:
                break

    if written != counts:
        raise ValueError(f"Only found {written} matching documents for requested {counts}.")

    for split, directory in directories.items():
        manifest = {
            "dataset": DATASET,
            "revision": revision,
            "subsets": args.datasets,
            "seed": args.seed,
            "num_documents": written[split],
            "paths": [path.name for path in paths[split]],
        }
        (directory / "source.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {written[split]} {split} documents to {directory}")


def main(args):
    revision = HfApi().dataset_info(DATASET, revision=args.dataset_revision).sha
    dataset = load_dataset(DATASET, revision=revision, split="train", streaming=True)
    write_splits(dataset, args, revision)


if __name__ == "__main__":
    main(get_args())
