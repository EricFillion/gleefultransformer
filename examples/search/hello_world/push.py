import argparse
import json
from pathlib import Path

from gleefultransformer import GleefulSearch, SearchTrainArgs


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Gleeful Search index from sample documents."
    )
    parser.add_argument("--repo_id", help="Hugging Face dataset ID for an optional upload")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", default="data/output")
    parser.add_argument("--cases", type=int, default=100, help="Number of sample documents")
    parser.add_argument("--query", help="Run a query against the local index")
    parser.add_argument("--bs", type=int, default=4, help="Upload batch size")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    if args.cases < 1 or args.bs < 1:
        parser.error("--cases and --bs must be positive")
    return args


def main(args: argparse.Namespace):
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.raw_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for number in range(args.cases):
            document = {
                "text": f"Hello world! This is case {number} out of {args.cases - 1}.",
                "metadata": {"number": number},
            }
            handle.write(json.dumps(document) + "\n")

    search = GleefulSearch()
    search.train(
        train_path=str(train_path),
        args=SearchTrainArgs(leaf_size=4, out_dir=args.out_dir),
    )
    print(f"Saved index to {args.out_dir}")

    if args.query:
        for result in search(args.query):
            print(result.text, result.metadata)
    if args.repo_id:
        search.push(repo_id=args.repo_id, bs=args.bs, branch=args.branch)


if __name__ == "__main__":
    main(get_args())
