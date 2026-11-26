import argparse
from pathlib import Path

from gleefultransformer import CHATCallArgs, GleefulChat


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--text", type=str, default="What could I use Gleeful Transformer for?")

    # Train Args Start
    p.add_argument("--model_name", type=str, default="model/")

    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--out_dir", default="runs/")
    p.add_argument("--max_len", type=int, default=128, help="Maximum new tokens")

    p.add_argument(
        "--context",
        type=str,
        default="Gleeful Transformer is a Python package for running, streaming, and fine-tuning chat models. It can add passages from a Gleeful Search index to a prompt.",
    )

    return p.parse_args()


def get_checkpoint_path(out_dir):
    checkpoints = [
        path for path in Path(out_dir).glob("*/checkpoint")
        if (path / "config.json").is_file() or (path / "adapter_config.json").is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found under {out_dir}")
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def main(args):
    if args.checkpoint:
        model_name = get_checkpoint_path(args.out_dir)
    elif args.model_name:
        model_name = args.model_name
    else:
        raise ValueError(
            "Provide a path to the --model_name parameter or enable --checkpoint after running train.py with checkpointing."
        )

    sample_input = f"{args.context}\n\n{args.text}" if args.context else args.text

    print(f"Loading model from {model_name}")
    gleeful_chat = GleefulChat(model_name=model_name)
    result = gleeful_chat(
        sample_input, args=CHATCallArgs(top_k=8, temp=1, max_len=args.max_len)
    )

    print("INPUT: ", sample_input)
    print("OUTPUT: ", result.text)


if __name__ == "__main__":
    main(get_args())
