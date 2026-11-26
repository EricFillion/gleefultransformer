import argparse
from pathlib import Path

from gleefultransformer import CHATCallArgs, GleefulChat


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="model/")

    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--max_len", type=int, default=128, help="Maximum new tokens")

    p.add_argument("--text", type=str, default="Artificial intelligence")
    return p.parse_args()


def get_checkpoint_path(out_dir):
    root = Path(out_dir)
    weight_names = {
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    }
    checkpoints = [
        path for pattern in ("*/checkpoint", "*/checkpoint-*")
        for path in root.glob(pattern)
        if (path / "config.json").is_file()
        and any((path / name).is_file() for name in weight_names)
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No loadable model checkpoints under {out_dir}. "
            "Pass the consolidated export to --model_name for sharded runs."
        )
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def main(args):
    if args.checkpoint:
        model_name = get_checkpoint_path(args.out_dir)
    elif args.model_name:
        model_name = args.model_name
    else:
        raise ValueError(
            "Provide a path to the --model_name parameter or --checkpoint after running train.py with checkpointing enabled."
        )

    print(f"Loading model from {model_name}")
    chat = GleefulChat(model_name=model_name)
    result = chat(args.text, args=CHATCallArgs(max_len=args.max_len))

    print("INPUT: ", args.text)
    print("OUTPUT: ", result.text)


if __name__ == "__main__":
    main(get_args())
