import argparse
import json
import os
from pathlib import Path

from torch import distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from gleefultransformer import GleefulChat, GleefulTrainArgs
from tok import COMPLETION_TEMPLATE, DEFAULT_MODEL


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", help="Model ID or path used by tok.py")
    p.add_argument("--fsdp_config", help="Path to an FSDP2 JSON config")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")

    # Learning parameters
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--eval_bs", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--token_budget", type=int, help="Maximum causal target tokens")
    p.add_argument("--gas", type=int, default=1)
    p.add_argument("--optim", choices=["adamw", "sgd"], type=str, default="adamw")
    p.add_argument(
        "--lr_sched",
        choices=["warmup_then_decay", "constant"],
        type=str,
        default="warmup_then_decay",
    )

    # Action Steps
    p.add_argument("--eval_steps", type=int, default=256)
    p.add_argument("--log_steps", type=int, default=256)
    p.add_argument("--checkpoint_steps", type=int, default=-1)
    p.add_argument("--save_best", action="store_true")

    # Paths
    p.add_argument("--tok_train", type=str, default="data/tok/train")
    p.add_argument("--tok_eval", type=str, default="data/tok/eval")
    p.add_argument("--save_model_path", type=str, default="model/")
    p.add_argument("--out_dir", type=str, default="runs/")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--resume_path", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if min(args.bs, args.gas) < 1:
        p.error("bs and gas must be positive")
    return args


def read_details(path):
    path = Path(path) / "gleeful_details.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def budget_steps(args, details, world_size):
    if args.token_budget is None:
        return None
    block_size = details.get("block_size")
    if not block_size:
        raise ValueError("--token_budget requires the packed data produced by tok.py.")
    blocks_per_step = args.bs * args.gas * world_size
    tokens_per_step = (block_size - 1) * blocks_per_step
    steps = args.token_budget // tokens_per_step
    if steps < 1:
        raise ValueError(f"Budget must allow at least one step ({tokens_per_step:,} target tokens).")
    if steps * blocks_per_step > details["num_cases"]:
        raise ValueError("Budget exceeds the prepared training data. Lower it or pack more data.")
    return steps


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = os.environ.get("RANK", "0") == "0"
    details = read_details(args.tok_train)
    model_name = details.get("model_name", args.model_name or DEFAULT_MODEL)
    if args.model_name and args.model_name != model_name:
        raise ValueError(f"Packed data uses {model_name!r}. Rerun tok.py to change models.")

    snapshot = Path(args.tok_train) / "model_source"
    model_source = snapshot if snapshot.is_dir() else model_name
    revision = None if snapshot.is_dir() else details.get("model_revision")
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=revision)

    if not tokenizer.chat_template:
        tokenizer.chat_template = COMPLETION_TEMPLATE
    model = None
    if not args.resume_path or args.fsdp_config:
        set_seed(args.seed)
        config = AutoConfig.from_pretrained(model_source, revision=revision)
        model = AutoModelForCausalLM.from_config(config)
    chat = GleefulChat(model_name=model, tokenizer=tokenizer)

    steps = budget_steps(args, details, world_size)
    if is_main and steps is not None:
        planned_tokens = steps * args.bs * args.gas * world_size * (details["block_size"] - 1)
        print(f"Planned {steps:,} optimizer steps and {planned_tokens:,} causal target tokens.")

    train_args = GleefulTrainArgs(
        out_dir=args.out_dir,
        epochs=args.epochs,
        max_steps=steps,
        bs=args.bs,
        gas=args.gas,
        eval_bs=args.eval_bs,
        lr=args.lr,
        optim=args.optim,
        lr_sched=args.lr_sched,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        checkpoint_steps=args.checkpoint_steps,
        save_best=args.save_best,
        run_name=args.run_name,
        seed=args.seed,
        fsdp_config=args.fsdp_config,
        precision=args.precision,
    )

    result = chat.train(
        args.tok_train,
        resume_path=args.resume_path,
        eval_path=args.tok_eval,
        args=train_args,
    )

    chat.save(args.save_model_path)

    if is_main:
        print(f"Model saved to {args.save_model_path}")
        if result.final_eval_loss is not None:
            print(f"Held-out loss {result.final_eval_loss:.4f}")


if __name__ == "__main__":
    main(get_args())
    if dist.is_initialized():
        dist.destroy_process_group()
