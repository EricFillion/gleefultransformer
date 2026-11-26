import argparse
import json
from pathlib import Path

from datasets import load_dataset

from gleefultransformer import CHATTokArgs, GleefulChat, GleefulTrainArgs


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Generic
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")

    adapters = p.add_argument_group("LoRA and QLoRA")
    adapters.add_argument(
        "--adapter", choices=["lora", "qlora"], help="Select an adapter training method"
    )
    adapters.add_argument("--lora_rank", type=int, help="Adapter rank (default: 16)")
    adapters.add_argument("--lora_alpha", type=int, help="Adapter scaling (default: 32)")
    adapters.add_argument("--lora_dropout", type=float, help="Adapter dropout (default: 0.05)")
    adapters.add_argument(
        "--lora_target_modules", nargs="+",
        help="Module names to adapt (default: all-linear)",
    )
    adapters.add_argument(
        "--qlora_quantization_config",
        type=Path,
        help="Path to a JSON object with Transformers BitsAndBytesConfig settings",
    )

    # Number of cases
    p.add_argument("--train_cases", type=int, default=2048)
    p.add_argument("--eval_cases", type=int, default=128)
    p.add_argument("--max_chars", type=int, default=10_000)
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=CHATTokArgs.max_len,
        help="Tokenized sequence length (default: inferred from the tokenizer)",
    )

    # Learning parameters
    p.add_argument("--bs", type=int, default=1)
    p.add_argument(
        "--eval_bs", type=int, default=0
    )  # when 0 uses the same value as --bs
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gas", type=int, default=1)
    p.add_argument("--optim", choices=["adamw", "sgd"], type=str, default="sgd")
    p.add_argument(
        "--lr_sched",
        choices=["warmup_then_decay", "constant"],
        type=str,
        default="constant",
    )

    # Action Steps
    p.add_argument("--eval_steps", type=int, default=256)
    p.add_argument("--log_steps", type=int, default=256)
    p.add_argument("--checkpoint_steps", type=int, default=-1)
    p.add_argument("--save_best", action="store_true")

    # Paths
    p.add_argument("--train_path", type=str, default="data/train.jsonl")
    p.add_argument("--eval_path", type=str, default="data/eval.jsonl")
    p.add_argument("--save_model_path", type=str, default="model/")
    p.add_argument("--out_dir", type=str, default="runs/")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--resume_path", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.adapter is None and any(
        value is not None
        for value in (args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_target_modules)
    ):
        p.error("LoRA settings require --adapter lora or --adapter qlora")
    if args.qlora_quantization_config is not None and args.adapter != "qlora":
        p.error("--qlora_quantization_config requires --adapter qlora")
    if "all-linear" in (args.lora_target_modules or []) and args.lora_target_modules != ["all-linear"]:
        p.error("Use all-linear by itself in --lora_target_modules")
    return args


def get_adapter_options(args: argparse.Namespace):
    if args.adapter is None:
        return {}

    from peft import LoraConfig, TaskType

    targets = args.lora_target_modules
    if targets is None or targets == ["all-linear"]:
        targets = "all-linear"
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16 if args.lora_rank is None else args.lora_rank,
        lora_alpha=32 if args.lora_alpha is None else args.lora_alpha,
        lora_dropout=0.05 if args.lora_dropout is None else args.lora_dropout,
        target_modules=targets,
    )
    if args.adapter == "lora":
        return {"adapter": adapter}
    if args.qlora_quantization_config is None:
        return {"adapter": adapter, "quantization": "nf4"}

    from transformers import BitsAndBytesConfig

    settings = json.loads(args.qlora_quantization_config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("The QLoRA quantization config must contain a JSON object.")
    quantization = BitsAndBytesConfig.from_dict({"load_in_4bit": True, **settings})
    return {"adapter": adapter, "quantization": quantization}


def main(args: argparse.Namespace):
    if args.train_cases + args.eval_cases > 15_011:
        raise ValueError("num_train and num_eval must be no more than 15_011")

    adapter_options = {} if args.resume_path else get_adapter_options(args)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    all_data = ds.train_test_split(test_size=args.eval_cases, seed=args.seed)
    train_cases = all_data["train"].select(range(args.train_cases))

    eval_cases = all_data["test"]

    format_data(args.train_path, train_cases, args)
    format_data(args.eval_path, eval_cases, args)

    if args.resume_path:
        gleeful_chat = GleefulChat(model_name=None, tokenizer=args.resume_path)
    else:
        gleeful_chat = GleefulChat(model_name=args.model_name, **adapter_options)

    train_args = GleefulTrainArgs(
        out_dir=args.out_dir,
        bs=args.bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        epochs=args.epochs,
        gas=args.gas,
        optim=args.optim,
        lr_sched=args.lr_sched,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        checkpoint_steps=args.checkpoint_steps,
        save_best=args.save_best,
        seed=args.seed,
        run_name=args.run_name,
    )
    gleeful_chat.train(
        args.train_path,
        resume_path=args.resume_path,
        args=train_args,
        eval_path=args.eval_path,
        tok_args=CHATTokArgs(max_len=args.max_seq_len),
    )
    gleeful_chat.save(args.save_model_path)
    print("model saved to {}".format(args.save_model_path))


def format_data(jsonl_path, dataset, args):
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case in dataset:
            prompt = case["instruction"][:args.max_chars]
            if case["context"]:
                prompt = f'{case["context"][:args.max_chars]}\n\n{prompt}'
            row = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": case["response"][:args.max_chars]},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(get_args())
