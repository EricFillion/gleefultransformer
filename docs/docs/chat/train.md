# Train a chat model

## JSONL format

Each line contains one conversation in a `messages` list. Each message has a `role` and `content`. For supervised fine-tuning, include an assistant reply in each conversation.

```json
{"messages":[{"role":"user","content":"What is an eclipse?"},{"role":"assistant","content":"An eclipse occurs when one object in space blocks the light reaching another."}]}
```

## Create the quickstart datasets

From the checkout root, create three JSONL files with one conversation each. The training file supplies the example to learn from, the validation file measures loss during training, and the test file is used for the final comparison. Use larger, representative splits when evaluating your own model.

```sh
mkdir -p outputs/fine-tune-data
cat > outputs/fine-tune-data/train.jsonl <<'JSONL'
{"messages":[{"role":"user","content":"What is an eclipse?"},{"role":"assistant","content":"An eclipse occurs when one object in space blocks the light reaching another."}]}
JSONL
cat > outputs/fine-tune-data/validation.jsonl <<'JSONL'
{"messages":[{"role":"user","content":"What causes a solar eclipse?"},{"role":"assistant","content":"The Moon passes between Earth and the Sun."}]}
JSONL
cat > outputs/fine-tune-data/test.jsonl <<'JSONL'
{"messages":[{"role":"user","content":"What causes a lunar eclipse?"},{"role":"assistant","content":"Earth passes between the Sun and the Moon, casting its shadow on the Moon."}]}
JSONL
```

## Train

Pass a JSONL file or a [tokenized directory](../tokenize.md) as `train_path`. Pass a separate validation file or tokenized directory as `eval_path` to measure loss during training.

```python
from gleefultransformer import CHATTokArgs, GleefulChat, GleefulTrainArgs

chat = GleefulChat(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
args = GleefulTrainArgs(out_dir="runs", eval_steps=1, log_steps=1, save_best=True)
result = chat.train(
    "outputs/fine-tune-data/train.jsonl",
    args=args,
    eval_path="outputs/fine-tune-data/validation.jsonl",
    tok_args=CHATTokArgs(max_len=128),
)
print(result.final_train_loss)
print(result.best_eval_loss)
```

### Full fine-tuning with FP32 weights and BF16 compute

On a CUDA GPU with native BF16 support, load the model in FP32 and pass it to `GleefulChat` with its tokenizer. This keeps the trainable weights in FP32 while using BF16 mixed precision for training. FP32 weights require more GPU memory. The run's `precision.json` should show `"training_precision": "bf16"` and `"weight_dtypes": ["float32"]`.

```python
import torch
from transformers import AutoModelForCausalLM

from gleefultransformer import GleefulChat, GleefulTrainArgs

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
chat = GleefulChat(model_name=model, tokenizer=model_name)
args = GleefulTrainArgs(precision="bf16", eval_steps=25, checkpoint_steps=25)
chat.train(
    "outputs/fine-tune-data/train.jsonl",
    args=args,
    eval_path="outputs/fine-tune-data/validation.jsonl",
)
```

## LoRA and QLoRA

LoRA and QLoRA train an adapter on a frozen base. Start with an unquantized chat checkpoint such as SmolLM2. From the checkout, run `python -m pip install '.[qlora]'` for either mode. The extra installs PEFT and includes bitsandbytes on Linux and Windows. Pass a `LoraConfig` as `adapter`.

```python
from peft import LoraConfig, TaskType

from gleefultransformer import GleefulChat, GleefulTrainArgs

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
adapter = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
)
chat = GleefulChat(model_name=model_name, adapter=adapter)
```

For CUDA QLoRA, pass `quantization="nf4"` to use 4-bit NF4 with double quantization. The preset uses native BF16 compute when available and FP16 otherwise. For LoRA, omit `quantization` or leave it set to `None`. [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora) · [Transformers bitsandbytes configuration](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

```python
chat = GleefulChat(model_name=model_name, adapter=adapter, quantization="nf4")
```

For custom settings, pass a native Transformers `BitsAndBytesConfig` as `quantization`. This example uses FP16 compute and disables double quantization.

```python
import torch
from transformers import BitsAndBytesConfig

chat = GleefulChat(
    model_name=model_name,
    adapter=adapter,
    quantization=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,
    ),
)
```

Train and save either adapter with the same methods.

```python
chat.train(
    "outputs/fine-tune-data/train.jsonl",
    args=GleefulTrainArgs(out_dir="runs", run_name="my-run", checkpoint_steps=25),
)
chat.save("outputs/my-adapter")
chat = GleefulChat(model_name="outputs/my-adapter")
```

Saved adapter directories contain the adapter weights, tokenizer, and model configuration. The original base checkpoint must remain available. To resume an interrupted run without loading the base twice, initialize with just the checkpoint tokenizer:

```python
checkpoint = "runs/my-run/checkpoint"
chat = GleefulChat(model_name=None, tokenizer=checkpoint)
chat.train(
    "outputs/fine-tune-data/train.jsonl",
    args=GleefulTrainArgs(out_dir="runs", run_name="my-resume"),
    resume_path=checkpoint,
)
```

## Evaluate

```python
from gleefultransformer import CHATTokArgs, GleefulEvalArgs

result = chat.eval(
    "outputs/fine-tune-data/test.jsonl",
    args=GleefulEvalArgs(bs=2),
    tok_args=CHATTokArgs(max_len=128),
)
print(result.loss)
```

See [Training Arguments](../train-args.md) for batch size, gradient accumulation, and checkpointing.
