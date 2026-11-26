# Gleeful Transformer

- Run training and evaluation jobs from your AI assistant via MCP.
- Fine-tune models with LoRA, QLoRA, or full-parameter training.
- Pretrain models across multiple GPUs with FSDP2.
- Build retrieval-augmented generation (RAG) workflows with Gleeful Search, a custom vector search engine.

## Install

Use Python 3.11 or later. Install Gleeful Transformer from PyPI. 

```sh
pip install gleefultransformer
```

## MCP

Use the built-in Model Context Protocol (MCP) server to chat with a local model, evaluate it on a dataset, or fine-tune it.

Paste this prompt into your assistant for setup help.

> Help me set up Gleeful Transformer MCP from <https://github.com/EricFillion/gleefultransformer>. Ask which MCP client I use and whether I want to chat with a local model, evaluate it on a dataset, or fine-tune it. Then guide me through setup and verify the connection by listing my registered models.

[Manual setup](https://ericfillion.github.io/gleefultransformer/mcp/) · [Connect a client](https://ericfillion.github.io/gleefultransformer/mcp/#connect)

## Chat

```python
from gleefultransformer import CHATCallArgs, GleefulChat

chat = GleefulChat(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
messages = [{"role": "user", "content": "Hi"}]
args = CHATCallArgs(max_len=96, temp=0)

print(chat(messages, args=args).text)
```

Pass the same messages and settings to `stream()` to display the reply as it arrives.

```python
for event in chat.stream(messages, args=args):
    if event.marker == "text":
        print(event.text, end="", flush=True)
print()
```

[Chat guide](https://ericfillion.github.io/gleefultransformer/chat/) · [Inference arguments](https://ericfillion.github.io/gleefultransformer/call-args/)

## Fine-tune

In your working directory, create separate JSONL files for training, validation, and the final comparison. [Create the quickstart datasets](https://ericfillion.github.io/gleefultransformer/chat/train/#create-the-quickstart-datasets)

```python
from gleefultransformer import CHATTokArgs, GleefulChat, GleefulTrainArgs

chat = GleefulChat(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
tok_args = CHATTokArgs(max_len=128)
test_path = "outputs/fine-tune-data/test.jsonl"

baseline = chat.eval(test_path, tok_args=tok_args).loss
chat.train(
    "outputs/fine-tune-data/train.jsonl",
    args=GleefulTrainArgs(eval_steps=1),
    eval_path="outputs/fine-tune-data/validation.jsonl",
    tok_args=tok_args,
)
chat.save("outputs/fine-tuned-chat")

tuned = GleefulChat(model_name="outputs/fine-tuned-chat")
after_reload = tuned.eval(test_path, tok_args=tok_args).loss
print(f"Before training {baseline:.3f}, after reload {after_reload:.3f}")
```

The training guide covers evaluation, LoRA, QLoRA, and checkpoint resumption. [Training guide](https://ericfillion.github.io/gleefultransformer/chat/train/) · [Training arguments](https://ericfillion.github.io/gleefultransformer/train-args/) · [Databricks Dolly example](https://github.com/EricFillion/gleefultransformer/tree/main/examples/fine_tuning/chat/databricks)

## Pretrain with FSDP2

After installing, clone the repository to get the example scripts. Run these commands on a Linux machine with two NVIDIA GPUs that support BF16.

```sh
git clone https://github.com/EricFillion/gleefultransformer.git
cd gleefultransformer/examples/pretraining/chat/common_corpus
python data.py
python tok.py
torchrun --standalone --nproc-per-node=2 train.py --fsdp_config fsdp2.json --precision bf16 --bs 2 --gas 4 \
  --token_budget 261120 --eval_steps 16 --checkpoint_steps 16 --save_model_path outputs/fsdp2
```

The consolidated model is saved to `outputs/fsdp2` in the example directory. If NCCL stalls at startup on Runpod, retry with `NCCL_P2P_DISABLE=1` before `torchrun`. [Full example and troubleshooting](https://github.com/EricFillion/gleefultransformer/tree/main/examples/pretraining/chat/common_corpus)

## RAG

Install Gleeful Transformer with the `search` extra to enable RAG.

```sh
python -m pip install 'gleefultransformer[search]'
```

Create a sample passage in your working directory.

```sh
mkdir -p outputs
cat > outputs/passages.jsonl <<'JSONL'
{"text":"Titan is the largest moon orbiting Saturn.","metadata":{"source":"moon-guide"}}
JSONL
```

```python
from gleefultransformer import (
    CHATCallArgs,
    GleefulChat,
    GleefulSearch,
    SearchCallArgs,
    SearchTrainArgs,
)

search = GleefulSearch()
search.train("outputs/passages.jsonl", args=SearchTrainArgs(out_dir="outputs/my-index"))

chat = GleefulChat(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    gleeful_search=GleefulSearch(data_name="outputs/my-index"),
)
args = CHATCallArgs(max_len=96, search_args=SearchCallArgs(limit=1))
print(chat("What is Saturn's largest moon?", args=args).text)
```

`search_result` stream events include the retrieved text and its metadata. [RAG guide](https://ericfillion.github.io/gleefultransformer/search/) · [RAG example](https://github.com/EricFillion/gleefultransformer/tree/main/examples/search/hello_world)
