# Tokenize data

Use `tok()` to tokenize data once and reuse the resulting directory in later `train()` or `eval()` calls. Use separate directories for training and validation data. Training labels include the rendered conversation and mask padding tokens.

Use `CHATTokArgs` to control chat tokenization. Its options include `max_len` for sequence length, `bs` for tokenization batch size, `max_cases` for the maximum examples, `shards` for the number of outputs, and `procs` for worker processes.

```python
from gleefultransformer import CHATTokArgs, GleefulChat

chat = GleefulChat(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
chat.tok("train.jsonl", "chat-tokens/train", args=CHATTokArgs(max_len=512))
chat.train("chat-tokens/train")
```

You can also pass `tok_args=CHATTokArgs(...)` to `train()` or `eval()` when using raw JSONL files.
