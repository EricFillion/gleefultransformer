# Gleeful Transformer

Gleeful Transformer is a Python package for running, streaming, and fine-tuning open-weight chat models. It includes Gleeful Search for retrieving passages and adding them to chat prompts.

## Install

Use Python 3.11 or later. On Linux or macOS, clone the repository and install it in a virtual environment.

```sh
git clone https://github.com/EricFillion/gleefultransformer.git
cd gleefultransformer
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

- [Run a chat model](chat/index.md) with a prompt or a conversation and stream the reply.
- [Fine-tune and evaluate a model](chat/train.md) on your own data.
- [Index and retrieve passages](search/index.md) with Gleeful Search.
- [Evaluate and fine-tune chat models from an MCP client](mcp.md).
