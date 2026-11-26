# Chat

## Initialization

`GleefulChat` loads a causal language model from a Hugging Face model ID or a local directory. You can also pass a loaded model with its tokenizer. The tokenizer needs a chat template, which determines the supported roles and how messages are formatted.

## Call

Pass a prompt string or a list of messages with `role` and `content`. The call returns a `CHATResult` with the reply in `text`.

```python
from gleefultransformer import CHATCallArgs, GleefulChat

chat = GleefulChat(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
args = CHATCallArgs(max_len=128)
messages = [{"role": "user", "content": "Explain why the sky is blue."}]

result = chat(messages, args=args)
print(result.text)
```

## Stream

`stream()` takes the same arguments and yields `CHATStreamResult` events. Gleeful uses the tokenizer's response template when available. Print events with the `text` marker to display the reply as it arrives.

```python
for event in chat.stream(messages, args=args):
    if event.marker == "text":
        print(event.text, end="", flush=True)
print()
```

Pass a `GleefulSearch` instance as `gleeful_search` to add retrieved passages to the prompt. See [Gleeful Search](../search/index.md#chat-with-search) for an example with source metadata and [Inference Arguments](../call-args.md) for sampling options.

## Cohere North Mini Code

For [North Mini Code 1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0), use the checkpoint ID `CohereLabs/North-Mini-Code-1.0`. North streams reasoning as `thinking` events, and a regular call returns the answer in `.text`.
