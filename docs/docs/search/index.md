# Gleeful Search

From the Gleeful Transformer checkout, install the search extra with `python -m pip install '.[search]'`.

## Index documents

`GleefulSearch` retrieves passages from a vector index. To build an index, use a JSONL file or a directory of JSONL files. Each record has a `text` field and can include a `metadata` object.

```json
{"text":"The planet Saturn has a ring system made mostly of ice.","metadata":{"source":"planet-guide"}}
{"text":"Titan is Saturn's largest moon.","metadata":{"source":"moon-guide"}}
```

```python
from gleefultransformer import GleefulSearch, SearchTrainArgs

index = GleefulSearch()
index.train("passages.jsonl", args=SearchTrainArgs(out_dir="my-index"))
```

## Search

Load a local index directory or a Hugging Face dataset ID as `data_name`. Each result includes `text`, `score`, `best_sentence`, and `metadata`.

```python
from gleefultransformer import GleefulSearch, SearchCallArgs

search = GleefulSearch(data_name="my-index")
args = SearchCallArgs(limit=2)
for result in search("What are Saturn's rings made of?", args=args):
    print(result.text, result.metadata)
```

`limit` sets the maximum number of results, `ranker_count` sets the maximum number of candidates sent to the ranker, and `bs` controls the ranker batch size.

Gleeful Search loads small indexes into memory when you open them or finish training. Each query scores every passage in memory before sending the top candidates to the ranker. The default cutoff is 16 MiB of saved embeddings and JSONL records combined. The loaded index can use more memory because embeddings are converted to float32 and records are decoded into Python objects.

Larger indexes load selected leaves from disk per query. `leaf_count` controls how many leaves this search visits. Set `in_memory_max_bytes` on `GleefulSearch` to change the cutoff, or set it to `0` to always search from disk.

## Chat with search

Pass the loaded index to `GleefulChat`. The retrieved passages are added to the prompt. During streaming, `search_result` events expose the passages and their metadata.

```python
from gleefultransformer import CHATCallArgs, GleefulChat, GleefulSearch, SearchCallArgs

search = GleefulSearch(data_name="my-index")
chat = GleefulChat(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    gleeful_search=search,
)
args = CHATCallArgs(max_len=128, search_args=SearchCallArgs(limit=2))
prompt = "What are Saturn's rings made of?"

sources = []
for event in chat.stream(prompt, args=args):
    if event.marker == "text":
        print(event.text, end="", flush=True)
    elif event.marker == "search_result":
        sources.append(event.payload["metadata"])
print()
print(sources)
```
