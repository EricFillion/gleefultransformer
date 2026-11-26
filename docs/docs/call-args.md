# Inference arguments

Pass `CHATCallArgs` as `args` when calling a `GleefulChat` instance or its `stream()` method.

| Argument | Default | Description |
| --- | --- | --- |
| `min_len` | `1` | Minimum number of new tokens to generate locally. |
| `max_len` | `4096` | Maximum number of new tokens to generate. |
| `temp` | `0.8` | Sampling temperature. Use `0` for greedy decoding. |
| `top_k` | `32` | Limits local sampling to the highest-scoring tokens. |
| `top_p` | `0.6` | Limits sampling by cumulative probability. |
| `search_args` | `SearchCallArgs()` | Retrieval options for chat with [Gleeful Search](search/index.md). |
