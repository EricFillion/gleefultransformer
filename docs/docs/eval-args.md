# Evaluation arguments

`GleefulChat.eval()` accepts a JSONL file of conversations or a [tokenized directory](tokenize.md). It takes `GleefulEvalArgs` and returns the evaluation loss in `EvalResult.loss`.

| Argument | Default | Description |
| --- | --- | --- |
| `bs` | `1` | Evaluation examples per device per batch. |
| `out_dir` | `"runs/"` | Parent directory for evaluation artifacts. |
| `run_name` | `""` | Name of the run directory. An empty value uses a generated name. |
| `seed` | `42` | Seed for evaluation. |

During training, the tracker records evaluation loss in `logs.jsonl`.
