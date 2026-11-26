# AGENTS.md

## Repository layout

- `gleefultransformer/` contains the Python package. Keep the public imports in `gleefultransformer/__init__.py` working when changing internal modules.
- `gleefultransformer/_core/` contains chat inference and training. `gleefultransformer/gleefulsearch/` contains retrieval and ranking.
- `gleefultransformer/mcp/` contains the stdio server, workspace registry, run store, and worker. The worker handles model execution.
- `docs/docs/` contains the MkDocs guides. Run example scripts from their own directories so their relative paths resolve.

## Development and checks

Use Python 3.11 or later. From the checkout root, run the test suite.

```sh
python -m pip install -e '.[test]'
python -m pytest
```

For MCP changes, run the focused tests after installing the test extra. The stdio integration tests create their own tiny local model.

```sh
python -m pytest tests/mcp
```

For tests that use downloaded chat and search checkpoints, install `.[search,test]` and run `python -m tests.get_test_models`. Set `GLEEFULTRANSFORMER_TEST_MODELS` to use a fixture directory other than `models/`. For guide changes, run `mkdocs build --strict` from `docs/` after installing `docs/requirements.txt`.

## Using the MCP tools

- Use the same absolute workspace path for the server and worker. The workspace registry maps model and dataset IDs to local files. [MCP setup](docs/docs/mcp.md)
- Discover registered inputs with `gleeful_find_model_profiles` and `gleeful_find_dataset_splits`. Check training data with `gleeful_check_dataset_contract`.
- Submit the recipe with `gleeful_submit_finetune`. Reuse the same request ID when retrying the same submission. Use `gleeful_read_run_snapshot` and follow `poll_after_ms` while a run is active.
- Compare base and exported models on the same held-out split with the same sequence limit. Use `artifacts.final_model_id` from a successful fine-tune for evaluation or replies.
