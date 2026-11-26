# Gleeful Transformer MCP

Use a local MCP client to evaluate chat models, run supervised fine-tunes, and ask a registered or exported model to reply. The client starts `gleeful-mcp` and communicates with it over stdio, a standard MCP transport. The separate `gleeful-mcp-worker` runs model jobs and stores their progress in the workspace, so jobs can continue if the client disconnects.

## Install and create a workspace

Follow the [installation quickstart](index.md#install) and keep the Python environment active. The standard installation includes MCP. Set the workspace path to a new directory for this example.

```sh
GLEEFUL_WORKSPACE="$HOME/gleeful-mcp-workspace"
mkdir -p "$GLEEFUL_WORKSPACE"/models "$GLEEFUL_WORKSPACE"/data
```

Place a complete local causal model checkpoint with a chat template under the workspace. Use an unquantized checkpoint for supervised fine-tuning. The following command downloads one pinned snapshot and selects the files the Transformers loader needs. [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) · [SmolLM2-135M-Instruct snapshot](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac)

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  config.json generation_config.json model.safetensors tokenizer.json \
  tokenizer_config.json special_tokens_map.json merges.txt vocab.json \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --local-dir "$GLEEFUL_WORKSPACE/models/smollm2-v1"
```

Create the workspace registry and three one-row JSONL files for the smoke test. The registry paths are relative to the workspace. Each line contains a conversation with a `messages` list, and each message has a `role` and string `content`. Use separate examples for training, validation during training, and the final comparison.

```sh
(
set -eC
cat > "$GLEEFUL_WORKSPACE/registry.json" <<'JSON'
{
  "version": 1,
  "models": [
    {
      "id": "smollm2-v1",
      "path": "models/smollm2-v1",
      "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    }
  ],
  "datasets": [
    {"id": "train-v1", "path": "data/train.jsonl", "role": "train"},
    {"id": "valid-v1", "path": "data/validation.jsonl", "role": "validation"},
    {"id": "test-v1", "path": "data/test.jsonl", "role": "test"}
  ]
}
JSON

cat > "$GLEEFUL_WORKSPACE/data/train.jsonl" <<'JSONL'
{"messages":[{"role":"user","content":"What planet do we live on?"},{"role":"assistant","content":"We live on Earth."}]}
JSONL

cat > "$GLEEFUL_WORKSPACE/data/validation.jsonl" <<'JSONL'
{"messages":[{"role":"user","content":"Is the Sun a star?"},{"role":"assistant","content":"Yes. The Sun is a star."}]}
JSONL

cat > "$GLEEFUL_WORKSPACE/data/test.jsonl" <<'JSONL'
{"messages":[{"role":"user","content":"Which planet is famous for its rings?"},{"role":"assistant","content":"Saturn is famous for its rings."}]}
JSONL
)
```

Keep model and dataset contents fixed under their registered IDs. When you replace a snapshot or split, give it a new ID and restart the stdio server to reload the registry.

## Connect

Start one worker in a terminal and leave it running. It uses the local device and processes one job at a time. The workspace and device locks prevent another worker from taking the same job or device.

```sh
gleeful-mcp-worker --workspace "$GLEEFUL_WORKSPACE"
```

In your MCP client, add a local stdio server named `gleeful`. Set its command to the absolute path of `gleeful-mcp` in the installed Python environment. Pass `--workspace` followed by the same absolute workspace path you used for the worker. Run `command -v gleeful-mcp` in that environment to find the executable.

If your client uses `mcpServers` JSON, use this example and replace both paths.

```json
{
  "mcpServers": {
    "gleeful": {
      "type": "stdio",
      "command": "/absolute/path/to/.venv/bin/gleeful-mcp",
      "args": ["--workspace", "/absolute/path/to/mcp-workspace"]
    }
  }
}
```

Open a client session and call `gleeful_find_model_profiles` to check that the connection works and your models are listed.

## Run a fine-tuning experiment

The following JSON objects are tool arguments. Your MCP client handles the tool calls.

1. Call `gleeful_find_model_profiles` and `gleeful_find_dataset_splits` with no arguments to list the IDs. Call `gleeful_check_dataset_contract` with `{"dataset_id":"train-v1","model_id":"smollm2-v1","max_sequence_tokens":128}` to review template compatibility and token counts. The checker also works with validation and test splits.
2. Call `gleeful_submit_dataset_evaluation` to establish a baseline. Save the returned `run_id`, then call `gleeful_read_run_snapshot` with that ID until the evaluation finishes.

    ```json
    {
      "model_id": "smollm2-v1",
      "dataset_id": "test-v1",
      "request_id": "smollm2-baseline-v1",
      "max_sequence_tokens": 128
    }
    ```

3. Call `gleeful_submit_finetune` with a model ID, training dataset ID, step limit, and request ID. The validation dataset and sequence limit are optional. The server checks the inputs before queuing the job. Repeat the same request and request ID after a client timeout to recover the existing run. Use a new request ID if you change the recipe.

    ```json
    {
      "recipe": {
        "model_id": "smollm2-v1",
        "train_dataset_id": "train-v1",
        "validation_dataset_id": "valid-v1",
        "max_steps": 2,
        "max_sequence_tokens": 128
      },
      "request_id": "smollm2-tune-v1"
    }
    ```

4. Reconnect with the same workspace and call `gleeful_read_run_snapshot` with the fine-tune's `run_id`. Use `gleeful_find_experiment_runs` to recover a lost run ID. On success, copy the ID from `artifacts.final_model_id`.
5. Submit another dataset evaluation using the export's ID, `test-v1`, the same `max_sequence_tokens`, and a new request ID. Check that the dataset and tokenizer revisions match the baseline before comparing `result.loss`. Fine-tune validation loss measures the weights at the time of training evaluation, and the base model ID records the input checkpoint. The separate evaluation measures the final exported weights. Both evaluations measure next-token loss over the templated conversation with padding masked.
6. Call `gleeful_reply_to_messages` with the export ID. Supply all the conversation turns needed for the reply. The tool returns `text` and `model_revision`.

    ```json
    {
      "model_id": "REPLACE_WITH_FINAL_MODEL_ID",
      "messages": [{"role": "user", "content": "Which planet is famous for its rings?"}],
      "max_new_tokens": 32
    }
    ```

## Read and stop runs

Run snapshots include state, phase, timestamps, and progress or metrics as they become available. Use `poll_after_ms` for active runs. Terminal states are `succeeded`, `failed`, `cancelled`, and `interrupted`. Failed runs stay readable and return a structured `error` when available.

The workspace database at `.gleeful/runs.sqlite3` stores messages submitted for reply jobs and the replies they generate. The worker writes run files under `.gleeful/artifacts/`, including copies of the JSONL files used for training and evaluation.

Call `gleeful_request_run_stop` with a `run_id` to cancel queued work. For an active run, the call records `cancel_requested`. Supervised training checks for the request at optimizer-step boundaries. Evaluation and generation honor a stop after the current blocking operation returns. When a replacement worker starts, it marks any abandoned active run as `interrupted`. Submit a fresh run with a new request ID to run it again.

## Limits

The MCP server uses local stdio and local registered inputs. Job submission enforces workspace limits for dataset size, optimizer steps, and sequence length. Training and evaluation use one example per batch. A workspace accepts up to 32 queued runs. The configured export size limit controls model publication. Rows over the effective sequence limit are rejected. `gleeful_reply_to_messages` accepts up to 512 requested new tokens and only starts when the workspace worker has no queued or running jobs. Generation also respects the model's remaining context. The server waits up to 30 seconds for a reply.

Use the [Python training guide](chat/train.md) to customize training settings, use LoRA or QLoRA, or resume from a checkpoint. See [Gleeful Search](search/index.md) for retrieval through Python.
