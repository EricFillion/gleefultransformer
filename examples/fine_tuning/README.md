# Fine-tuning with Gleeful Transformer

The [Databricks Dolly example](chat/databricks/) fine-tunes a chat model on instruction and response pairs.

From `chat/databricks/`, run the following commands.

```sh
python train.py --train_cases 128 --eval_cases 32 --checkpoint_steps 64
python sample.py
```

`train.py` loads the dataset and writes prepared conversations to `data/train.jsonl` and `data/eval.jsonl`. Training starts from a Hugging Face model ID or directory passed as `--model_name` and saves the final model to `model/` by default.

Use `--bs`, `--eval_bs`, `--gas`, `--epochs`, and `--lr` to adjust training. Set `--eval_steps` or `--log_steps` to zero to disable that action. Set `--checkpoint_steps` to save checkpoints and use `--save_best` to keep the model with the lowest validation loss.

Pass `--checkpoint` to `sample.py` to load the latest checkpoint. If you changed the training output directory with `--out_dir`, use the same value in both commands. Use `--text` to supply your own input.

To upload the saved model to a private Hugging Face repository, sign in and run `python push.py --repo_id your-account/your-model`.
