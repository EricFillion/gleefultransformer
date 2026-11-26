# Training arguments

Pass `GleefulTrainArgs` to `train()` on a `GleefulChat` instance.

| Argument | Default | Description |
| --- | --- | --- |
| `lr` | `2e-5` | Optimizer learning rate. |
| `bs` | `1` | Training examples per device per batch. |
| `eval_bs` | `0` | Validation examples per device per batch. Zero uses `bs`. |
| `epochs` | `1` | Number of passes through the training data. |
| `max_steps` | `None` | A positive optimizer-step limit overrides `epochs`. |
| `gas` | `1` | Batches per optimizer step. |
| `optim` | `"adamw"` | Optimizer. The other option is `"sgd"`. |
| `lr_sched` | `"constant"` | Learning-rate schedule. The other option is `"warmup_then_decay"`. |
| `eval_steps` | `256` | Optimizer steps between validation runs. Use `0` to disable. |
| `log_steps` | `256` | Optimizer steps between log entries. Use `0` to disable. |
| `checkpoint_steps` | `-1` | Optimizer steps between checkpoints. Use `-1` or `0` to disable. |
| `save_best` | `False` | Save the model with the lowest validation loss when validation is enabled. |
| `out_dir` | `"runs/"` | Parent directory for run artifacts. |
| `run_name` | `""` | Name of the run directory. An empty value uses a generated name. |
| `seed` | `42` | Seed used when shuffling the tokenized dataset. |
| `precision` | `"auto"` | Training precision. Also accepts `"fp32"`, `"bf16"`, or `"fp16"`. |
| `fsdp_config` | `None` | Provide a dictionary or JSON file path to enable FSDP2. Use `{}` for defaults. |

Launch distributed training with `torchrun` and set `fsdp_config` to enable FSDP2. Call `save()` on all ranks to export a distributed model. [FSDP2 configuration](https://huggingface.co/docs/transformers/en/fsdp)

`"auto"` selects native BF16 on CUDA when available, FP16 on other CUDA GPUs, and FP32 on CPU and Apple Silicon. When training resolves to FP32, it casts ordinary model weights to FP32, including weights loaded from a lower-precision checkpoint. A cast preserves the values already loaded. Quantized models keep their quantization settings. `"bf16"` and `"fp16"` enable mixed precision on supported devices.

For full fine-tuning on a CUDA GPU with native BF16 support, see [how to load FP32 weights and train with BF16 compute](chat/train.md#full-fine-tuning-with-fp32-weights-and-bf16-compute).

## Run artifacts

Every training run records the requested precision, the resolved training mode, and the model's parameter dtypes in `precision.json`. For bitsandbytes 4-bit models, the report also includes the quantized layers' compute dtype as `quantization_compute_dtype` when available.

Positive logging, validation, and checkpoint intervals also trigger at the final optimizer step. Validation requires a validation file or tokenized directory. Run directories can contain `logs.jsonl`, plots, and Trainer's `checkpoint-N` directories. Trainer keeps two checkpoints, including the best when tracked. The `checkpoint` link identifies the latest checkpoint. With `save_best=True`, the `best_model` link identifies the checkpoint with the lowest validation loss. An FSDP2 sharded checkpoint is for resume. Use `save()` for a consolidated model export.

`train()` returns a `TrainResult` with `final_train_loss`, `final_eval_loss`, and `best_eval_loss`. The evaluation fields report losses when validation ran.

To continue an interrupted run, pass its `checkpoint` link or a `checkpoint-N` directory as `resume_path`. Use the original data and distributed settings.
