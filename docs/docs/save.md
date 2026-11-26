# Save and load a model

`save(path)` writes model weights, tokenizer files, and configuration to a local directory in Hugging Face format. An adapter export stores the adapter weights and a reference to the base model. To reload the model, pass the directory as `model_name` to `GleefulChat`.

```python
from gleefultransformer import GleefulChat

model = GleefulChat(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
model.save("saved-model")

restored = GleefulChat(model_name="saved-model")
```

Use `checkpoint_steps` to save resumable training checkpoints and `save_best=True` to keep the checkpoint with the lowest validation loss in the run directory. For distributed training, call `save()` on every rank to gather and export weights. Load the exported directory for inference or as the starting point for another training run. See [Training Arguments](train-args.md) for checkpoint settings.
