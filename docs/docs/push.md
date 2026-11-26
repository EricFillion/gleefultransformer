# Push to the Hugging Face Hub

Sign in with an account that has write access to the destination repository.

```sh
hf auth login
```

Call `push()` with the model repository ID to upload the model, tokenizer, and configuration. New repositories are private by default.

```python
from gleefultransformer import GleefulChat

model = GleefulChat(model_name="saved-model")
model.push("your-username/your-model", private=True)
```

To load the uploaded model, pass the repository ID as `model_name`. Accessing a private repository requires a signed-in account with permission to read it.
