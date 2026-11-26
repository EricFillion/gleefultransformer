"""Download the local checkpoints used by the chat and search integration tests."""

import os
from pathlib import Path


MODEL_DIR = Path(os.environ.get("GLEEFULTRANSFORMER_TEST_MODELS", "models")).resolve()
CHAT_MODEL_PATH = str(MODEL_DIR / "chat")
EMBEDDING_MODEL_PATH = str(MODEL_DIR / "embedding")
CROSS_ENCODER_PATH = str(MODEL_DIR / "cross_encoder")


def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from gleefultransformer import GleefulChat

    GleefulChat(model_name="microsoft/DialoGPT-small").save(CHAT_MODEL_PATH)
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(EMBEDDING_MODEL_PATH)
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2").save(CROSS_ENCODER_PATH)


if __name__ == "__main__":
    main()
