from pathlib import Path

from gleefultransformer import CHATCallArgs


CHAT_DATA_JSONL = str(Path(__file__).resolve().parent / "data/data.jsonl")
CHAT_CALL_TEXT = "Tell me a joke"
CHAT_CALL_ARGS = CHATCallArgs(min_len=5, max_len=5, top_k=0, top_p=0.0)
