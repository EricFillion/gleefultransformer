import os

from gleefultransformer import CHATTokArgs
from tests.chat import CHAT_DATA_JSONL


def test_detok_length(gleeful_chat, tmp_path):
    MAX_LEN = 2

    save_path = os.path.join(str(tmp_path), "text_len")

    gleeful_chat.tok(CHAT_DATA_JSONL, save_path, args=CHATTokArgs(max_len=MAX_LEN))

    tok_data = gleeful_chat.tok_dir_to_dataset(save_path)

    for case in tok_data["input_ids"]:
        r = gleeful_chat.tokenizer.decode(case)
        assert len(case) == MAX_LEN
