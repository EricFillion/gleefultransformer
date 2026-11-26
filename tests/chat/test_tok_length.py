from gleefultransformer import CHATTokArgs
from gleefultransformer._core.gleeful_tasks.tok.tok_functions import get_max_in_len
from tests.chat import CHAT_DATA_JSONL
from tests.functions.tok_length import t_tok_length


def test_pad(gleeful_chat, tmp_path):
    target_length = 256

    args = CHATTokArgs(max_len=target_length)

    t_tok_length(
        gleeful_model=gleeful_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=str(tmp_path),
        args=args,
        target_length=target_length,
    )


def test_pad_max(gleeful_chat, tmp_path):
    target_length = get_max_in_len(
        max_len=-1, tokenizer=gleeful_chat.tokenizer
    )
    args = CHATTokArgs()
    t_tok_length(
        gleeful_model=gleeful_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=str(tmp_path),
        args=args,
        target_length=target_length,
    )


def test_trunc(gleeful_chat, tmp_path):
    target_length = 4
    args = CHATTokArgs(max_len=target_length)

    t_tok_length(
        gleeful_model=gleeful_chat,
        input_data_list=[CHAT_DATA_JSONL],
        out_dir=str(tmp_path),
        args=args,
        target_length=target_length,
    )
