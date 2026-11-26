from tests.chat import CHAT_DATA_JSONL
from tests.functions.eval import t_eval_bs, t_eval_formats


def test_eval_formats(gleeful_chat, chat_eval_args):
    data = [CHAT_DATA_JSONL]
    t_eval_formats(gleeful_chat, eval_args=chat_eval_args, eval_data=data)


def test_eval_bs(gleeful_chat, chat_eval_args):
    t_eval_bs(gleeful_chat, eval_args=chat_eval_args, eval_data=CHAT_DATA_JSONL)
