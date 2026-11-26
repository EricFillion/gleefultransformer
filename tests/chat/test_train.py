import copy

from tests.chat import (
    CHAT_CALL_ARGS,
    CHAT_CALL_TEXT,
    CHAT_DATA_JSONL,
)
from tests.functions.train import (
    t_train_basic,
    t_train_bs,
    t_train_eff,
    t_train_formats,
    t_train_long,
    t_train_sgd,
    t_train_checkpoint
)


def test_train_basic(gleeful_chat, chat_train_args):
    t_train_basic(
        gleeful_model=gleeful_chat,
        train_args=chat_train_args,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_eff(gleeful_chat_train, chat_train_args, chat_eval_args):
    t_train_eff(
        gleeful_model=gleeful_chat_train,
        train_args=chat_train_args,
        eval_args=chat_eval_args,
        train_eval_data=CHAT_DATA_JSONL,
        call_text=CHAT_CALL_TEXT,
        call_args=CHAT_CALL_ARGS,
    )


def test_train_formats(gleeful_chat, chat_train_args):
    # trains with all file formats
    train_data_list = [CHAT_DATA_JSONL]
    t_train_formats(
        gleeful_model=gleeful_chat, train_args=chat_train_args, train_data=train_data_list
    )


def test_train_bs(gleeful_chat, chat_train_args):
    # increase batch size
    t_train_bs(
        gleeful_model=gleeful_chat,
        train_args=chat_train_args,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_long(gleeful_chat_train, chat_train_args, chat_eval_args):
    # long training run with multiple save/eval/log steps
    t_train_long(
        gleeful_model=gleeful_chat_train,
        train_args=chat_train_args,
        eval_args=chat_eval_args,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_sgd(gleeful_chat, chat_train_args):
    t_train_sgd(
        gleeful_model=gleeful_chat,
        train_args=chat_train_args,
        train_eval_data=CHAT_DATA_JSONL,
    )


def test_train_constant_lr(gleeful_chat, chat_train_args):
    args = copy.deepcopy(chat_train_args)
    args.lr_sched = "constant"
    t_train_basic(
        gleeful_model=gleeful_chat, train_args=args, train_eval_data=CHAT_DATA_JSONL
    )



def test_train_checkpoint(gleeful_chat, chat_train_args):
    # just runs train
    t_train_checkpoint(
        gleeful_model=gleeful_chat, train_args=chat_train_args, train_eval_data=CHAT_DATA_JSONL
    )
