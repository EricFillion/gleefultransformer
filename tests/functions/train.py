import copy
from typing import List, Union
import pathlib
import tempfile
import pytest

from gleefultransformer import CHATCallArgs, CHATTokArgs, GleefulChat, GleefulEvalArgs, GleefulTrainArgs

from gleefultransformer._core.loops.train_loop import debug_hook_post_checkpoint, debug_hook_steps

CHAT_TOK_ARGS = CHATTokArgs(max_len=64)

def t_train_basic(
    gleeful_model: GleefulChat, train_args: GleefulTrainArgs, train_eval_data: str
):
    # Smoke test.
    gleeful_model.train(train_eval_data, eval_path=train_eval_data, args=train_args, tok_args=CHAT_TOK_ARGS)


def t_train_eff(
    gleeful_model: GleefulChat,
    train_args: GleefulTrainArgs,
    eval_args: GleefulEvalArgs,
    train_eval_data: str,
    call_text: str,
    call_args: Union[CHATCallArgs, None] = None
):
    # Assert train decreases loss.

    before_result = gleeful_model.eval(train_eval_data, args=eval_args, tok_args=CHAT_TOK_ARGS).loss
    if call_args is not None:
        output_before = gleeful_model(call_text, args=call_args)
    else:
        output_before = gleeful_model(call_text)

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.epochs = 10
    train_args_copy.lr = 1e-4
    gleeful_model.train(train_eval_data, args=train_args_copy, tok_args=CHAT_TOK_ARGS)

    # we run __call__  before and after just to make sure nothing breaks from training
    if call_args is not None:
        output_after = gleeful_model(call_text, args=call_args)

    else:
        output_after = gleeful_model(call_text)

    after_result = gleeful_model.eval(train_eval_data, args=eval_args, tok_args=CHAT_TOK_ARGS).loss

    print("after_result", after_result)
    print("before_result", before_result)

    assert after_result < before_result


def t_train_long(
    gleeful_model: GleefulChat,
    train_args: GleefulTrainArgs,
    eval_args: GleefulEvalArgs,
    train_eval_data: str,
):
    # Smoke test for longer train run.

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.lr = 0.01
    train_args_copy.eval_steps = 2
    train_args_copy.log_steps = 2
    train_args_copy.checkpoint_steps = 2
    train_args_copy.epochs = 5
    gleeful_model.train(
        train_eval_data, eval_path=train_eval_data, args=train_args_copy, tok_args=CHAT_TOK_ARGS,
    )


def t_train_formats(
    gleeful_model: GleefulChat, train_args: GleefulTrainArgs, train_data: List[str]
):
    # Smoke test for multiple import formats.

    for data in train_data:
        gleeful_model.train(data, eval_path=data, args=train_args, tok_args=CHAT_TOK_ARGS)


def t_train_bs(
    gleeful_model: GleefulChat, train_args: GleefulTrainArgs, train_eval_data: str
):
    # Smoke test for non-default batch size.

    new_args = copy.deepcopy(train_args)
    new_args.bs = 2
    gleeful_model.train(train_eval_data, args=new_args, tok_args=CHAT_TOK_ARGS)


def t_train_sgd(
    gleeful_model: GleefulChat, train_args: GleefulTrainArgs, train_eval_data: str
):
    # Smoke test for SGD optim.

    train_args_copy = copy.deepcopy(train_args)
    train_args_copy.optim = "sgd"
    gleeful_model.train(
        train_eval_data, eval_path=train_eval_data, args=train_args_copy, tok_args=CHAT_TOK_ARGS,
    )


def raise_error():
    raise Exception('fake exception')


def t_train_checkpoint(
    gleeful_model: GleefulChat, train_args: GleefulTrainArgs, train_eval_data: str
):

    train_args = copy.deepcopy(train_args)

    default_checkpoint_steps = train_args.checkpoint_steps
    train_args.checkpoint_steps = 1
    checkpoint_test_dir = pathlib.Path(train_args.out_dir) / "t_train_checkpoint"
    checkpoint_test_dir.mkdir(parents=True, exist_ok=True)
    train_args.out_dir = tempfile.mkdtemp(prefix="checkpoint-", dir=checkpoint_test_dir)

    # Setup hook to track steps.
    steps = None
    def hook_steps(n):
        nonlocal steps
        steps = n

    # On first checkpoint, error after to interrupt.
    with debug_hook_post_checkpoint.set(raise_error):
        train_args.run_name='run'
        # Run loop, expecting it to save a checkpoint and then get interrupted.
        with pytest.raises(Exception):
            gleeful_model.train(
                train_eval_data,
                eval_path=train_eval_data,
                args=train_args, tok_args=CHAT_TOK_ARGS,
            )

    train_args.checkpoint_steps = default_checkpoint_steps

    # On resumed run, check steps at end to see what training actually did.
    with debug_hook_steps.set(hook_steps):
        gleeful_model.train(
            train_eval_data,
            eval_path=train_eval_data,
            args=train_args, tok_args=CHAT_TOK_ARGS,
            resume_path=str(
                pathlib.Path(train_args.out_dir)
                / 'run' / 'checkpoint'
            )
        )

    assert steps is not None
    assert 0 < steps["start_step"] < steps["total_steps"]

