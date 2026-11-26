import copy
import math
from typing import List

from gleefultransformer import CHATTokArgs, EvalResult, GleefulChat, GleefulEvalArgs


CHAT_TOK_ARGS = CHATTokArgs(max_len=64)

def t_eval_formats(
    gleeful_model: GleefulChat, eval_args: GleefulEvalArgs, eval_data: List[str]
):
    for data in eval_data:
        result = gleeful_model.eval(data, args=eval_args, tok_args=CHAT_TOK_ARGS)
        assert type(result) == EvalResult
        assert type(result.loss) == float
        assert math.isfinite(result.loss)
        assert {name for name in dir(result) if not name.startswith("_")} == {"loss"}


def t_eval_bs(gleeful_model: GleefulChat, eval_args: GleefulEvalArgs, eval_data: str):
    eval_args_copy = copy.deepcopy(eval_args)
    eval_args_copy.bs = 2
    result = gleeful_model.eval(eval_data, args=eval_args_copy, tok_args=CHAT_TOK_ARGS)
    assert type(result) == EvalResult
    assert type(result.loss) == float
