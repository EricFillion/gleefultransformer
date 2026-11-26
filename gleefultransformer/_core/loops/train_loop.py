from dataclasses import dataclass

from gleefultransformer._core.utils.test import DebugHook


@dataclass
class TrainResult:
    final_train_loss: float
    final_eval_loss: float | None = None
    best_eval_loss: float | None = None


debug_hook_post_checkpoint = DebugHook()
debug_hook_steps = DebugHook()
