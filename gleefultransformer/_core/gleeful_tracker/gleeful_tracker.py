import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from gleefultransformer._core.exceptions import GleefulIOError
from gleefultransformer._core.gleeful_tracker.save_plot import (
    save_lr_plot,
    save_train_eval_loss_plot,
)


@dataclass
class TrackerState:
    current_step: int = 0
    last_eval_step: int = 0
    original_eval_loss: float = 0.0
    eval_loss: float = 0.0
    best_eval_loss: Optional[float] = None
    eval_loss_improvement: float = 0.0
    train_loss: float = 0.0
    last_checkpoint_step: Optional[int] = None
    epoch: int = 1
    lr: float = 0
    num_tokens: int = 0
    tokens_per_second: Optional[float] = None
    checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        # Older checkpoints may contain retired metrics.
        data.pop("metrics", None)
        data.pop("target_tokens", None)
        data.pop("target_tokens_per_second", None)
        return cls(**data)

    def set_eval_loss(self, eval_loss):
        if self.last_eval_step == 0:
            self.original_eval_loss = eval_loss
        self.eval_loss = eval_loss
        is_best_model = math.isfinite(eval_loss) and (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        if is_best_model:
            self.best_eval_loss = eval_loss
        self.eval_loss_improvement = self.original_eval_loss - eval_loss
        self.last_eval_step = self.current_step
        return is_best_model



class GleefulTracker:
    def __init__(self, train_steps: int, out_dir: Path, state: TrackerState):
        self.state = state
        self.out_dir = out_dir
        self.log_path = out_dir / "logs.jsonl"
        try:
            self.log_path.touch(exist_ok=True)
        except OSError as exc:
            raise GleefulIOError(f"Could not create training log {self.log_path}: {exc}") from exc
        self.history = []
        self.pbar = tqdm(total=train_steps, initial=state.current_step, desc="Model Training")

    def record(self):
        entry = self.state.to_dict()
        del entry["original_eval_loss"]
        try:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as exc:
            raise GleefulIOError(f"Could not write training log {self.log_path}: {exc}") from exc

        self.history.append(copy.copy(self.state))
        evaluation_history = [
            point for point in self.history if point.last_eval_step == point.current_step
        ]
        eval_steps = [point.current_step for point in evaluation_history]
        save_train_eval_loss_plot(self.history, eval_steps, self.out_dir)
        save_lr_plot(self.history, self.out_dir)
        if evaluation_history:
            save_train_eval_loss_plot(self.history, eval_steps, self.out_dir, by_tokens=True)

        self.pbar.set_postfix(
            last_eval_step=self.state.last_eval_step,
            eval_loss_improvement=f"{self.state.eval_loss_improvement:.4f}",
            eval_loss=self.state.eval_loss,
            loss=f"{self.state.train_loss:.4f}",
            num_tokens=self.state.num_tokens,
        )

    def move_to_step(self, step):
        self.pbar.update(step - self.state.current_step)
        self.state.current_step = step

    def close(self):
        self.pbar.close()
