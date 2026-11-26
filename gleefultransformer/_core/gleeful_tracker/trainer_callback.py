import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from gleefultransformer._artifact_paths import find_artifact
from gleefultransformer._core.gleeful_tracker.gleeful_tracker import GleefulTracker, TrackerState


TrainingObserver = Callable[[str, dict], bool | None]


class GleefulTrackerCallback(TrainerCallback):
    """Report Trainer progress and publish completed checkpoints from rank zero."""

    def __init__(
        self, gleeful_args, out_dir, tracker_state=None, *,
        resume_path=None, resume_metadata=None, observer: TrainingObserver | None = None,
    ):
        self.gleeful_args = gleeful_args
        self.out_dir = Path(out_dir).absolute()
        self.state = TrackerState.from_dict(tracker_state or {})
        self.resume_path = Path(resume_path).resolve() if resume_path else None
        self.resume_metadata = resume_metadata
        self.observer = observer
        self.precision_report = None
        if self.resume_path is not None and self.resume_metadata is None:
            metadata_path = find_artifact(self.resume_path, "gleeful_resume.json")
            if metadata_path.is_file():
                self.resume_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.checkpoint = None
        self.best_checkpoint = None
        self._tracker = None
        self._best_step = None
        self._record_step = False
        self._start_step = self.state.current_step
        self._last_loss_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_step = state.global_step
        self._start_time = monotonic()
        self._start_tokens = state.num_input_tokens_seen
        self.state.current_step = state.global_step
        self.state.num_tokens = state.num_input_tokens_seen

        if self.resume_path is not None:
            self.state.checkpoint_path = os.path.relpath(self.resume_path, self.out_dir)
            if self.state.best_checkpoint_path:
                self.best_checkpoint = (self.resume_path.parent / self.state.best_checkpoint_path).resolve()
            elif state.best_model_checkpoint:
                self.best_checkpoint = Path(state.best_model_checkpoint).resolve()
            if self.best_checkpoint is not None:
                if self.best_checkpoint.is_dir():
                    self.state.best_checkpoint_path = os.path.relpath(self.best_checkpoint, self.out_dir)
                else:
                    self.best_checkpoint = None
                    self.state.best_checkpoint_path = None

        if state.is_world_process_zero:
            if self.precision_report is not None:
                report = self.precision_report()
                if self.resume_metadata is not None:
                    self.resume_metadata["precision"] = report
                _write_json(self.out_dir / "precision.json", report)
                logging.getLogger("gleefultransformer").info(
                    "Training precision %s (requested %s). Model weight dtypes: %s",
                    report["training_precision"], report["requested_precision"],
                    ", ".join(report["weight_dtypes"]),
                )
            self._tracker = GleefulTracker(state.max_steps, self.out_dir, self.state)
            if self.best_checkpoint is not None:
                _link(self.best_checkpoint, self.out_dir / "best_model")
        self._notify("train_begin", state, control)

    def on_step_end(self, args, state, control, **kwargs):
        if self._tracker is not None:
            self._tracker.move_to_step(state.global_step)
        else:
            self.state.current_step = state.global_step
        self.state.epoch = max(1, math.ceil(state.epoch or 0))
        self.state.num_tokens = state.num_input_tokens_seen

        if state.global_step >= state.max_steps:
            if self.gleeful_args.log_steps > 0:
                control.should_log = True
            if self.gleeful_args.eval_steps > 0 and args.eval_strategy == "steps":
                control.should_evaluate = True
            if self.gleeful_args.checkpoint_steps > 0:
                control.should_save = True
        self._schedule_record(control)
        self._notify("step", state, control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._schedule_record(control)

    def _schedule_record(self, control):
        # Snapshot the latest loss before an evaluation or checkpoint, even when
        # the user disables the JSONL log and plots.
        if control.should_evaluate or control.should_save:
            if self._last_loss_step != self.state.current_step:
                control.should_log = True
        self._record_step |= self.gleeful_args.log_steps > 0 and (
            control.should_log or control.should_evaluate or control.should_save
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self.state.train_loss = float(logs["loss"])
            self._last_loss_step = state.global_step
        elif "train_loss" in logs and self._last_loss_step != state.global_step:
            self.state.train_loss = float(logs["train_loss"])
        if "learning_rate" in logs:
            self.state.lr = float(logs["learning_rate"])
        self._finish_record(control)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            improved = self.state.set_eval_loss(float(eval_loss))
            if improved and self.gleeful_args.save_best:
                self._best_step = state.global_step
                control.should_save = True

        self._finish_record(control)
        self._notify("evaluation", state, control, loss=eval_loss)

    def on_save(self, args, state, control, **kwargs):
        # Other ranks may still be writing RNG, optimizer, or model shards.
        if args.world_size > 1:
            args.distributed_state.wait_for_everyone()

        self.checkpoint = self.out_dir / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        self.state.last_checkpoint_step = state.global_step
        self.state.checkpoint_path = self.checkpoint.name
        new_best = self._best_step == state.global_step
        if new_best:
            self.best_checkpoint = self.checkpoint
            self.state.best_checkpoint_path = self.checkpoint.name

        self._update_throughput()
        if state.is_world_process_zero:
            _write_json(self.checkpoint / "gleeful_tracker_state.json", self.state.to_dict())
            _write_json(self.checkpoint / "train_args.json", asdict(self.gleeful_args))
            if self.resume_metadata is not None:
                _write_json(self.checkpoint / "gleeful_resume.json", self.resume_metadata)
            for native_name, previous_name in (("optimizer.pt", "optim.pt"), ("scheduler.pt", "lr_sched.pt")):
                native_path = self.checkpoint / native_name
                if native_path.is_file():
                    _link(native_path, self.checkpoint / previous_name)
            _link(self.checkpoint, self.out_dir / "checkpoint")
            if new_best:
                _link(self.checkpoint, self.out_dir / "best_model")

        self._finish_record(control)
        self._notify("checkpoint", state, control)
        from gleefultransformer._core.loops.train_loop import debug_hook_post_checkpoint

        debug_hook_post_checkpoint()

    def on_train_end(self, args, state, control, **kwargs):
        self._finish_record(control)
        self.close()
        from gleefultransformer._core.loops.train_loop import debug_hook_steps

        debug_hook_steps({"start_step": self._start_step, "total_steps": state.global_step})

    def _notify(self, event, state, control, **fields):
        if self.observer is not None and self.observer(
            event, {"current_step": state.global_step, "total_steps": state.max_steps, **fields}
        ):
            control.should_training_stop = True
            control.should_evaluate = False
            control.should_save = False

    def _finish_record(self, control):
        if self._record_step and not (control.should_log or control.should_evaluate or control.should_save):
            self._record_step = False
            if self._tracker is not None:
                if self.state.last_checkpoint_step != self.state.current_step:
                    self._update_throughput()
                self._tracker.record()

    def _update_throughput(self):
        duration = monotonic() - self._start_time
        self.state.tokens_per_second = (self.state.num_tokens - self._start_tokens) / duration

    def close(self):
        self.precision_report = None
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=os.fspath)
    temporary.replace(path)


def _link(target, link):
    temporary = link.with_name(f".{link.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())
    temporary.replace(link)
