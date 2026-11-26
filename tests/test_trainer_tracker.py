import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer, TrainerControl, TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import EvalLoopOutput

from gleefultransformer import GleefulTrainArgs
from gleefultransformer._core.gleeful_tracker.trainer_callback import GleefulTrackerCallback
from gleefultransformer._core.trainer import GleefulTrainer
from gleefultransformer._core.utils.train.get_tok_data import TokenizedDataset


def test_final_best_checkpoint_has_matching_gleeful_tracker_report(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=8, n_positions=8, n_layer=1, n_head=1, n_embd=8,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))
    rows = [
        {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0], "labels": [1, 2, -100]},
        {"input_ids": [1, 3, 0], "attention_mask": [1, 1, 0], "labels": [1, 3, -100]},
    ]
    gleeful_args = GleefulTrainArgs(log_steps=3, eval_steps=3, save_best=True)
    resume_metadata = {"world_size": 1, "train_path": "/train.jsonl"}
    callback = GleefulTrackerCallback(
        gleeful_args, tmp_path, tracker_state={"metrics": {"legacy_metric": 3.0}},
        resume_metadata=resume_metadata,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, disable_tqdm=True,
            max_steps=2, per_device_train_batch_size=1,
            logging_steps=3, eval_strategy="steps", eval_steps=3,
            save_strategy="best", metric_for_best_model="loss", greater_is_better=False,
            include_num_input_tokens_seen="non_padding", report_to="none",
        ),
        train_dataset=rows, eval_dataset=rows,
        callbacks=[callback],
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()

    reported, = map(json.loads, (tmp_path / "logs.jsonl").read_text().splitlines())
    assert reported["current_step"] == reported["last_eval_step"] == 2
    assert reported["num_tokens"] == 4
    assert math.isfinite(reported["train_loss"])
    assert math.isfinite(reported["eval_loss"])
    assert "metrics" not in reported
    checkpoint = tmp_path / "checkpoint-2"
    assert reported["checkpoint_path"] == reported["best_checkpoint_path"] == checkpoint.name
    assert (tmp_path / "checkpoint").resolve() == (tmp_path / "best_model").resolve() == checkpoint
    assert (checkpoint / "optim.pt").resolve() == checkpoint / "optimizer.pt"
    assert json.loads((checkpoint / "gleeful_resume.json").read_text()) == resume_metadata
    checkpoint_state = json.loads((checkpoint / "gleeful_tracker_state.json").read_text())
    assert checkpoint_state["eval_loss"] == reported["eval_loss"]
    assert "metrics" not in checkpoint_state
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "loss_vs_tokens.png").is_file()


def test_replica_keeps_tracker_state_without_writing_shared_artifacts(tmp_path):
    barrier = Mock()
    args = SimpleNamespace(
        world_size=2, distributed_state=SimpleNamespace(wait_for_everyone=barrier),
        eval_strategy="steps",
    )
    state = TrainerState(max_steps=2, is_world_process_zero=False)
    control = TrainerControl()
    callback = GleefulTrackerCallback(GleefulTrainArgs(checkpoint_steps=1), tmp_path)
    callback.on_train_begin(args, state, control)

    state.global_step = 1
    state.num_input_tokens_seen = 12
    callback.on_step_end(args, state, control)
    callback.on_save(args, state, control)
    callback.close()

    barrier.assert_called_once_with()
    assert callback.state.last_checkpoint_step == 1
    assert callback.state.num_tokens == 12
    assert not list(tmp_path.iterdir())


def test_fsdp_evaluation_excludes_repeated_rows_and_averages_target_tokens(tmp_path, monkeypatch):
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=8, n_positions=8, n_layer=1, n_head=1, n_embd=8,
    )).eval()
    trainer = GleefulTrainer(
        model=model, causal=True,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
    )
    trainer.is_fsdp_enabled = True
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 4, 4, 4, 4], [3, 1, 1, 4, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5], [2, 4, -100, -100, -100], [3, -100, 1, 4, -100]]),
    }
    shard = tmp_path / "data.jsonl"
    shard.write_text("".join(
        json.dumps({key: value[row].tolist() for key, value in inputs.items()}) + "\n"
        for row in range(3)
    ))
    (tmp_path / "gleeful_details.json").write_text(json.dumps({
        "num_cases": 3, "paths": [str(shard)],
    }))
    dataloader = SimpleNamespace(dataset=TokenizedDataset(str(tmp_path), seed=42))

    def evaluate_padded_batches(self, loader, description, loss_only, ignore_keys, prefix):
        for rows in ([0, 1], [2, 0]):
            batch = {key: value[rows] for key, value in inputs.items()}
            self.prediction_step(model, batch, loss_only)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics={}, num_samples=4)

    monkeypatch.setattr(Trainer, "evaluation_loop", evaluate_padded_batches)
    with torch.no_grad():
        expected = model(**inputs).loss.item()
    result = trainer.evaluation_loop(dataloader, "Evaluation", prediction_loss_only=True)

    assert result.metrics["eval_loss"] == pytest.approx(expected, rel=1e-6)
    assert result.num_samples == 3
