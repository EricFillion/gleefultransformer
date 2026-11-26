import copy
import math
import os
from typing import Union

from transformers import PreTrainedModel

from gleefultransformer._core.args import GleefulTrainArgs
from gleefultransformer._core.exceptions import GleefulInputError
from gleefultransformer._core.validator import GleefulValidator


class TrainValidator(GleefulValidator):
    def __init__(
        self,
        train_path: str = "",
        args: GleefulTrainArgs = GleefulTrainArgs(),
        eval_path: str = "",
        resume_path: str = "",
        model: Union[PreTrainedModel, None] = None,
    ):
        self.train_path = train_path
        self.eval_path = eval_path
        self.resume_path = resume_path
        self.args = copy.deepcopy(args)
        self.model = model

        super().__init__()

    def validate_init(self):
        self._validate_required_inputs()
        self._validate_model_or_resume_path()
        self._validate_file_paths_exist()
        self._validate_args_type()
        self._validate_train_args()

    def _validate_required_inputs(self):
        if not self.train_path and not self.resume_path:
            raise GleefulInputError(
                "You must provide at least one of: `train_path` or `resume_path`."
            )

    def _validate_model_or_resume_path(self):
        if self.model is None and not self.resume_path:
            raise ValueError(
                "Either load a model when calling init or provide resume_path"
            )

    def _validate_file_paths_exist(self):
        if self.train_path and not (
            os.path.isfile(self.train_path) or os.path.isdir(self.train_path)
        ):
            raise GleefulInputError(
                f"Input file or tok dir not found: {self.train_path}"
            )
        if self.eval_path and not (
            os.path.isfile(self.eval_path) or os.path.isdir(self.eval_path)
        ):
            raise GleefulInputError(f"Eval file not found: {self.eval_path}")
        if self.resume_path and not os.path.isdir(self.resume_path):
            raise GleefulInputError(
                f"Resume path directory not found: {self.resume_path}"
            )

    def _validate_args_type(self):
        if not isinstance(self.args, GleefulTrainArgs):
            raise TypeError(
                f"Expected args to be of type TrainArgs but got {type(self.args).__name__}"
            )


    def _validate_train_args(self):
        args = self.args

        if not math.isfinite(args.lr) or args.lr <= 0:
            raise GleefulInputError("`lr` must be finite and positive.")
        if not math.isfinite(args.epochs) or args.epochs <= 0:
            raise GleefulInputError("`epochs` must be finite and positive.")

        for field, minimum in (
            ("gas", 1), ("bs", 1), ("eval_bs", 0), ("eval_steps", 0),
            ("log_steps", 0), ("checkpoint_steps", -1),
        ):
            value = getattr(args, field)
            if type(value) is not int or value < minimum:
                raise GleefulInputError(f"`{field}` must be an integer >= {minimum}.")

        if args.lr_sched not in ["constant", "warmup_then_decay"]:
            raise GleefulInputError(
                "`lr_sched` must be 'constant' or 'warmup_then_decay'."
            )

        if not isinstance(args.seed, int) or args.seed < 0:
            raise GleefulInputError("`seed` must be a non-negative int.")
        if not isinstance(args.run_name, str):
            raise GleefulInputError("`run_name` must be a string.")

        if not isinstance(args.save_best, bool):
            raise GleefulInputError("`save_best` must be a boolean.")

        if args.max_steps is not None and (type(args.max_steps) is not int or args.max_steps < 1):
            raise GleefulInputError("`max_steps` must be a positive integer or None.")
        if args.precision not in ("auto", "fp32", "bf16", "fp16"):
            raise GleefulInputError("`precision` must be 'auto', 'fp32', 'bf16', or 'fp16'.")
        if args.fsdp_config is not None and not isinstance(args.fsdp_config, (dict, str, os.PathLike)):
            raise GleefulInputError("`fsdp_config` must be a dictionary or a JSON file path.")
        if args.optim not in ("adamw", "sgd"):
            raise GleefulInputError("`optim` must be 'adamw' or 'sgd'.")
