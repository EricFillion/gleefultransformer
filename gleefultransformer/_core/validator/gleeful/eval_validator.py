import os

from gleefultransformer._core.args import GleefulEvalArgs
from gleefultransformer._core.exceptions import GleefulInputError
from gleefultransformer._core.validator import GleefulValidator


class EvalValidator(GleefulValidator):
    def __init__(
        self,
        eval_path: str = "",
        args: GleefulEvalArgs = GleefulEvalArgs(),
        out_dir: str = "",
    ):
        self.eval_path = eval_path
        self.args = args
        self.out_dir = out_dir

        super().__init__()

        self.eval_source = self.resolve_eval_source()

    def validate_init(self):
        self._validate_file_paths_exist()
        self._validate_args_type()
        self._validate_eval_args_fields()
        self._validate_out_dir()

    def _validate_file_paths_exist(self):
        if not os.path.isfile(self.eval_path) and not os.path.isdir(
            self.eval_path
        ):
            raise FileNotFoundError(
                f"Input file or tokenized directory  not found: {self.eval_path}"
            )

    def _validate_args_type(self):
        if not isinstance(self.args, GleefulEvalArgs):
            raise TypeError(
                f"Expected args to be of type EvalArgs but got {type(self.args).__name__}"
            )

    def _validate_eval_args_fields(self):
        args = self.args

        if not isinstance(args.bs, int) or args.bs <= 0:
            raise GleefulInputError("`bs` must be a positive integer.")

        if not isinstance(args.seed, int) or args.seed < 0:
            raise GleefulInputError("`seed` must be a non-negative integer.")

    def resolve_eval_source(self) -> str:
        if os.path.isfile(self.eval_path):
            return "file"
        elif os.path.isdir(self.eval_path):
            return "folder"

    def _validate_out_dir(self):
        if self.eval_path:
            if self.out_dir == "":
                raise GleefulInputError(
                    "`out_dir` cannot be empty when using `eval_path`."
                )
