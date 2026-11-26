import json
import random
from itertools import islice
from pathlib import Path

from torch.utils.data import IterableDataset

from gleefultransformer._artifact_paths import find_artifact
from gleefultransformer._core.exceptions import GleefulDatasetError, GleefulInputError


class TokenizedDataset(IterableDataset):
    """Read JSONL shards with bounded, repeatable shuffling for Trainer."""

    def __init__(self, tok_dir: str, seed: int):
        details_path = find_artifact(tok_dir, "gleeful_details.json")
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
            self.num_cases = details["num_cases"]
            self.paths = [Path(path) for path in details["paths"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise GleefulInputError(f"Could not read tokenized data details at {details_path}: {exc}") from exc

        if type(self.num_cases) is not int or self.num_cases < 1:
            raise GleefulInputError(f"Invalid example count in {details_path}.")
        if not self.paths or not all(path.is_file() and path.suffix == ".jsonl" for path in self.paths):
            raise GleefulInputError(f"All dataset paths in {details_path} must be existing .jsonl files.")
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_cases

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        rng.shuffle(paths)
        examples = self._read_examples(paths)
        buffer = list(islice(examples, 10_000))
        for example in examples:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example
        rng.shuffle(buffer)
        yield from buffer

    def _read_examples(self, paths):
        count = 0
        for path in paths:
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            count += 1
                            yield json.loads(line)
            except (OSError, ValueError) as exc:
                raise GleefulDatasetError(f"Could not read examples from {path}: {exc}") from exc
        if count != self.num_cases:
            raise GleefulDatasetError(f"Expected {self.num_cases} tokenized examples, found {count}.")
