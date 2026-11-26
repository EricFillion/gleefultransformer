import json
import os

from datasets import Dataset

from gleefultransformer._artifact_paths import find_artifact
from gleefultransformer._core.exceptions import GleefulDatasetError, GleefulInputError, GleefulIOError


def tok_dir_to_dataset(tok_dir: str):
    # Validate the input directory
    if not os.path.isdir(tok_dir):
        raise GleefulInputError(f"Invalid tok_dir: {tok_dir} is not a directory.")

    try:
        gleeful_details_path = find_artifact(tok_dir, "gleeful_details.json")
        with gleeful_details_path.open(encoding="utf-8") as f:
            gleeful_details = json.load(f)
    except Exception as e:
        raise GleefulIOError(f"Unable to read dataset details in '{tok_dir}': {e}")

    # List of full paths to all JSONL files
    full_paths = gleeful_details.get("paths", [])
    if not full_paths:
        raise GleefulDatasetError(f"No file paths found in {gleeful_details_path}.")

    # Load all entries from each JSONL file
    all_data = []
    for path in full_paths:
        try:
            with open(path, "r") as f:
                for line in f:
                    all_data.append(json.loads(line))
        except Exception as e:
            raise GleefulIOError(f"Unable to read '{path}': {e}")

    # Convert list of dicts to Hugging Face Dataset
    try:
        dataset = Dataset.from_list(all_data)
    except Exception as e:
        raise GleefulDatasetError(f"Unable to create dataset: {e}")

    return dataset
