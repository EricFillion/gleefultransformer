from pathlib import Path

import huggingface_hub


def load_from_repo_or_path(path: str, path_in_repo: str) -> str:
    if Path(path).exists():
        return path

    snapshot = Path(huggingface_hub.snapshot_download(
        repo_id=path, repo_type="dataset", token=True,
    ))
    preferred = snapshot / path_in_repo
    if (preferred / "centroids.safetensors").is_file() and (preferred / "ivf").is_dir():
        return str(preferred)

    # Earlier Hub exports stored the same index layout under a different name.
    candidates = [
        entry.parent for entry in snapshot.glob("*/centroids.safetensors")
        if entry.is_file() and (entry.parent / "ivf").is_dir()
    ]
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        raise ValueError(f"Multiple search indexes found in dataset {path!r}.")
    raise FileNotFoundError(f"No search index found in dataset {path!r}.")
