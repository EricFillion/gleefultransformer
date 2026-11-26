from pathlib import Path


def find_artifact(directory: str | Path, filename: str) -> Path:
    path = Path(directory) / filename
    if path.is_file():
        return path

    # Older artifacts use a different prefix for the same metadata format.
    suffix = filename.split("_", 1)[1]
    matches = [item for item in path.parent.glob(f"*_{suffix}") if item.is_file()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise FileExistsError(f"Multiple candidates for {path}")
    return path
