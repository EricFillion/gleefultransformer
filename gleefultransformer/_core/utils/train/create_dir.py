import datetime as dt
import logging
import os
import re
from pathlib import Path

from gleefultransformer._core.exceptions import GleefulIOError


def _slugify(name: str) -> str:
    safe_chars = re.compile(r"[^A-Za-z0-9._-]+")
    s = name.strip()
    s = s.replace(" ", "-")
    s = safe_chars.sub("-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-._")
    if not s:
        raise GleefulIOError(f"Invalid directory name: {name}")
    return s


def _increment_path(base_path: Path) -> Path:
    parent = base_path.parent
    name = base_path.name

    pat = re.compile(rf"^{re.escape(name)}(?:-(\d+))?$")

    max_i = -1
    if parent.exists():
        for child in parent.iterdir():
            m2 = pat.match(child.name)
            if m2:
                i = int(m2.group(1)) if m2 and m2.group(1) else 0
                if i > max_i:
                    max_i = i

    if max_i < 0:
        return base_path
    return parent / f"{name}-{max_i + 1}"


def make_dir(name):
    try:
        os.makedirs(name, exist_ok=True)
        logging.info(f"Directory {name} created successfully.")
    except FileExistsError:
        logging.warning(f"Directory {name} already exists.")


def create_tracker_dir(out_dir: str, label: str, run_name: str):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _slugify(run_name) if run_name else f"{label}_{timestamp}"
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    while True:
        directory = _increment_path(output / name)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return str(directory)
