from dataclasses import dataclass
import json
import pathlib
import struct
import typing
from tempfile import TemporaryDirectory

import safetensors.torch
import torch
from torch import device

from gleefultransformer._artifact_paths import find_artifact


RECORD_OFFSETS_FILE = "data.offsets"
RECORD_OFFSET = struct.Struct("<Q")


def write_record_offsets(cluster_dir: pathlib.Path) -> None:
    """Index the byte position of each nonblank JSONL row in vector order."""
    with TemporaryDirectory(dir=cluster_dir) as staging:
        staged_offsets = pathlib.Path(staging) / RECORD_OFFSETS_FILE
        with (
            (cluster_dir / "data.jsonl").open("rb") as records,
            staged_offsets.open("wb") as offsets,
        ):
            position = 0
            for line in records:
                if line.strip():
                    offsets.write(RECORD_OFFSET.pack(position))
                position += len(line)
        staged_offsets.replace(cluster_dir / RECORD_OFFSETS_FILE)


@dataclass(kw_only=True)
class Cluster:
    embeddings: torch.Tensor
    texts: list[str]
    metas: list[typing.Any]


@dataclass(kw_only=True)
class Centroids:
    ivf0_centroids: torch.Tensor

    ivf1_centroids: list[torch.Tensor]
    """ ivf1_centroids[i] is a tensor of centroids
        for ivf0_c==i. """


class GleefulSearchDirStateReader:
    """Read centroids and document clusters from a saved index."""

    def __init__(self, root: str | pathlib.Path, d: device) -> None:
        self.root = pathlib.Path(root)
        self.device = d

    def load_centroids(self) -> Centroids:
        all_centroids = safetensors.torch.load_file(
            filename=self.root / "centroids.safetensors", device=str(self.device)
        )
        return Centroids(
            ivf0_centroids=all_centroids["ivf0"],
            ivf1_centroids=list(all_centroids["ivf1"]),
        )

    def _cluster_dir(self, ivf0_c: int, ivf1_c: int) -> pathlib.Path:
        return self.root / "ivf" / str(ivf0_c) / str(ivf1_c)

    def load_cluster_embeddings(self, ivf0_c: int, ivf1_c: int) -> torch.Tensor:
        return safetensors.torch.load_file(
            filename=self._cluster_dir(ivf0_c, ivf1_c) / "embeddings.safetensors",
            device=str(self.device),
        )["embeddings"]

    def load_cluster_records(
        self, ivf0_c: int, ivf1_c: int, rows: list[int]
    ) -> dict[int, dict[str, typing.Any]]:
        """Read the requested records for a leaf in embedding row coordinates."""
        cluster_dir = self._cluster_dir(ivf0_c, ivf1_c)
        found: dict[int, dict[str, typing.Any]] = {}
        with (cluster_dir / "data.jsonl").open("rb") as records:
            try:
                offsets = (cluster_dir / RECORD_OFFSETS_FILE).open("rb")
            except FileNotFoundError:
                # Older indexes have the same row order but no byte offsets.
                wanted = set(rows)
                for row, line in enumerate(line for line in records if line.strip()):
                    if row in wanted:
                        found[row] = json.loads(line)
                        wanted.remove(row)
                        if not wanted:
                            break
                return found

            with offsets:
                for row in sorted(rows):
                    offsets.seek(row * RECORD_OFFSET.size)
                    (position,) = RECORD_OFFSET.unpack(offsets.read(RECORD_OFFSET.size))
                    records.seek(position)
                    found[row] = json.loads(records.readline())
                return found

    def load_cluster(self, ivf0_c: int, ivf1_c: int) -> Cluster:
        """Load a specific leaf cluster into memory, including all embeddings and texts."""
        embeddings = self.load_cluster_embeddings(ivf0_c, ivf1_c)
        cluster_dir = self._cluster_dir(ivf0_c, ivf1_c)
        with (cluster_dir / "data.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        texts = [record["text"] for record in records]
        metas = [record.get("metadata") for record in records]

        return Cluster(embeddings=embeddings, texts=texts, metas=metas)

    def load_in_memory(self, k0: int, k1: int, max_bytes: int) -> Cluster | None:
        """Load all leaves if their serialized size is within the threshold."""
        if max_bytes == 0:
            return None

        leaves = []
        total_bytes = 0
        for c0 in range(k0):
            for c1 in range(k1):
                cluster_dir = self._cluster_dir(c0, c1)
                try:
                    total_bytes += (cluster_dir / "embeddings.safetensors").stat().st_size
                    total_bytes += (cluster_dir / "data.jsonl").stat().st_size
                except FileNotFoundError:
                    return None
                if total_bytes > max_bytes:
                    return None
                leaves.append((c0, c1))

        embeddings = []
        texts = []
        metas = []
        for c0, c1 in leaves:
            cluster = self.load_cluster(c0, c1)
            embeddings.append(cluster.embeddings.to(dtype=torch.float32))
            texts.extend(cluster.texts)
            metas.extend(cluster.metas)

        return Cluster(embeddings=torch.cat(embeddings), texts=texts, metas=metas)

    def load_embeddings_model_name(self) -> str:
        with find_artifact(self.root, "gleeful_details.json").open(encoding="utf-8") as f:
            config = json.load(f)["embeddings_model"]

        if config["type"] != "sentence_transformers":
            raise ValueError("expected type = sentence_transformers")

        return config["model_name"]
