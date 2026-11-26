from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
import os
import pathlib
import time

import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from torch import device

from gleefultransformer._artifact_paths import find_artifact
from gleefultransformer.gleefulsearch.db.gleefuldata_dir import (
    Cluster,
    GleefulSearchDirStateReader,
    RECORD_OFFSETS_FILE,
)
from gleefultransformer.gleefulsearch.train import GleefulKMeans
from gleefultransformer.gleefulsearch.gleeful_ranker import GleefulRanker, RankerCallArgs
from gleefultransformer.gleefulsearch.utils.misc import choose_k0_k1, ensure_readme
from gleefultransformer.gleefulsearch.utils import (
    GleefulVector,
    GleefulTimer,
    es_get_device,
    es_get_logger,
    load_from_repo_or_path,
    EmbeddingsModel,
    GleefulDocument,
    SearchResult
)
from gleefultransformer.gleefulsearch.train import train_gleeful_search

DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_IN_MEMORY_MAX_BYTES = 16 * 1024**2

def resolve_embeddings_model(
    embeddings_model: str | None,
    device: device | None = None,
    bs: int | None = None,
) -> EmbeddingsModel:
    if embeddings_model is None:
        embeddings_model = DEFAULT_EMBEDDINGS_MODEL

    if bs is None:
        bs = 16

    if device is None:
        device = es_get_device()

    return GleefulVector(
        model_name_or_path=embeddings_model, bs=bs, d=device
    )

@dataclass(kw_only=True)
class SearchTrainArgs:
    out_dir: str = "gleeful_search/"
    leaf_size: int = 0
    bs: int = 32

@dataclass(kw_only=True)
class SearchCallArgs:
    limit: int = 1
    leaf_count: int = 32
    ranker_count: int = 4
    bs: int = 32


class GleefulSearch:
    def __init__(
        self,
        *,
        data_name: str | None = None,
        model_name: str | None = None,
        gleeful_ranker: GleefulRanker | None = None,
        in_memory_max_bytes: int = DEFAULT_IN_MEMORY_MAX_BYTES,
    ):
        if in_memory_max_bytes < 0:
            raise ValueError("in_memory_max_bytes must be nonnegative")
        self._in_memory_max_bytes = in_memory_max_bytes
        self._in_memory_cluster: Cluster | None = None
        self.state_reader: GleefulSearchDirStateReader | None = None
        self.ivf0_cluster: GleefulKMeans | None = None
        self.ivf1_clusters: list[GleefulKMeans] = []

        if gleeful_ranker is not None:
            self.gleeful_ranker = gleeful_ranker
        else:
            self.gleeful_ranker = GleefulRanker()

        self.device = es_get_device()
        self.logger = es_get_logger()

        if data_name:
            directory = load_from_repo_or_path(data_name, path_in_repo="gleeful_search")
            reader = GleefulSearchDirStateReader(directory, d=self.device)
            saved_model = reader.load_embeddings_model_name()
            if model_name is None:
                model_name = saved_model
            elif model_name != saved_model:
                self.logger.info(
                    "Using model_name %s for an index built with %s. "
                    "Both paths should point to the same embedding model.",
                    model_name,
                    saved_model,
                )
            self._load_index(reader)

        self.embeddings_model = resolve_embeddings_model(model_name, device=self.device)

    def _load_index(
        self, reader: GleefulSearchDirStateReader, gleeful_timer: GleefulTimer | None = None
    ) -> None:
        centroids = reader.load_centroids()
        k0 = len(centroids.ivf1_centroids)
        k1 = centroids.ivf1_centroids[0].shape[0]
        in_memory_cluster = reader.load_in_memory(k0, k1, self._in_memory_max_bytes)

        ivf0_cluster = GleefulKMeans(n_clusters=k0, gleeful_timer=gleeful_timer)
        ivf0_cluster.centroids = centroids.ivf0_centroids
        ivf1_clusters = []
        for leaf_centroids in centroids.ivf1_centroids:
            cluster = GleefulKMeans(n_clusters=k1, gleeful_timer=gleeful_timer)
            cluster.centroids = leaf_centroids
            ivf1_clusters.append(cluster)

        self.k0, self.k1 = k0, k1
        self.ivf0_cluster = ivf0_cluster
        self.ivf1_clusters = ivf1_clusters
        self._in_memory_cluster = in_memory_cluster
        self.state_reader = reader

    def train(self, train_path: str, args: SearchTrainArgs | None = None) -> None:
        if args is None:
            args = SearchTrainArgs()
        # Train the object, saving output to disk as we go.
        input_count = 0
        input_dir_path = pathlib.Path(train_path)
        out_dir_path = pathlib.Path(args.out_dir)
        timer_path = out_dir_path.joinpath("time")
        gleeful_timer = GleefulTimer(str(timer_path))
        gleeful_timer.report()
        with gleeful_timer.section("file_operations", "Iterate over input_dir"):
            if input_dir_path.is_file() and input_dir_path.suffix == ".jsonl":
                with open(input_dir_path) as f:
                    while line := f.readline():
                        if line.strip():
                            input_count += 1
            else:
                for filepath in input_dir_path.glob("*.jsonl"):
                    with open(filepath) as f:
                        while line := f.readline():
                            if line.strip():
                                input_count += 1

        if input_count == 0:
            raise ValueError("train_path contains no data")

        # we can consider making k0 and k1 configurable in the future.
        # for now the users can just adjust leaf size.
        k0, k1 = choose_k0_k1(
            k0=0, k1=0, input_count=input_count, leaf_size=args.leaf_size
        )

        # Train and save results to disk.
        train_gleeful_search(
            k0=k0,
            k1=k1,
            input_count=input_count,
            input_path=input_dir_path,
            out_dir=out_dir_path,
            embeddings_model=self.embeddings_model,
            bs=args.bs,
            bs_kmeans=args.bs,
            device=self.device,
            logger=self.logger,
            gleeful_timer=gleeful_timer,
        )

        with gleeful_timer.section("file_operations", "Load final clusters"):
            self._load_index(GleefulSearchDirStateReader(out_dir_path, d=self.device), gleeful_timer)
        gleeful_timer.report()

    @torch.no_grad()
    def __call__(self, text: str, args: SearchCallArgs | None = None) -> list[SearchResult]:
        """Find relevant passages for a query."""
        if args is None:
            args = SearchCallArgs()

        reader = self.state_reader
        if reader is None:
            raise ValueError(
                "state_reader is unset. Call train() or load from existing dir?"
            )

        vector = self.embeddings_model.encode([text])  # [1, D]
        vector = vector.to(device=self.device, dtype=torch.float32)

        cached = self._in_memory_cluster
        if cached is not None:
            sims = (vector @ cached.embeddings.T).squeeze(0)
            scores, indices = torch.topk(sims, min(args.ranker_count, sims.numel()))
            documents = [
                GleefulDocument(
                    text=cached.texts[i], score=score, metadata=deepcopy(cached.metas[i])
                )
                for i, score in zip(indices.tolist(), scores.tolist())
            ]
        else:
            documents = self._disk_candidates(vector, args, reader)

        ranker_args = RankerCallArgs(bs=args.bs, limit=args.limit)
        results = self.gleeful_ranker(text=text, docs=documents, args=ranker_args)
        return [
            SearchResult(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                best_sentence=result.best_sentence,
            )
            for result in results
        ]

    def _disk_candidates(
        self, vector: torch.Tensor, args: SearchCallArgs, reader: GleefulSearchDirStateReader
    ) -> list[GleefulDocument]:
        if not self.ivf0_cluster:
            raise ValueError("ivf0_cluster is unset")

        c0_k = max(1, int(math.floor(math.sqrt(args.leaf_count))))
        c1_k = int(math.ceil(args.leaf_count / c0_k))

        # Look into a few top-level clusters.
        c0s = self.ivf0_cluster.predict_multi(vector, k=c0_k)[0].tolist()

        # For each selected top level cluster,
        # look into a few subclusters.
        c0_c1s = [
            self.ivf1_clusters[c0].predict_multi(vector, k=c1_k)[0].tolist()
            for c0 in c0s
        ]

        # Score one leaf at a time and keep only the scores for global selection.
        leaf_scores: list[torch.Tensor] = []
        leaves: list[tuple[int, int]] = []
        row_ends: list[int] = []
        row_end = 0
        for i, c0 in enumerate(c0s):
            for c1 in c0_c1s[i]:
                embeddings = reader.load_cluster_embeddings(c0, c1)
                scores = (
                    vector @ embeddings.to(device=self.device, dtype=torch.float32).T
                ).squeeze(0)
                leaf_scores.append(scores)
                leaves.append((c0, c1))
                row_end += scores.numel()
                row_ends.append(row_end)

        sims = torch.cat(leaf_scores)
        top_scores, top_indices = torch.topk(sims, min(args.ranker_count, sims.numel()))

        # Resolve the selected rows by leaf while keeping their global score order.
        hits: list[tuple[int, int]] = []
        rows_by_leaf: dict[int, list[int]] = defaultdict(list)
        for index in top_indices.tolist():
            leaf = bisect_right(row_ends, index)
            row = index - (row_ends[leaf - 1] if leaf else 0)
            hits.append((leaf, row))
            rows_by_leaf[leaf].append(row)

        records = {
            leaf: reader.load_cluster_records(*leaves[leaf], rows)
            for leaf, rows in rows_by_leaf.items()
        }
        doc_search_results = []
        for (leaf, row), score in zip(hits, top_scores.tolist()):
            record = records[leaf][row]
            doc_search_results.append(
                GleefulDocument(
                    text=record["text"], score=score, metadata=record.get("metadata")
                )
            )

        return doc_search_results

    def push(
        self,
        repo_id: str,
        *,
        bs: int = 4,
        branch: str = "main",
        overwrite: bool = False,
        private: bool = True,
    ) -> None:
        api = HfApi()

        if not self.state_reader:
            raise ValueError("Must have local state to push")

        # Ensure repo exists
        api.create_repo(
            repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset"
        )

        # Ensure branch exists
        api.create_branch(
            repo_id=repo_id,
            repo_type="dataset",
            branch=branch,
            revision="main",
            exist_ok=True,
        )

        hf_paths = set(
            api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=branch)
        )

        root = self.state_reader.root
        index_details = find_artifact(root, "gleeful_details.json")

        def _chunked(groups, size):
            batch = []
            for group in groups:
                if batch and size > 0 and len(batch) + len(group) > size:
                    yield batch
                    batch = []
                batch.extend(group)
            if batch:
                yield batch

        def iter_operations():
            skip_dirs = {
                ".ipynb_checkpoints",
                "time",
                "artifacts",
                "tmp",
                ".cache",
                ".git",
                "__pycache__",
            }

            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]

                for filename in filenames:
                    local_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(local_path, root)
                    if pathlib.Path(local_path) == index_details:
                        rel_path = "gleeful_details.json"

                    path_in_repo = f"gleeful_search/{rel_path}".replace("\\", "/")
                    is_leaf_file = path_in_repo.startswith("gleeful_search/ivf/")
                    if is_leaf_file and filename == RECORD_OFFSETS_FILE:
                        continue

                    if not overwrite and path_in_repo in hf_paths:
                        self.logger.info("Skipping %s", local_path)
                        continue

                    self.logger.info("Adding %s", local_path)
                    operations = [
                        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
                    ]
                    if is_leaf_file and filename == "data.jsonl":
                        # Publish a leaf's text and its offsets in the same commit.
                        offsets_path = os.path.join(dirpath, RECORD_OFFSETS_FILE)
                        offsets_in_repo = path_in_repo.rsplit("/", 1)[0] + "/" + RECORD_OFFSETS_FILE
                        if RECORD_OFFSETS_FILE in filenames:
                            operations.append(
                                CommitOperationAdd(
                                    path_in_repo=offsets_in_repo, path_or_fileobj=offsets_path
                                )
                            )
                        elif offsets_in_repo in hf_paths:
                            operations.append(CommitOperationDelete(path_in_repo=offsets_in_repo))
                    yield operations

        for i, ops_batch in enumerate(_chunked(iter_operations(), bs), start=1):
            self.logger.info("Committing...")
            success = False
            try:
                # first attempt
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=ops_batch,
                    commit_message=f"batch {i}",
                    num_threads=4,
                    revision=branch,
                )
                success = True
            except Exception as e:
                self.logger.warning(
                    f"Failed to push commit. Sleeping for two minutes. {e}"
                )
                sleep(120.0)

            if not success:
                # second attempt
                try:
                    self.logger.info("Trying again")
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=ops_batch,
                        commit_message=f"batch {i} retry",
                        num_threads=2,
                        revision=branch,
                    )
                    success = True
                except Exception as e:
                    self.logger.warning(
                        f"Failed to push commit. Sleeping for five minutes. {e}"
                    )
                    sleep(300.0)

            if not success:
                # third attempt
                try:
                    self.logger.info("Trying again (final attempt)")
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=ops_batch,
                        commit_message=f"batch {i} retry 2",
                        num_threads=1,
                        revision=branch,
                    )
                    success = True
                except Exception as e:
                    raise RuntimeError(
                        "Failed to push commit after two re-tries. Ending"
                    ) from e

            self.logger.info("Success. Sleeping for 0.1 seconds...\n\n")
            sleep(0.1)

        self.logger.info("\n\n\nUploading README")

        try:
            ensure_readme(
                repo_name=repo_id, logger=self.logger, api=api, branch=branch
            )
        except Exception as e:
            self.logger.warning(
                f"Error ensuring README exists, but data uploaded OK: {e}"
            )

        self.logger.info("DONE")


# This wrapper is here for mocking during testing.
def sleep(seconds: float) -> None:
    time.sleep(seconds)
