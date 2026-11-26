import json
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

import gleefultransformer.gleefulsearch.gleeful_search as gleeful_search_module
from gleefultransformer.gleefulsearch import GleefulSearch, SearchCallArgs
from gleefultransformer.gleefulsearch.utils.types import RankerResult


def test_small_index_searches_all_leaves_without_rereading(tmp_path, monkeypatch):
    centroids = torch.eye(2)
    save_file(
        {"ivf0": centroids[:1].clone(), "ivf1": centroids.unsqueeze(0)},
        tmp_path / "centroids.safetensors",
    )
    details = {"embeddings_model": {"type": "sentence_transformers", "model_name": "test"}}
    (tmp_path / "gleeful_details.json").write_text(json.dumps(details))
    total_bytes = 0
    for i, vector in enumerate(([0.5, 0.0], [1.0, 0.0])):
        leaf = tmp_path / "ivf" / "0" / str(i)
        leaf.mkdir(parents=True)
        vectors_path = leaf / "embeddings.safetensors"
        records_path = leaf / "data.jsonl"
        save_file({"embeddings": torch.tensor([vector], dtype=torch.float16)}, vectors_path)
        records_path.write_text(json.dumps({"text": f"passage {i}", "metadata": {"ids": [i]}}))
        total_bytes += vectors_path.stat().st_size + records_path.stat().st_size

    embedding = Mock()
    embedding.encode.return_value = torch.tensor([[1.0, 0.0]])
    monkeypatch.setattr(
        gleeful_search_module, "resolve_embeddings_model", Mock(return_value=embedding)
    )
    monkeypatch.setattr(gleeful_search_module, "es_get_device", lambda: torch.device("cpu"))

    def rank(text, docs, args):
        return [
            RankerResult(
                text=doc.text, score=doc.score, metadata=doc.metadata, best_sentence=doc.text
            )
            for doc in docs[:args.limit]
        ]

    args = SearchCallArgs(leaf_count=1, ranker_count=1)
    cached = GleefulSearch(
        data_name=str(tmp_path), gleeful_ranker=rank, in_memory_max_bytes=total_bytes
    )
    for threshold in (0, total_bytes - 1):
        disk = GleefulSearch(
            data_name=str(tmp_path), gleeful_ranker=rank, in_memory_max_bytes=threshold
        )
        assert disk("query", args)[0].text == "passage 0"

    def unexpected_disk_read(*args):
        pytest.fail("An in-memory query read a leaf from disk")

    monkeypatch.setattr(cached.state_reader, "load_cluster_embeddings", unexpected_disk_read)
    monkeypatch.setattr(cached.state_reader, "load_cluster_records", unexpected_disk_read)
    result = cached("query", args)[0]
    assert result.text == "passage 1"
    assert result.score == 1.0
    result.metadata["ids"].clear()
    assert cached("query", args)[0].metadata == {"ids": [1]}


def test_hub_index_discovery_prefers_current_layout_and_reads_older_layout(tmp_path, monkeypatch):
    from gleefultransformer.gleefulsearch.utils.load_from_repo_or_path import load_from_repo_or_path

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(tmp_path))
    previous = tmp_path / "legacy_index"
    (previous / "ivf").mkdir(parents=True)
    (previous / "centroids.safetensors").touch()
    assert load_from_repo_or_path("example/index", "gleeful_search") == str(previous)

    current = tmp_path / "gleeful_search"
    (current / "ivf").mkdir(parents=True)
    (current / "centroids.safetensors").touch()
    assert load_from_repo_or_path("example/index", "gleeful_search") == str(current)
