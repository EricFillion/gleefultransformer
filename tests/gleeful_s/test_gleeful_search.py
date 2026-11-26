import contextlib
import pathlib
from unittest.mock import Mock

import pytest
import torch

from tests.get_test_models import (
    CROSS_ENCODER_PATH,
    EMBEDDING_MODEL_PATH,
)
from gleefultransformer.gleefulsearch import GleefulRanker, GleefulSearch, SearchTrainArgs, SearchCallArgs
from gleefultransformer.gleefulsearch.db.gleefuldata_dir import (
    GleefulSearchDirStateReader,
    RECORD_OFFSETS_FILE,
    write_record_offsets,
)
import gleefultransformer.gleefulsearch.gleeful_search as gleeful_search_module


def pad_or_trim(items: list[float], target_len: int):
    items = items[:target_len]
    items += [0] * (target_len - len(items))
    return items


vector_data = """
{ "text": "Ottawa is the capital of Canada", "metadata": {"id": "a" } }
{ "text": "Kingston was once the capital of Canada", "metadata": {"id": "b" }  }
{ "text": "Toronto is the largest city in Canada", "metadata": {"id": "c" }  }
""".strip()


class MockEmbeddingsModel:
    """Very naive embeddings model that runs super fast."""

    invert: bool

    def __init__(self) -> None:
        self.invert = False

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Literally just number characters. Works in this tiny example.
        emb = torch.tensor(
            [pad_or_trim([float(ord(char)) for char in text], 10) for text in texts]
        )
        if self.invert:
            emb *= -1
        return emb

    def metadata(self):
        return "metadata"


@pytest.mark.usefixtures("search_checkpoints")
def test_train_gleeful_search(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()
    input_str_path = str((tmp_path / "inputs"))
    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
        in_memory_max_bytes=0,
    )
    gleeful_search.train(
        train_path=input_str_path,
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
    )

    # train it twice
    gleeful_search.train(
        input_str_path,
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_1")),
    )

    first_prediction = gleeful_search("ottawa")
    print(f"{first_prediction=}")

    gleeful_search = GleefulSearch(
        data_name=str(tmp_path / "output_1"),
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
        in_memory_max_bytes=0,
    )

    second_prediction = gleeful_search("ottawa")
    print(f"{second_prediction=}")

    assert first_prediction == second_prediction

    offsets = list((tmp_path / "output_1" / "ivf").rglob(RECORD_OFFSETS_FILE))
    assert offsets and sum(path.stat().st_size for path in offsets) == 3 * 8
    for path in offsets:
        path.unlink()

    legacy_search = GleefulSearch(
        data_name=str(tmp_path / "output_1"),
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
        in_memory_max_bytes=0,
    )
    assert legacy_search("ottawa") == first_prediction
    assert not list((tmp_path / "output_1" / "ivf").rglob(RECORD_OFFSETS_FILE))


def test_indexed_record_reads_handle_utf8_and_blank_lines(tmp_path: pathlib.Path):
    leaf = tmp_path / "ivf" / "0" / "0"
    leaf.mkdir(parents=True)
    (leaf / "data.jsonl").write_bytes(
        '{"text":"Québec"}\r\n\r\n{"text":"Vancouver","metadata":{"id":2}}\n'.encode()
    )
    write_record_offsets(leaf)
    reader = GleefulSearchDirStateReader(tmp_path, d=torch.device("cpu"))

    assert reader.load_cluster_records(0, 0, [1]) == {
        1: {"text": "Vancouver", "metadata": {"id": 2}}
    }
    assert reader.load_cluster_records(0, 0, [0]) == {0: {"text": "Québec"}}
    (leaf / RECORD_OFFSETS_FILE).unlink()
    assert reader.load_cluster_records(0, 0, [1]) == {
        1: {"text": "Vancouver", "metadata": {"id": 2}}
    }


@pytest.mark.usefixtures("search_checkpoints")
def test_custom_embedding_model(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=gleeful_ranker)

    gleeful_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    first_prediction = gleeful_search("ottawa")
    print(f"{first_prediction=}")

    gleeful_search = GleefulSearch(
        data_name=str(tmp_path / "output"),
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
    )

    second_prediction = gleeful_search("ottawa",
                                    args=SearchCallArgs(bs=2, limit=1))
    print(f"{second_prediction=}")

    assert first_prediction == second_prediction


@pytest.mark.usefixtures("search_checkpoints")
def test_check_other_clusters(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(
        # Use a non-default model such that this test
        # asserts the same model is used after the load.
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
    )
    # Tell the fine-grained information extractor
    # not to weigh the initial score at all,
    # as we are intentionally screwing it up.
    gleeful_search.gleeful_ranker.ignore_original_score = True

    gleeful_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    first_prediction = gleeful_search("ottawa")
    print(f"{first_prediction=}")

    second_prediction = gleeful_search("ottawa")
    print(f"{second_prediction=}")

    assert first_prediction[0].text == second_prediction[0].text


@pytest.mark.usefixtures("search_checkpoints")
def test_metadata(tmp_path: pathlib.Path):
    """Assert that metadata from training appears in inference query."""
    (tmp_path / "inputs").mkdir()
    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=gleeful_ranker)

    gleeful_search.gleeful_ranker.ignore_original_score = True
    gleeful_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(
            out_dir=str(tmp_path / "output_0"),
        )
    )

    prediction = gleeful_search("ottawa")
    assert prediction[0].text == "Ottawa is the capital of Canada"
    assert prediction[0].metadata == {"id": "a"}



@pytest.mark.usefixtures("search_checkpoints")
def test_push_smoke(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=GleefulRanker(CROSS_ENCODER_PATH))
    gleeful_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:
        gleeful_search.push(repo_id="dontcare", bs=1)

        data_path = "gleeful_search/ivf/0/0/data.jsonl"
        offsets_path = "gleeful_search/ivf/0/0/" + RECORD_OFFSETS_FILE
        batches = [call.kwargs["operations"] for call in api.create_commit.call_args_list]
        assert [data_path, offsets_path] in [
            [operation.path_in_repo for operation in batch] for batch in batches
        ]

        api.create_commit.reset_mock()
        api.list_repo_files = lambda **kwargs: [data_path]
        gleeful_search.push(repo_id="dontcare")
        assert all(
            operation.path_in_repo != offsets_path
            for call in api.create_commit.call_args_list
            for operation in call.kwargs["operations"]
        )

        (tmp_path / "output" / "ivf" / "0" / "0" / RECORD_OFFSETS_FILE).unlink()
        api.create_commit.reset_mock()
        api.list_repo_files = lambda **kwargs: [data_path, offsets_path]
        gleeful_search.push(repo_id="dontcare", overwrite=True, bs=1)
        batches = [call.kwargs["operations"] for call in api.create_commit.call_args_list]
        data_batch = next(batch for batch in batches if batch[0].path_in_repo == data_path)
        assert data_batch[1] == gleeful_search_module.CommitOperationDelete(
            path_in_repo=offsets_path
        )


@contextlib.contextmanager
def mock_hf_api():
    original = gleeful_search_module.HfApi
    try:
        original_sleep = gleeful_search_module.sleep
        gleeful_search_module.sleep = lambda *a, **k: None

        mock = Mock()
        mock.list_repo_files = lambda *a, **k: []
        gleeful_search_module.HfApi = lambda: mock

        yield mock
    finally:
        gleeful_search_module.HfApi = original
        gleeful_search_module.sleep = original_sleep


@pytest.mark.usefixtures("search_checkpoints")
def test_push_exceed_retries(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=GleefulRanker(CROSS_ENCODER_PATH))
    gleeful_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:

        def fn(*a, **k):
            raise Exception("fake error")

        api.create_commit = fn
        with pytest.raises(RuntimeError):
            gleeful_search.push(repo_id="dontcare")


@pytest.mark.usefixtures("search_checkpoints")
def test_push_within_retry(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=GleefulRanker(CROSS_ENCODER_PATH))
    gleeful_search.train(
        train_path=str(tmp_path / "inputs" / "input_data.jsonl"),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output")),
    )

    with mock_hf_api() as api:

        def fn(*a, commit_message: str, **k):
            if "retry" not in commit_message:
                raise Exception("fake error")

        api.create_commit = fn

        # This should succeed since we only err before retry
        gleeful_search.push(repo_id="dontcare")


@pytest.mark.usefixtures("search_checkpoints")
def test_single_file_training(tmp_path: pathlib.Path):
    (tmp_path / "inputs").mkdir()
    tmp_data_path = tmp_path / "inputs" / "input_data.jsonl"
    with open(tmp_data_path, "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=gleeful_ranker)
    gleeful_search.train(
        train_path=str(tmp_data_path),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
    )

    # train it twice
    gleeful_search.train(
        str(tmp_data_path),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_1")),
    )

    first_prediction = gleeful_search("ottawa")
    print(f"{first_prediction=}")

    gleeful_search = GleefulSearch(
        data_name=str(tmp_path / "output_1"),
        model_name=EMBEDDING_MODEL_PATH,
        gleeful_ranker=gleeful_ranker,
    )

    second_prediction = gleeful_search("ottawa")
    print(f"{second_prediction=}")

