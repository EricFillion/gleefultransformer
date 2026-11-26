import pathlib

import pytest

from gleefultransformer.gleefulsearch import  (GleefulRanker,
    SearchCallArgs,
    GleefulSearch,
    SearchTrainArgs)


from gleefultransformer import CHATCallArgs, GleefulChat

from tests.get_test_models import (
    CROSS_ENCODER_PATH,
    EMBEDDING_MODEL_PATH,
)


vector_data = """
{ "text": "Ottawa is the capital of Canada", "metadata": {"id": "a" } }
{ "text": "Kingston was once the capital of Canada", "metadata": {"id": "b" }  }
{ "text": "Toronto is the largest city in Canada", "metadata": {"id": "c" }  }
""".strip()


@pytest.mark.usefixtures("search_checkpoints")
def test_gleeful_search_chat(tmp_path: pathlib.Path, chat_checkpoint):
    (tmp_path / "inputs").mkdir()

    with open(tmp_path / "inputs" / "input_data.jsonl", "w") as f:
        f.write(vector_data.strip())

    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    gleeful_search = GleefulSearch(model_name=EMBEDDING_MODEL_PATH, gleeful_ranker=gleeful_ranker)
    gleeful_search.gleeful_ranker.ignore_original_score = True
    gleeful_search.train(
        train_path=str((tmp_path / "inputs")),
        args=SearchTrainArgs(out_dir=str(tmp_path / "output_0")),
    )
    gleeful_chat = GleefulChat(model_name=chat_checkpoint, gleeful_search=gleeful_search)

    hd_args = SearchCallArgs(
        limit=2, leaf_count=4, ranker_count=2, bs=2
    )
    args = CHATCallArgs(min_len=1, max_len=12, search_args=hd_args)
    result = gleeful_chat("tell me about ottawa", args=args)

    assert type(result.text) == str

    for i, r in enumerate(gleeful_chat.stream("tell me about ottawa", args=args)):
        if i == 0:
            assert r.marker == "search"
        elif i == 1 or i == 2:
            assert r.marker == "search_result"
            assert r.payload["best_sentence"]
            assert r.payload["metadata"]
