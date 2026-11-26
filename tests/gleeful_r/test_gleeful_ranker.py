import pytest

from gleefultransformer.gleefulsearch import GleefulDocument, GleefulRanker, RankerCallArgs
from tests.get_test_models import CROSS_ENCODER_PATH


@pytest.mark.usefixtures("search_checkpoints")
def test_gleeful_ranker():
    gleeful_ranker = GleefulRanker(model_name=CROSS_ENCODER_PATH)

    text = "Gleeful Transformer "

    best_text = "Gleeful Transformer is a Python package for AI that supports pretraining and fine-tuning LLMs"
    best_metadata = {"best": "metadata"}

    docs = [
        GleefulDocument(
            text="Ottawa is the capital of Canada. ",
            score=0.5,
            metadata={"sample": "metadata"},
        ),
        GleefulDocument(text=best_text, score=1.0, metadata=best_metadata),
        GleefulDocument(
            text="Kingston was once the capital of Canada. ",
            score=0.5,
            metadata={"sample": "metadata"},
        ),
    ]

    ranker_args = RankerCallArgs(bs=32, limit=2)
    result = gleeful_ranker(text=text, docs=docs, args=ranker_args)

    assert len(result) == 2

    for out in result:
        assert type(out.text) == str
        assert type(out.best_sentence) == str
        assert type(out.score) == float
        assert type(out.metadata) == dict

    assert result[0].text == best_text
    assert result[0].metadata == best_metadata
