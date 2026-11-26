import logging
from unittest.mock import Mock, call

import numpy as np
import pytest

from gleefultransformer.gleefulsearch.gleeful_ranker import GleefulRanker, RankerCallArgs
from gleefultransformer.gleefulsearch.utils import GleefulDocument


def test_ranker_reuses_duplicate_scores_and_preserves_results():
    first = "This sentence contains more than five words."
    second = "The second sentence is more relevant to the query."
    longer = f"{first} {second}"
    raw_scores = {first: 2.0, second: 5.0, longer: 7.0}
    ranker = GleefulRanker.__new__(GleefulRanker)
    ranker.ignore_original_score = False
    ranker.logger = logging.getLogger(__name__)
    ranker.model = Mock()
    ranker.model.predict.side_effect = lambda pairs, **kwargs: np.array(
        [raw_scores[passage] for _, passage in pairs], dtype=np.float32
    )
    docs = [
        GleefulDocument(text=first, score=0.5, metadata={"id": 1}),
        GleefulDocument(text=longer, score=0.9, metadata={"id": 2}),
        GleefulDocument(text=first, score=0.6, metadata={"id": 3}),
    ]

    results = ranker("query", docs, RankerCallArgs(bs=8, limit=3))

    assert ranker.model.predict.call_args_list == [
        call([("query", first), ("query", longer)], batch_size=8, show_progress_bar=False),
        call([("query", second)], batch_size=8, show_progress_bar=False),
    ]
    assert [result.metadata["id"] for result in results] == [2, 3, 1]
    assert [result.best_sentence for result in results] == [second, first, first]
    assert [result.score for result in results] == pytest.approx([3.14, 1.16, 1.1])

    ranker.model.predict.reset_mock()
    assert ranker("query", [GleefulDocument(text="Too short.")]) == []
    ranker.model.predict.assert_not_called()
