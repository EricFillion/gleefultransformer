from pathlib import Path

import pytest

from gleefultransformer import GleefulChat, GleefulEvalArgs, GleefulTrainArgs
from tests.get_test_models import CHAT_MODEL_PATH, CROSS_ENCODER_PATH, EMBEDDING_MODEL_PATH


def require_checkpoint(path):
    if not Path(path).is_dir():
        pytest.skip(f"Local model missing at {path}. Run python -m tests.get_test_models.")


@pytest.fixture(scope="session")
def chat_checkpoint():
    require_checkpoint(CHAT_MODEL_PATH)
    return CHAT_MODEL_PATH


@pytest.fixture(scope="session")
def search_checkpoints():
    pytest.importorskip("sentence_transformers")
    require_checkpoint(EMBEDDING_MODEL_PATH)
    require_checkpoint(CROSS_ENCODER_PATH)


@pytest.fixture(scope="module")
def gleeful_chat(chat_checkpoint):
    return GleefulChat(model_name=chat_checkpoint)


@pytest.fixture(scope="module")
def gleeful_chat_train(chat_checkpoint):
    return GleefulChat(model_name=chat_checkpoint)


@pytest.fixture
def chat_train_args(tmp_path):
    return GleefulTrainArgs(save_best=False, checkpoint_steps=-1, out_dir=str(tmp_path / "train"))


@pytest.fixture
def chat_eval_args(tmp_path):
    return GleefulEvalArgs(out_dir=str(tmp_path / "eval"))
