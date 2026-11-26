from gleefultransformer import GleefulChat
from tests.chat import CHAT_CALL_TEXT, CHAT_CALL_ARGS
from tests.functions.save import t_save


def test_save(gleeful_chat, tmp_path):
    t_save(
        gleeful_model=gleeful_chat,
        gleeful_type=GleefulChat,
        call_input=CHAT_CALL_TEXT,
        save_path=str(tmp_path),
        args = CHAT_CALL_ARGS
    )
