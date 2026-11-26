from tests.chat import CHAT_CALL_ARGS, CHAT_CALL_TEXT
from tests.functions.inf import t_inf_algos, t_inf_simple_chat


def test_special(gleeful_chat):
    # list of dictionaries
    input_list_of_dict = [
        {"role": "system", "content": "You are an AI assistant called GleefulChat."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "5"},
    ]

    output_dict = gleeful_chat(input_list_of_dict, args=CHAT_CALL_ARGS)
    assert type(output_dict.text) == str

    # single string
    input_string = "Hi!"
    output_string = gleeful_chat(input_string, args=CHAT_CALL_ARGS)
    assert type(output_string.text) == str


def test_inf_basic(gleeful_chat):
    t_inf_simple_chat(
        gleeful_model=gleeful_chat,
        call_args=CHAT_CALL_ARGS,
        call_text=CHAT_CALL_TEXT,
        output_length=CHAT_CALL_ARGS.max_len,
    )


def test_inf_algos(gleeful_chat):
    t_inf_algos(
        gleeful_model=gleeful_chat, call_args=CHAT_CALL_ARGS, call_text=CHAT_CALL_TEXT
    )
