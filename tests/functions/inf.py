import copy

from transformers import PreTrainedTokenizer

from gleefultransformer import CHATCallArgs, GleefulChat


def _check_length_and_type(
    text: str, tokenizer: PreTrainedTokenizer, token_length: int
):
    assert type(text) == str
    tokens = tokenizer.encode(text, return_tensors="pt")
    print(text)
    length = tokens.size(1)
    print(length,  token_length)
    assert length == token_length


def t_inf_simple_chat(
    gleeful_model: GleefulChat,
    call_args: CHATCallArgs,
    call_text: str,
    output_length: int,
):
    # str
    call_output = gleeful_model(call_text, args=call_args)
    call_str = call_output.text

    _check_length_and_type(
        call_str, gleeful_model.tokenizer, output_length
    )


def t_inf_algos(gleeful_model: GleefulChat, call_args: CHATCallArgs, call_text: str):
    greedy_settings = copy.deepcopy(call_args)
    output_greedy = gleeful_model(call_text, args=greedy_settings)

    temperature_sampling_settings = copy.deepcopy(call_args)
    temperature_sampling_settings.top_k = 0
    temperature_sampling_settings.temp = 0.7

    output_temperature_sampling = gleeful_model(call_text, args=temperature_sampling_settings)

    top_k_sampling_settings = copy.deepcopy(call_args)

    top_k_sampling_settings.top_k = 50
    top_k_sampling_settings.temp = 0.7

    output_top_k_sampling = gleeful_model(call_text, args=top_k_sampling_settings)

    top_p_sampling_settings = copy.deepcopy(call_args)
    top_p_sampling_settings.top_k = 0
    top_p_sampling_settings.top_p = 0.8
    top_p_sampling_settings.temp = 0.7

    output_top_p_sampling = gleeful_model(call_text, args=top_p_sampling_settings)

    assert type(output_greedy.text) == str
    assert type(output_temperature_sampling.text) == str
    assert type(output_top_k_sampling.text) == str
    assert type(output_top_p_sampling.text) == str
