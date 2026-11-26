from threading import Event, Thread

from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from gleefultransformer._core.exceptions import GleefulInferenceError
from gleefultransformer._core.gleeful_tasks.args import CHATCallArgs


def format_messages(text):
    if isinstance(text, str):
        messages = [{"role": "user", "content": text}]
    elif isinstance(text, list) and text and all(
        isinstance(message, dict) and isinstance(message.get("role"), str)
        for message in text
    ):
        messages = list(text)
    else:
        raise GleefulInferenceError("Pass a prompt string or a nonempty list of chat messages.")

    return messages


def generate_gen_kwargs(
    input_ids,
    attention_mask,
    streamer: TextIteratorStreamer,
    args: CHATCallArgs,
    eos_token_id: int,
    pad_token_id: int,
    context_limit: int | None = None,
) -> dict:
    max_new_tokens = max(args.min_len, args.max_len)
    if context_limit:
        available = context_limit - input_ids.shape[-1]
        if available < 1:
            raise GleefulInferenceError("The prompt fills the model's context window.")
        max_new_tokens = min(max_new_tokens, available)

    do_sample = args.temp > 0 and args.top_p > 0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "streamer": streamer,
        "num_beams": 1,
        "min_new_tokens": min(args.min_len, max_new_tokens),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": args.temp if do_sample else 1.0,
        "top_k": args.top_k if do_sample else 50,
        "top_p": args.top_p if do_sample else 1.0,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }


class _StopGeneration(StoppingCriteria):
    def __init__(self, stopped: Event):
        self.stopped = stopped

    def __call__(self, input_ids, scores, **kwargs):
        return self.stopped.is_set()


class GenerationThread(Thread):
    def __init__(self, model, kwargs):
        self.stopped = Event()
        kwargs["stopping_criteria"] = StoppingCriteriaList([_StopGeneration(self.stopped)])
        super().__init__(target=model.generate, kwargs=kwargs)
        self.streamer = kwargs["streamer"]
        self.error = None

    def run(self):
        try:
            super().run()
        except Exception as error:
            self.error = error
            self.streamer.end()

    def join(self):
        self.stopped.set()
        super().join()
        if self.error is not None:
            raise self.error
