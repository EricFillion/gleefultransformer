import re
from collections.abc import Iterable, Iterator

from gleefultransformer._core.gleeful_tasks.chat_stream_handlers.args import CHATStreamResult

_HARMONY_MARKERS = ("<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|return|>")
_THINK_MARKERS = ("<think>", "</think>")
_COHERE_RESPONSE_TEMPLATE = {
    "start_anchor": "<|CHATBOT_TOKEN|>",
    "fields": {
        "reasoning_content": {"open": "<|START_THINKING|>", "close": "<|END_THINKING|>"},
        "content": {
            "close_pattern": r"<\|(?:START_(?:RESPONSE|TEXT)|END_(?:RESPONSE|TEXT|OF_TURN_TOKEN))\|>",
        },
    },
}


def get_response_parser(tokenizer, chat_template, prefix):
    response_template = getattr(tokenizer, "response_template", None)
    if response_template is None and all(
        marker in chat_template for marker in ("<|CHATBOT_TOKEN|>", "<|START_THINKING|>")
    ):
        response_template = _COHERE_RESPONSE_TEMPLATE
    if response_template is not None:
        return tokenizer.get_response_parser(response_template, prefix=prefix)
    return None


def parsed_stream_events(chunks: Iterable[str], parser) -> Iterator[CHATStreamResult]:
    def events():
        yield from parser.initial_events
        for chunk in chunks:
            yield from parser.feed(chunk)
        _, final_events = parser.finalize()
        yield from final_events

    for event in events():
        field = event["field"]
        kind = event["type"]
        if kind == "region_chunk" and not event["dirty"]:
            if field == "content":
                yield CHATStreamResult(text=event["text"], marker="text")
            elif field == "reasoning_content":
                yield CHATStreamResult(text=event["text"], marker="thinking")
        elif field == "reasoning_content" and kind in {"region_open", "region_close"}:
            yield CHATStreamResult(text="", marker="think_start" if kind == "region_open" else "think_end")


def stream_protocol(template: str) -> str | None:
    if "<|channel|>" in template and "<|message|>" in template:
        return "harmony"
    if all(marker in template for marker in _THINK_MARKERS):
        return "thinking"
    return None


def _split_markers(chunks: Iterable[str], markers: set[str]) -> Iterator[str]:
    pattern = re.compile("|".join(re.escape(marker) for marker in sorted(markers, key=len, reverse=True)))
    prefixes = {
        marker[:length] for marker in markers for length in range(1, len(marker))
    }
    max_prefix = max(map(len, prefixes), default=0)
    pending = ""
    for chunk in chunks:
        pending += chunk
        start = 0
        for match in pattern.finditer(pending):
            if start < match.start():
                yield pending[start:match.start()]
            yield match.group()
            start = match.end()

        remaining = pending[start:]
        suffix_length = next(
            (length for length in range(min(len(remaining), max_prefix), 0, -1)
             if remaining[-length:] in prefixes),
            0,
        )
        emit_length = len(remaining) - suffix_length
        if emit_length:
            yield remaining[:emit_length]
        pending = remaining[emit_length:]
    if pending:
        yield pending


def stream_events(
    chunks: Iterable[str], protocol: str | None, special_tokens: Iterable[str]
) -> Iterator[CHATStreamResult]:
    if protocol is None:
        for chunk in chunks:
            if chunk:
                yield CHATStreamResult(text=chunk, marker="text")
        return

    special_tokens = {token for token in special_tokens if token}
    if protocol == "harmony":
        yield from _harmony_events(_split_markers(chunks, special_tokens | set(_HARMONY_MARKERS)))
        return

    thinking = False
    for piece in _split_markers(chunks, special_tokens | set(_THINK_MARKERS)):
        if piece in _THINK_MARKERS:
            thinking = piece == "<think>"
            marker = "think_start" if thinking else "think_end"
        elif piece in special_tokens:
            marker = "special"
        else:
            marker = "thinking" if thinking else "text"
        yield CHATStreamResult(text=piece, marker=marker)


def _harmony_events(pieces: Iterable[str]) -> Iterator[CHATStreamResult]:
    channel = ""
    reading_channel = False
    channel_parts: list[str] = []
    for piece in pieces:
        if piece == "<|channel|>":
            reading_channel = True
            channel_parts.clear()
            yield CHATStreamResult(text=piece, marker="special")
        elif piece == "<|message|>":
            if reading_channel:
                channel = "".join(channel_parts).strip().partition(" ")[0]
                reading_channel = False
                marker = "think_start" if channel == "analysis" else "special"
                yield CHATStreamResult(text=channel, marker=marker)
            yield CHATStreamResult(text=piece, marker="special")
        elif piece in {"<|end|>", "<|return|>", "<|start|>"}:
            marker = "think_end" if channel == "analysis" else "special"
            channel = ""
            reading_channel = False
            yield CHATStreamResult(text=piece, marker=marker)
        elif reading_channel:
            channel_parts.append(piece)
        elif channel == "analysis":
            yield CHATStreamResult(text=piece, marker="thinking")
        elif channel == "final":
            yield CHATStreamResult(text=piece, marker="text")
