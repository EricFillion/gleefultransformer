from dataclasses import dataclass, field


@dataclass(kw_only=True)
class CHATStreamResult:
    text: str
    marker: str
    payload: dict = field(default_factory=dict)
