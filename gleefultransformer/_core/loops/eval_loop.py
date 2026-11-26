from dataclasses import dataclass


@dataclass(kw_only=True)
class EvalResult:
    loss: float
