from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SolveResult:
    status: Literal["solved", "infeasible", "unparsed"]
    answer: str | None = None
    note: str = ""
    tentative: bool = False
    extra: dict | None = None
