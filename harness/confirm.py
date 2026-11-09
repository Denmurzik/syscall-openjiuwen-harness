"""Confirmation of a 2a answer by a majority of independent votes (round 8, task 40).

2a's answer comes from a model spec run by an exact executor. A MODEL_FAST program written from the task text alone
(it never sees the spec or the answer) is the second vote; when it differs, a MODEL_ALT program is the third. 2a's
answer is replaced only when the two programs agree with each other on another answer of the right shape. A vote that
crashes, times out, prints IMPOSSIBLE or the wrong shape does not count, and without a majority 2a stays.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from harness import config
from harness.shape import mismatch
from harness.trajectory import CALLS

log = logging.getLogger("harness")


@dataclass(frozen=True)
class Verdict:
    replacement: str | None   # the answer to write instead of 2a's, or None: 2a stays
    note: str
    fast_agrees: bool         # the FAST program gave 2a's answer (a group sample counts only then)


def same_answer(a: str, b: str) -> bool:
    """Numbers compare as numbers (12.50 = 12.5), anything else without case and extra spaces."""
    a, b = " ".join(str(a).split()), " ".join(str(b).split())
    try:
        x, y = Decimal(a), Decimal(b)
        if x.is_nan() or y.is_nan():
            return x.is_nan() and y.is_nan()
        return x == y
    except (InvalidOperation, ValueError):
        return a.casefold() == b.casefold()


async def _vote(program, task, client, model: str) -> tuple[str | None, set]:
    """One program on `model` (its shape retry too): its answer, or None when it does not count, and the models
    whose replies it used (a failing model falls through to another one)."""
    from harness.llm import BudgetExhausted
    calls = CALLS.get()
    start = len(calls) if calls is not None else 0
    try:
        res = await program(task, client, model=model, shape_model=model)
    except BudgetExhausted:
        raise
    except Exception as e:  # the vote is lost, 2a's answer is not
        log.warning("confirm vote %s on %s failed: %r", model, task.task_id, e)
        res = None
    used = {c.get("model") for c in (calls[start:] if calls is not None else []) if not c.get("error")}
    answer = res.answer if res is not None and res.status == "solved" else None
    if answer is None or answer.strip() == "" or answer.strip() == "IMPOSSIBLE" or mismatch(task, answer) is not None:
        return None, used
    return answer.strip(), used


async def confirm(task, answer: str, client, program, can_afford) -> Verdict:
    """`program(task, client, model=, shape_model=)` is the 2b program route; `can_afford()` says whether the third
    vote may be bought."""
    fast, fast_models = await _vote(program, task, client, config.MODEL_FAST)
    if fast is None:
        return Verdict(None, "confirm: fast vote did not count, 2a kept", False)
    if same_answer(fast, answer):
        return Verdict(None, "confirm: fast agrees with 2a", True)
    if not can_afford():
        return Verdict(None, f"confirm: fast {fast!r} differs, no budget for the third vote, 2a kept", False)
    alt, alt_models = await _vote(program, task, client, config.MODEL_ALT)
    if alt is None:
        return Verdict(None, f"confirm: fast {fast!r} differs, alt vote did not count, 2a kept", False)
    if same_answer(alt, answer):
        return Verdict(None, f"confirm: fast {fast!r} differs, alt sides with 2a", False)
    if not same_answer(alt, fast):
        return Verdict(None, f"confirm: three different answers (fast {fast!r}, alt {alt!r}), 2a kept", False)
    if fast_models & alt_models:
        return Verdict(None, f"confirm: fast and alt votes not independent ({sorted(fast_models & alt_models)}), 2a kept",
                       False)
    return Verdict(fast, f"confirm: fast and alt agree on {fast!r} against 2a {answer!r}", False)
