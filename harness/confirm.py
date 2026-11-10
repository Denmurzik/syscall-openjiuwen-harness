"""Confirmation of a 2a answer by a majority of independent votes (round 8, task 40).

2a's answer comes from a model spec run by an exact executor. A MODEL_FAST program written from the task text alone
(it never sees the spec or the answer) is the second vote; when it differs, a MODEL_ALT program is the third. 2a's
answer is replaced only when the two programs agree with each other on another answer of the right shape. A vote that
crashes, times out, prints IMPOSSIBLE or the wrong shape does not count, nor one whose output is not a single bare
value in canonical form (see `_canonical`), nor a second vote answered by the same weights as the first; numbers
compare at 2a's precision. Without a majority
2a stays.

math (round 10, task 56, `confirm_reason`): the second vote is MODEL_REASON's own answer and the third a MODEL_FAST
program; 2a's answer is replaced only when those two agree.
"""
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from harness import config
from harness.shape import mismatch
from harness.trajectory import CALLS

log = logging.getLogger("harness")
_PLAIN = re.compile(r"-?\d+(?:\.\d+)?")


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


def family(model: str | None) -> str:
    """The weights behind a model name: MODEL_REASON is MODEL_FAST with reasoning on (qwen3.8-27b and its -noreason)."""
    m = str(model or "")
    return m[:-len("-noreason")] if m.endswith("-noreason") else m


_INT_LINE = re.compile(r"-?\d+(?:\.0+)?")
# a text value that is not a name (a region, a product): words only, no label, markup or closing period
_TEXT_LINE = re.compile(r"[^\W_][\w'&-]*(?: [\w'&-]+){0,4}")


def _canonical(task, answer: str, line: str, answer_2a: str) -> bool:
    """A vote counts only in the canonical form of a bare answer: `line` is the program's only output line. exact_int:
    the whole line an integer (24, 24.0; not 12.6, 2e+19, 'checked 2 steps'). A plain number for 2a: a plain number
    (not 1,468.50, $1468.50, 100.0%). A name: the bare-answer name check. Other text: words only (not 'Region: east')."""
    line = line.strip()
    if task.check == "exact_int":
        return bool(_INT_LINE.fullmatch(line))
    if _PLAIN.fullmatch(answer_2a.strip()):
        return bool(_PLAIN.fullmatch(line))
    if _PLAIN.fullmatch(answer):
        return bool(_PLAIN.fullmatch(line))
    from harness.llm_routes import _looks_like_name
    from harness.shape import NAME, expected_shape
    if expected_shape(task) == NAME:
        return _looks_like_name(answer)
    return bool(_TEXT_LINE.fullmatch(answer))


def _places(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def at_precision(task, answer: str, answer_2a: str = "") -> str:
    """A number at 2a's precision, by the phase-1 executor's rule (`_fmt`): the places the question asks for, else two
    with trailing zeros stripped, and never fewer places than 2a's answer shows (a spec may carry decimals=3 for a
    phrase the regex misses). Float noise (0.30000000000000004) then reads 0.3; other answers stay as they are."""
    if task.check == "exact_int" or not _PLAIN.fullmatch(answer):
        return answer
    from harness.llm_routes import _requested_decimals
    from harness.solvers.data_solver import _fmt
    asked = _requested_decimals(task.instruction)
    shown = _places(answer_2a.strip()) if _PLAIN.fullmatch(answer_2a.strip()) else 0
    if asked is None and shown <= 2:
        places = None   # two places, trailing zeros stripped: what the executor writes
    else:
        places = max(asked or 0, shown, 2 if asked is None else 0)
    try:
        return _fmt(Decimal(answer), "exact_str", places)
    except (InvalidOperation, ValueError):
        return answer


async def _vote(program, task, client, model: str, answer_2a: str) -> tuple[str | None, set]:
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
    used = {family(c.get("model")) for c in (calls[start:] if calls is not None else []) if not c.get("error")}
    answer = res.answer if res is not None and res.status == "solved" else None
    if answer is None or answer.strip() == "" or answer.strip() == "IMPOSSIBLE" or mismatch(task, answer) is not None:
        return None, used
    answer = answer.strip()
    lines = [ln for ln in str((res.extra or {}).get("stdout", answer)).splitlines() if ln.strip()]
    # one bare line only: two programs share habits (a debug line, a label, thousands commas, a currency sign), so
    # agreeing on a non-canonical print says nothing about the value
    if len(lines) != 1 or not _canonical(task, answer, lines[0], answer_2a):
        return None, used
    return at_precision(task, answer, answer_2a), used


def _is_number(text: str) -> bool:
    try:
        return Decimal(" ".join(str(text).split())).is_finite()
    except (InvalidOperation, ValueError):
        return False


def _is_zero(text: str) -> bool:
    return _is_number(text) and Decimal(" ".join(str(text).split())) == 0


async def confirm(task, answer: str, client, program, can_afford) -> Verdict:
    """`program(task, client, model=, shape_model=)` is the 2b program route; `can_afford()` says whether the third
    vote may be bought."""
    fast, fast_models = await _vote(program, task, client, config.MODEL_FAST, answer)
    if fast is None:
        return Verdict(None, "confirm: fast vote did not count, 2a kept", False)
    if same_answer(fast, answer):
        return Verdict(None, "confirm: fast agrees with 2a", True)
    if not can_afford():
        return Verdict(None, f"confirm: fast {fast!r} differs, no budget for the third vote, 2a kept", False)
    alt, alt_models = await _vote(program, task, client, config.MODEL_ALT, answer)
    if alt is None:
        return Verdict(None, f"confirm: fast {fast!r} differs, alt vote did not count, 2a kept", False)
    if same_answer(alt, answer):
        return Verdict(None, f"confirm: fast {fast!r} differs, alt sides with 2a", False)
    if not same_answer(alt, fast):
        return Verdict(None, f"confirm: three different answers (fast {fast!r}, alt {alt!r}), 2a kept", False)
    if fast_models & alt_models:   # the same weights answered both votes (after a fallback): the second does not count
        return Verdict(None, f"confirm: alt vote did not count, not independent of fast ({sorted(fast_models & alt_models)}), "
                       "2a kept", False)
    if _is_zero(fast) and not _is_zero(answer) and _is_number(answer):
        # two programs sharing one habit (an empty selection, e.g. a case-sensitive filter) is no evidence for zero
        return Verdict(None, f"confirm: both votes {fast!r} against non-zero 2a {answer!r}: kept 2a", False)
    return Verdict(fast, f"confirm: fast and alt agree on {fast!r} against 2a {answer!r}", False)


async def confirm_reason(task, answer: str, client, reason, program, can_afford) -> Verdict:
    """math (round 10, task 56): `reason(task, client)` is the MODEL_REASON route; it never sees 2a's answer. It
    agrees: 2a kept. It differs: a MODEL_FAST program (`program`, as in `confirm`) is the third vote, and 2a's answer
    is replaced only when that program gives REASON's answer. One REASON answer alone never changes 2a."""
    from harness.llm import BudgetExhausted
    try:
        res = await reason(task, client)
    except BudgetExhausted:
        raise
    except Exception as e:  # the vote is lost, 2a's answer is not
        log.warning("confirm reason vote on %s failed: %r", task.task_id, e)
        res = None
    vote = str(res.answer).strip() if res is not None and res.status == "solved" and res.answer is not None else ""
    if vote in ("", "IMPOSSIBLE") or mismatch(task, vote) is not None:
        return Verdict(None, "confirm: reason vote did not count, 2a kept", False)
    vote = at_precision(task, vote, answer)
    if same_answer(vote, answer):
        return Verdict(None, "confirm: reason agrees with 2a", True)
    if not can_afford():
        return Verdict(None, f"confirm: reason {vote!r} differs, no budget for the third vote, 2a kept", False)
    calls = CALLS.get()
    start = len(calls) if calls is not None else 0
    third, _ = await _vote(program, task, client, config.MODEL_FAST, answer)
    if third is not None and any(c.get("model") == config.MODEL_REASON and not c.get("error")
                                 for c in (calls[start:] if calls is not None else [])):
        # the FAST program fell through to MODEL_REASON (FAST failing): the same model twice is one vote
        return Verdict(None, f"confirm: reason {vote!r} differs, third vote answered by {config.MODEL_REASON}, "
                       "2a kept", False)
    if third is not None and same_answer(third, vote):
        return Verdict(vote, f"confirm: reason and fast program agree on {vote!r} against 2a {answer!r}", False)
    return Verdict(None, f"confirm: reason {vote!r} differs, fast program {third!r} does not side with it, 2a kept",
                   False)
