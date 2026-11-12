"""All thresholds and timeouts in one place."""
import logging
import os

# model names: the single place (chosen by measurement, see README)
MODEL_FAST = "qwen3.8-27b-noreason"
MODEL_ALT = "qwen3.6-35b-a3b-noreason"
MODEL_REASON = "qwen3.8-27b"
# 2c arbitration: both are asked independently, the phase-1 choice changes only if they agree
# (round 3: one model alone broke 4 of 42 correct choices on hard coding)
ARBITRATE_MODELS = (MODEL_FAST, MODEL_ALT)

PHASE1_WORKERS = 8
CODING_TIME_LIMIT = 85
PHASE1_TIMEOUT = {"coding": 100, "default": 15}


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return v if v >= 1 else default


# 3 is the dev provider's account ceiling (HTTP 429 above it); the final proxy allows up to 5
LLM_CONCURRENCY = _int_env("HARNESS_LLM_CONCURRENCY", 3)
LLM_TIMEOUT = 180
DEADLINE_S = 3300
STOP_NEW_BEFORE_DEADLINE_S = 150
RESERVE_FRACTION = 0.05
TIMEOUT_CHARGE = 1500  # charged for an LLM call that ended by timeout
MIN_CALL_TOKENS = 100  # a call cut to fit the budget is not made with a smaller output limit
MIN_REASONING_CALL_TOKENS = 1000  # the same for a call with reasoning: below this it reasons and never answers
REASON_FALLBACK_TOKENS = 1500  # output limit of a route's call that fell through to MODEL_REASON, at least
PROGRAM_MAX_TOKENS = 900  # output limit of a 2b program: search programs that read the KB were cut at 400
PROXY_PAUSE_MAX_S = 60

RETRY_DELAYS = (2, 4, 8, 16, 30)  # the shared account ceiling resets within ~40 s

_log = logging.getLogger("harness")
_warned: set[str] = set()


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
        if v != v or v < minimum or v in (float("inf"), float("-inf")):
            raise ValueError(raw)
        return v
    except ValueError:
        if name not in _warned:
            _warned.add(name)
            _log.warning("invalid %s=%r, using default %r", name, raw, default)
        return default


BATCH_2A = True  # default of HARNESS_BATCH_2A (round 6, task 32): measured, no accuracy loss, see the task report


# default of HARNESS_VERIFY_CODING. Round 7, task 38: 0 broken of 221 correct fixes on all sets; 0 fixed on the
# reviewer's set, seed 74 and LIB (none has a wrong green single fix), 2 fixed on a held-out set. On by the lead's
# ruling: hidden tests are probably stronger than the visible ones, see the report
VERIFY_CODING = True
VERIFY_MAX_TOKENS = 400  # output limit of the model writing its own fix, as the first rung of fix_coding


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if name not in _warned:
        _warned.add(name)
        _log.warning("invalid %s=%r, using default %r", name, raw, default)
    return default


def batch_2a() -> bool:
    """Several small 2a groups of one type in one call. Read at call time: HARNESS_BATCH_2A=1/0."""
    return _env_bool("HARNESS_BATCH_2A", BATCH_2A)


MINE_TEXTS = True  # default of HARNESS_MINE_TEXTS


def mine_texts() -> bool:
    """Before phase 1 the coding tasks of the run are read for function texts that several of them hold with green
    tests (`coding_solver.mine_known_texts`); those join the known texts of the phase-1 coding solver. Off: phase 1
    as in round 10. Read at call time: HARNESS_MINE_TEXTS=1/0."""
    return _env_bool("HARNESS_MINE_TEXTS", MINE_TEXTS)


# default of HARNESS_AUDIT_FORMS. Round 14, task 71 (expert 7, idea a): on trap sets 65 -> 83 of 90, v4 coding
# 14 -> 16 of 20, 41 replacements and none wrong; no call on sets without rival forms. See the task report
AUDIT_FORMS = True


def audit_forms() -> bool:
    """The end-of-run audit of coding functions that stand green in rival forms across the task set
    (`harness.audit`). Off: round 13. Read at call time: HARNESS_AUDIT_FORMS=1/0."""
    return _env_bool("HARNESS_AUDIT_FORMS", AUDIT_FORMS)


# default of HARNESS_REVIEW_CODING (expert 9, idea a): two models review every answered coding file for bugs the
# visible tests do not catch (`harness.review`); no call for a file of known texts only
REVIEW_CODING = True


def review_coding() -> bool:
    """The end-of-run review of answered coding files. Read at call time: HARNESS_REVIEW_CODING=1/0."""
    return _env_bool("HARNESS_REVIEW_CODING", REVIEW_CODING)


# default of HARNESS_REVIEW_REASON (round 19, task 82): where FAST and ALT both rewrote a function of the review but
# their rewrites disagree, MODEL_REASON is asked the same; its rewrite and one of theirs must pass the same rules
# (task 87: also where only one of them rewrote it)
REVIEW_REASON = True


def review_reason() -> bool:
    """The review's third opinion on a disagreement. Off: round 18. Read at call time: HARNESS_REVIEW_REASON=1/0."""
    return _env_bool("HARNESS_REVIEW_REASON", REVIEW_REASON)


# default of HARNESS_REVIEW_NAMED (round 19, task 83): a raise counts as named by a visible test also inside a
# `with pytest.raises(X):` / `with self.assertRaises(X):` block that calls the function, and a test line names the
# function only by the whole word (`mean` is not `mean_abs`)
REVIEW_NAMED = True


def review_named() -> bool:
    """The review's reading of which exceptions the visible tests name. Off: round 18 (lines holding the name as a
    substring). Read at call time: HARNESS_REVIEW_NAMED=1/0."""
    return _env_bool("HARNESS_REVIEW_NAMED", REVIEW_NAMED)


# default of HARNESS_REVIEW_METHODS (round 19, task 84): the review also checks methods of classes, on scenarios
# built from the visible tests (construct the objects, call the methods as the tests do, with other values)
REVIEW_METHODS = True


def review_methods() -> bool:
    """The review of class methods. Off: round 18 (a method is never replaced). Read at call time:
    HARNESS_REVIEW_METHODS=1/0."""
    return _env_bool("HARNESS_REVIEW_METHODS", REVIEW_METHODS)


def verify_coding() -> bool:
    """A model checks green single coding fixes on its own. Read at call time: HARNESS_VERIFY_CODING=1/0."""
    return _env_bool("HARNESS_VERIFY_CODING", VERIFY_CODING)


# default of HARNESS_CONFIRM_2A. Round 8, task 40: a 2a answer is checked by an independent MODEL_FAST program; on a
# difference a MODEL_ALT program is the third vote and two votes against 2a replace it. See the task report
CONFIRM_2A = True
# data only. search: a vote cost ~1380 units a task (programs reading the KB), 36K of 43K on a 113-task set, and never
# changed an answer. math: the 2a spec states the rounding (a loss floors the amount removed), while both models' programs
# floored the remainder the same way and outvoted a correct 82920 with 82908 (runs/r5/llm7 math_0008); no math vote ever
# fixed an answer. data: the only measured fixes (data_9007, data_9020), none broken
CONFIRM_TYPES = ("data_analysis",)
# math (round 10, task 56): one MODEL_REASON answer is the vote (expert pass 3: 20 of 20 right, among them the 14
# hardest rounding chains of llm8 and the 3 new-meaning questions the FAST program got wrong, and 82920 on math_0008
# where the two programs said 82908). 2a's answer changes only when a FAST program and then a MODEL_ALT program
# side with REASON against it: REASON and FAST are one set of weights and share readings (llm8 math_0076: both said
# 7978 against a right 9846). The same model answers in 2b a math task that 2a declined. A type in CONFIRM_TYPES
# keeps the two-program vote
CONFIRM_REASON_TYPES = ("math",)
REASON_MATH = True  # default of HARNESS_REASON_MATH; 0 is round 9 (no REASON in 2b math, no math votes)
# the math votes cost ~12K units on llm8 (28 2a math tasks) and changed no answer except one wrongly (math_0076,
# before the ALT vote); all the measured gain is in 2b. Off: math 2a answers are not confirmed (round 9)
CONFIRM_MATH = False  # default of HARNESS_CONFIRM_MATH
CONFIRM_SPEND_LIMIT = 0.5  # no new vote once this share of the budget is spent (as the final REASON pass)
CONFIRM_GROUP_SAMPLE = 2   # a 2a group larger than this is accepted when its best-paid members' FAST votes all agree
VERIFY_UNIT_S = 60         # time set aside per 2d unit (per worker) when the confirmation stage runs before 2d


def reason_math() -> bool:
    """MODEL_REASON answers math in 2b (and votes on 2a math answers when `confirm_math`). Read at call time:
    HARNESS_REASON_MATH=1/0."""
    return _env_bool("HARNESS_REASON_MATH", REASON_MATH)


def confirm_math() -> bool:
    """MODEL_REASON, FAST and ALT votes on 2a math answers (`confirm_reason`); needs `reason_math` too. Read at call
    time: HARNESS_CONFIRM_MATH=1/0."""
    return _env_bool("HARNESS_CONFIRM_MATH", CONFIRM_MATH)


def confirm_2a() -> bool:
    """Majority vote of independent programs on 2a answers. Read at call time: HARNESS_CONFIRM_2A=1/0."""
    return _env_bool("HARNESS_CONFIRM_2A", CONFIRM_2A)


# default of HARNESS_PROSE_AGREE. Round 17, task 79: a 2b answer on a prose knowledge base stands when a MODEL_ALT
# program gives it too; else a MODEL_REASON program is the third attempt (llm_routes.prose_agreement). Round 16 gave
# other answers to byte-identical prompts (runs/r16b, r16c prose2: 26 and 23 of 31). Off: one FAST attempt as in round
# 16, but the multi-hop literal rule stays (on prose2 search_8104 FAST's program then fails: no answer)
PROSE_AGREE = True


def prose_agree() -> bool:
    """Two agreeing attempts on prose answers. Read at call time: HARNESS_PROSE_AGREE=1/0 (the multi-hop literal
    rule stays on either way)."""
    return _env_bool("HARNESS_PROSE_AGREE", PROSE_AGREE)


# default of HARNESS_SECOND_IMPOSSIBLE. Round 19, task 86: MODEL_FAST's 2b program on a search or data task printed
# IMPOSSIBLE (or asked the knowledge base for a page or field it lacks); one MODEL_ALT program on the same prompt, and a
# well-formed answer of it is stored as a guess (far8 search_8002: FAST printed IMPOSSIBLE in 1 run of 9). An answer
# to an infeasible task is not counted and costs nothing but the call
SECOND_IMPOSSIBLE = True


def second_impossible() -> bool:
    """A second opinion on FAST's first IMPOSSIBLE in 2b. Read at call time: HARNESS_SECOND_IMPOSSIBLE=1/0."""
    return _env_bool("HARNESS_SECOND_IMPOSSIBLE", SECOND_IMPOSSIBLE)


# default of HARNESS_DEEP_TESTS. Round 19, task 85: a coding workspace whose test files sit below the root
# (`tests/test_x.py`, `test/...`, `tests/unit/x_test.py`) or whose code under test is a package (`pkg/core.py`,
# `src/pkg/...`) goes to the ladder and the review; off, such a task ends "cannot read workspace" with no call (round 18)
DEEP_TESTS = True


def deep_tests() -> bool:
    """Test discovery below the workspace root and package targets. Read at call time: HARNESS_DEEP_TESTS=1/0."""
    return _env_bool("HARNESS_DEEP_TESTS", DEEP_TESTS)


# default of HARNESS_MINIMAL_PATCH. Round 20, task 88: a green model patch of the ladder gives back the calls of the
# file's own functions it inlined, one line at a time, while the visible tests stay green (far8 coding_8001: the
# visible test needs `transfer` changed, the model also wrote `other.balance += amount` for `other.deposit(amount)`
# and lost b's history entry). Off: round 19
MINIMAL_PATCH = True
# default of HARNESS_MINIMAL_DEFS: before that, every changed definition takes its original text back while the visible
# tests stay green. Off by default: it fired on no recorded task, and it undoes a silent fix the model made in another
# definition (`//` -> `/` in a `mean` the visible test cannot tell)
MINIMAL_DEFS = False
# seconds of extra test runs per task; checked before each run, so the real cap is this plus one test run
MINIMAL_PATCH_S = 20.0


def minimal_patch() -> bool:
    """The cut-back of a green ladder patch. Read at call time: HARNESS_MINIMAL_PATCH=1/0."""
    return _env_bool("HARNESS_MINIMAL_PATCH", MINIMAL_PATCH)


def minimal_defs() -> bool:
    """The whole-definition pass of the cut-back (needs HARNESS_MINIMAL_PATCH). HARNESS_MINIMAL_DEFS=1/0."""
    return _env_bool("HARNESS_MINIMAL_DEFS", MINIMAL_DEFS)


# default of HARNESS_REVIEW_PROBES. Expert 10, patch a: the review's trial inputs reach what the generic battery
# never did (x10: 5 files where FAST and ALT wrote the same right fix and it was kept as "no evidence"):
# variants of a visible string argument (a flipped sign, other numbers in place of its digits), visible arguments built
# by Decimal/Fraction/date/datetime/timedelta, calls that leave out the parameters with defaults (twice: a mutable
# default shows), and an input on which the file's function hangs while both rewrites return the same value. Off: 55354db
REVIEW_PROBES = True


def review_probes() -> bool:
    """The wider trial inputs of the review. Read at call time: HARNESS_REVIEW_PROBES=1/0."""
    return _env_bool("HARNESS_REVIEW_PROBES", REVIEW_PROBES)


def llm_timeout() -> float:
    return _env_float("HARNESS_LLM_TIMEOUT", LLM_TIMEOUT, 0.001)


def deadline_s() -> float:
    return _env_float("HARNESS_DEADLINE_S", DEADLINE_S, 0.0)


def proxy_dead_s() -> float:
    """No successful reply for this long, with failures all along: the proxy is gone. The run deadline by default: at
    the end of the budget the proxy does nothing and the run is stopped, so there is nothing to detect early, while
    giving up on an outage would lose the rest of the hour."""
    return _env_float("HARNESS_PROXY_DEAD_S", deadline_s(), 0.0)


def stop_new_before_deadline_s() -> float:
    return _env_float("HARNESS_STOP_NEW_BEFORE_DEADLINE_S", STOP_NEW_BEFORE_DEADLINE_S, 0.0)


def phase1_timeout(task_type: str) -> float:
    if task_type == "coding":
        return _env_float("HARNESS_PHASE1_TIMEOUT_CODING", PHASE1_TIMEOUT["coding"], 0.001)
    default = _env_float("HARNESS_PHASE1_TIMEOUT_DEFAULT", PHASE1_TIMEOUT["default"], 0.001)
    return PHASE1_TIMEOUT.get(task_type, default) if task_type in PHASE1_TIMEOUT else default


def coding_time_limit() -> float:
    """The coding solver's own limit inside the phase-1 child: 15 s under the kill timeout (85 at the
    default 100), so it stops by itself (tentative answer, arbitration hint) before the parent kills it."""
    t = phase1_timeout("coding")
    return t - 15 if t > 30 else t * 0.5


def retry_delays() -> tuple:
    raw = os.environ.get("HARNESS_RETRY_DELAYS")
    if raw is None or raw.strip() == "":
        return RETRY_DELAYS
    try:
        vals = tuple(float(x) for x in raw.split(","))
        if any(v != v or v < 0 or v == float("inf") for v in vals):
            raise ValueError(raw)
        return vals
    except ValueError:
        if "HARNESS_RETRY_DELAYS" not in _warned:
            _warned.add("HARNESS_RETRY_DELAYS")
            _log.warning("invalid HARNESS_RETRY_DELAYS=%r, using default %r", raw, RETRY_DELAYS)
        return RETRY_DELAYS


def proxy_pause(step: int) -> float:
    """Length of the shared pause after `step` earlier ones of an outage: the retry delays, then doubling."""
    delays = retry_delays()
    if step < len(delays):
        return min(delays[step], PROXY_PAUSE_MAX_S)
    return min(delays[-1] * 2 ** min(step - len(delays) + 1, 20), PROXY_PAUSE_MAX_S)
