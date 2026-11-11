"""One cheap model call per template group returns a JSON spec; deterministic executors do the rest."""
import asyncio
import hashlib
import json
import logging
import re

from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable, chat_or_alt
from harness.result import SolveResult
from harness.shape import DESCRIBE, expected_shape, mismatch
from harness.solvers import data_solver, math_solver, search_solver
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

log = logging.getLogger("harness")
MODEL = MODEL_FAST
MAX_TOKENS = 250

_WHOLE = re.compile(r"\{([SEN]\d+)\}")
_ANY = re.compile(r"\{([A-Za-z]+\d+)\}")

_RULES = (
    "Use placeholders {N1}, {S1}, {E1} (exactly as they appear in the task text) instead of concrete values: "
    "{N#} stands for a whole number, {S#} for a quoted string, {E#} for a person or company name. "
    "A placeholder is a JSON string such as \"{N1}\", used as the whole value (in a math step it may also stand inside the expression). "
    "If the task text does not contain enough data to build the specification, return {\"impossible\": true}. "
    "String-valued fields (product, region, entity, file) take {S#} or {E#} placeholders or literal text, never {N#}. "
    "Write the JSON object compactly: no line breaks or indentation (keep spaces inside string values). "
    "Output only the JSON object, nothing else."
)

_SCHEMAS = {
    "math": (
        "You turn a short arithmetic word problem into a JSON specification.\n"
        "Schema: {\"start\": <integer>, \"steps\": [<expression>, ...]}\n"
        "start is the initial quantity. Each step is a string expression over the running value x, applied in "
        "order; the result of a step becomes the new x. Allowed: x, integers (or placeholders), + - * //, "
        "unary minus, parentheses; // is integer division rounding down. Examples: 'loses N boxes' is "
        "\"x-{N1}\"; 'receives N boxes' is \"x+{N1}\"; 'doubled' is \"x*2\"; 'split equally into N parts, "
        "keeps one share rounded down' is \"x//{N1}\"; 'loses a quarter (rounded down)' is \"x-x//4\"; "
        "'loses N percent of the stock (rounded down)' is \"x-x*{N1}//100\" and 'grows by N percent (rounded down)' is "
        "\"x+x*{N1}//100\": multiply first, then divide by 100, never x//100*N; "
        "'ships N per day for M days' is \"x-{N2}*{N3}\"; 'receives N deliveries of M boxes each' is "
        "\"x+{N2}*{N3}\". Full example on one line: "
        "{\"start\":\"{N1}\",\"steps\":[\"x-{N2}*{N3}\",\"x-x//4\",\"x*2\"]}.\n"
        "The specification can only give the quantity left at the end. If the question asks for anything else (how "
        "many were shipped, received or lost in total, a difference between two moments, a number of days), return "
        "{\"impossible\": true}.\n"
    ),
    "data_analysis": (
        "You turn a question about a CSV file into a JSON specification.\n"
        "Schema: {\"file\": <csv file name string>, \"metric\": <metric>, "
        "\"product\": <string or null>, \"region\": <string or null>, "
        "\"month_from\": <integer or null>, \"month_to\": <integer or null>, \"decimals\": <integer, optional>}\n"
        # round 5: a bare {N2} where the schema says <integer> opens an object under guided json_object
        # decoding and the reply degenerates into {"N2":{"N2":...}}; a quoted one-line example prevents it
        "Every placeholder is written as a quoted JSON string, also where an integer is expected: "
        "\"month_from\":\"{N2}\", never \"month_from\":{N2}. Full example on one line: "
        "{\"file\":\"sales.csv\",\"metric\":\"avg_price\",\"product\":\"{S1}\",\"region\":null,"
        "\"month_from\":\"{N2}\",\"month_to\":\"{N2}\",\"decimals\":\"{N3}\"}.\n"
        "metric is one of: \"sum_units\" (total units sold or ordered), \"sum_revenue\" (total of units*price), "
        "\"count_orders\" (number of rows), \"avg_price\" (average price), \"avg_units\" (average units per order), "
        "\"max_units\" (the largest number of units in a single order), "
        "\"min_units\" (the smallest number of units in a single order); "
        "or, for 'which product/region sold the most units / had the most revenue / most orders': "
        "\"top_product_sum_units\", \"top_product_sum_revenue\", \"top_product_count_orders\", "
        "\"top_region_sum_units\", \"top_region_sum_revenue\", \"top_region_count_orders\" "
        "(the answer is the winning product or region name; product/region filters stay null). "
        "decimals is given only when the question asks to round a sum_revenue or avg_* result to N decimal "
        "places; otherwise omit it. "
        "If the question asks for a quantity that none of these metrics can give (for example a total "
        "discount, tax or profit, or a column that is not listed), return {\"impossible\": true}. "
        "product and region filter rows by exact value (case-insensitive); null means no filter. "
        "month_from and month_to are an inclusive month range; give both or set both to null (a single month m "
        "is month_from = month_to = m). file is the literal file name from the text.\n"
    ),
}

# search speaks `hops` (one relation per hop, the last one is the asked field): with path + field the model
# had to move the last relation word into field and often miscounted (research report, round 4)
_SEARCH_PROMPT = (
    "You turn a question about a markdown knowledge base into a JSON specification.\n"
    "Each page describes one person or company with 'Key: value' lines. "
    "Person pages: Born (year), Employer (company name), Mentor (person name). "
    "Company pages: Founded (year), Headquarters (city).\n"
    "Schema: {\"entity\": <name>, \"hops\": [<field>, ...]}\n"
    "entity is the person or company the question starts from: its placeholder ({E#} or {S#}) or, if it has "
    "none, its literal name, exactly as in the task text. hops is the list of fields to read, "
    "starting at the entity and moving outward; each hop reads one field of the page reached by the previous hop; "
    "the LAST hop is the field that is asked for. Write exactly one hop for every relation word in the question, "
    "counting repeats: each 'mentor' / 'mentoring' / 'mentors' / 'mentor of' = Mentor; each 'employer' / "
    "'employs' / 'employed by' / 'works for' / 'payroll' = Employer; 'headquarters' / 'head office' / 'based' / "
    "'base' / 'city' of a company = Headquarters; 'born' / 'birth year' = Born; 'founded' / 'established' = "
    "Founded. The question words 'find the person', 'which person', 'who' add no hop. Mentor is a person; "
    "Employer is a company, so a company is asked for with Employer, never a person.\n"
    "Examples (read inside out): 'Find the person mentoring {E1}' -> {\"entity\":\"{E1}\",\"hops\":[\"Mentor\"]}. "
    "'Who is the mentor of the mentor of {E1}?' -> hops [\"Mentor\",\"Mentor\"]. "
    "'City of the headquarters of the employer of the mentor of {E1}' -> hops "
    "[\"Mentor\",\"Employer\",\"Headquarters\"]. "
    "'Birth year of the mentor of the mentor of the mentor of {E1}' -> hops [\"Mentor\",\"Mentor\",\"Mentor\",\"Born\"].\n"
    "Hops only go forward, from a page to the field it lists: if the question goes the reverse way (whom X mentors, "
    "X's mentee, who works for company C), return {\"impossible\": true}. "
    "If the question cannot be answered this way return {\"impossible\": true}. Placeholders: write {E1} exactly "
    "as in the task text. Write the JSON compactly on one line. Output only the JSON object."
)
_HOPS = {name.casefold(): name for name in ("Born", "Employer", "Mentor", "Founded", "Headquarters")}

# questions that walk a relation backwards (whom X mentors, who works for C): hops cannot express them, and a
# forward reading gives a well-shaped wrong answer, so 2a leaves them to 2b (which can scan the whole KB)
_SLOT = r"\{[ES]\d+\}"
_REVERSE = re.compile("|".join([
    r"\bmentees?\b",
    rf"\bmentored by {_SLOT}",
    rf"{_SLOT} mentor(?:s|ed)?\b(?! by)",                       # does X mentor / X mentors / X mentored
    rf"{_SLOT} (?:is|was|has been) mentoring\b", rf"\b(?:is|was) {_SLOT} mentoring\b",
    rf"\b(?:counts?|counted|regards?|regarded|considers?|considered|treats?|views?|sees?|takes?|names?|lists?|calls?) "
    rf"{_SLOT} as (?:(?:his|her|their|its|a|an|the) )?(?:mentor|employer)\b",   # who counts X as his mentor
    rf"\bcalls? {_SLOT} (?:his|her|their|its) (?:mentor|employer)\b",
    rf"\b(?:has|had|have) {_SLOT} as (?:(?:his|her|their|its|a|an|the) )?(?:mentor|employer)\b",  # X is the mentor
    rf"{_SLOT} (?:is|was) (?:(?:the|a|an) )?mentor of\b",          # X is the mentor of whom
    r"\bwhose (?:mentor|employer)\b",                              # whose mentor is X: always asks for the mentee
    rf"\b(?:works?|working|worked) (?:for|at) {_SLOT}",
    rf"\bemployed (?:by|at) {_SLOT}", rf"{_SLOT} employs?\b", r"\bemployees?\b", rf"\bstaff (?:of|at) {_SLOT}",
]), re.I)


# math 2a: the number of operation sentences in the template against the number of steps in the spec
_START_WORD = re.compile(r"\b(?:start|begin|outset|initial|opening|contain|hold|there (?:are|is)|stocked)", re.I)
_FAMILIES = {  # one operation family per sentence; a sentence with 0 or 2+ families is not counted reliably
    "ship": r"\bships?\b|\bshipped|dispatch|send|sends|go out|goes out|leave the warehouse|gone out",
    "receive": r"deliver|consign|lorr|truck|takes in|\bgets\b|receives? \{N|arrive|incoming|shipments",
    "split": r"split|divid|distribut|equal|cut into|portion|shared out",
    "scale": r"doubl|tripl|twice|three times|merg|absorb|joining|\bjoins?\b|multipl",
    "loss": r"theft|stolen|stole|burglar|burglary|thie|leak|water damage|disappear|destroy|discard|ruin",
    "pct": r"percent|restock",
    "frac": r"\bhalf\b|quarter|fourth|\bthird\b|\bfifth\b|\btenth\b|\bsells?\b|\bsold\b|customers",
}
_FAMILY_RX = {k: re.compile(v, re.I) for k, v in _FAMILIES.items()}
_COMPOUND = re.compile(r"\bplus\b|\band then\b|, then\b|\bthen\b.*\bthen\b", re.I)
_ASKING = re.compile(r"(?:tell|give|determine|find|calculate|state|report|what|how)\b", re.I)


def _families(sentence: str) -> int:
    return sum(1 for rx in _FAMILY_RX.values() if rx.search(sentence))


def _count_operations(template: str) -> int | None:
    """Operation sentences of a math template, or None when the text does not read as intro + operations +
    question, or when any operation sentence is compound (two operation cues, `plus`, `, then`, an operation in
    the intro): then the count would be too low and the step check could punish a correct spec."""
    sentences = split_sentences(strip_boilerplate(template))
    if len(sentences) < 3:
        return None
    last = sentences[-1]
    if not (last.endswith("?") or (_ASKING.match(
            re.sub(r"^finally,?\s+", "", last, flags=re.I)) and not re.search(r"\{N\d+\}", last))):
        return None
    intro = sentences[0]
    if (len(re.findall(r"\{N\d+\}", intro)) != 1 or not _START_WORD.search(intro) or _families(intro)
            or _COMPOUND.search(intro)):
        return None
    middle = sentences[1:-1]
    for sent in middle:
        if (sent.endswith("?") or _ASKING.match(sent) or _COMPOUND.search(sent) or _families(sent) != 1
                or len(re.findall(r"\{N\d+\}", sent)) > 2):
            return None
    return len(middle)


def _step_count(spec: object) -> int | None:
    if not isinstance(spec, dict):
        return None
    for key in ("steps", "ops"):
        if isinstance(spec.get(key), list):
            return len(spec[key])
    return None


def _step_mismatch(task_type: str, template: str, spec: object) -> str | None:
    if task_type != "math":
        return None
    expected, got = _count_operations(template), _step_count(spec)
    if expected is None or got is None or got >= expected:
        return None
    return (f"The text has {expected} operations, your spec has {got} steps; each sentence is a separate step "
            "even if it repeats an earlier one")


def _reverse_relation(template: str) -> str | None:
    m = _REVERSE.search(template)
    return None if m is None else m.group(0)


def _system_prompt(task_type: str) -> str | None:
    if task_type == "search":
        return _SEARCH_PROMPT
    schema = _SCHEMAS.get(task_type)
    return None if schema is None else schema + _RULES


def _from_hops(spec: dict) -> tuple[dict | None, str]:
    """{"entity", "hops"} -> the executor's {"entity", "path": hops[:-1], "field": hops[-1]}.

    A reply without hops (the old path/field shape, impossible, ...) passes through unchanged."""
    if "hops" not in spec or spec.get("impossible") is True:
        return spec, ""
    hops = spec["hops"]
    if not isinstance(hops, list) or not hops:
        return None, "hops must be a non-empty list of field names"
    names = []
    for hop in hops:
        name = _HOPS.get(hop.strip().casefold()) if isinstance(hop, str) else None
        if name is None:
            return None, f"unknown hop {hop!r} in hops (allowed: {', '.join(_HOPS.values())})"
        names.append(name)
    return {"entity": spec.get("entity"), "path": names[:-1], "field": names[-1]}, ""


def _as_reply(task_type: str, spec: dict) -> str:
    """The spec for the follow-up turn's assistant message. search is shown in hops form even when the model
    replied in the old path/field shape: the follow-up then speaks the schema the system prompt asks for."""
    if task_type == "search" and set(spec) == {"entity", "path", "field"} and isinstance(spec["path"], list):
        spec = {"entity": spec["entity"], "hops": [*spec["path"], spec["field"]]}
    return json.dumps(spec, ensure_ascii=False)


def _slot_text(name: str, slots: dict[str, str]) -> str:
    value = slots[name]
    if name[0] == "N":
        try:
            return str(int(value))  # "007" must not become an invalid literal inside an expression
        except ValueError:
            pass
    return value


def fill(spec: object, slots: dict[str, str]) -> object:
    if isinstance(spec, dict):
        return {k: fill(v, slots) for k, v in spec.items()}
    if isinstance(spec, list):
        return [fill(v, slots) for v in spec]
    if isinstance(spec, str):
        m = _WHOLE.fullmatch(spec)
        if m:
            value = slots[m.group(1)]
            return int(value) if m.group(1)[0] == "N" else value
        return _ANY.sub(lambda mm: _slot_text(mm.group(1), slots), spec)
    return spec


_BARE_SLOT = re.compile(r"\{[SEN]\d+\}")


def _quote_bare_placeholders(body: str) -> str:
    """Wrap bare {N#}/{S#}/{E#} in quotes where a JSON value stands: after ':' in an object, or after '['
    or ',' in an array. Text inside strings, keys and real objects is never touched."""
    out, stack, prev, i, n = [], [], "", 0, len(body)
    while i < n:
        ch = body[i]
        if ch == '"':  # copy the whole string literal, escapes included
            j = i + 1
            while j < n and body[j] != '"':
                j += 2 if body[j] == "\\" else 1
            out.append(body[i:j + 1])
            i, prev = j + 1, '"'
            continue
        if ch == "{":
            m = _BARE_SLOT.match(body, i)
            value_pos = prev == ":" or (bool(stack) and stack[-1] == "[" and prev in ("[", ","))
            if m and value_pos:
                out.append(f'"{m.group(0)}"')
                i, prev = m.end(), '"'
                continue
            stack.append("{")
        elif ch == "[":
            stack.append("[")
        elif ch in "]}" and stack:
            stack.pop()
        out.append(ch)
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out)


def _parse_json(text: str) -> tuple[dict | None, str]:
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S | re.I)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except ValueError as e:
        try:  # a bare placeholder value ({"month_from":{N2}}) is invalid JSON; quoted it is the intended spec
            value = json.loads(_quote_bare_placeholders(body))
        except ValueError:
            return None, f"invalid JSON: {e}"
    if not isinstance(value, dict):
        return None, "the JSON value must be an object"
    return value, ""


async def request_spec(task_type: str, template: str, client: LLMClient, model: str = MODEL,
                       history: list[dict] | None = None) -> dict | None:
    """JSON spec for a template; `history` continues an earlier exchange (assistant + user turns)."""
    system = _system_prompt(task_type)
    if system is None:
        return None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": template}] + (history or [])
    for attempt in range(2):
        result = await chat_or_alt(client, model, messages, MAX_TOKENS, response_format={"type": "json_object"})
        spec, error = _parse_json(result.text)
        if spec is not None and task_type == "search":
            spec, error = _from_hops(spec)
        if spec is not None:
            return spec
        messages = messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": f"{error}. Reply again with only a valid JSON object."},
        ]
    return None


def _collapse_products(spec: object) -> object:
    """Math ops may carry a list operand meaning a product; the executor wants one integer."""
    if not isinstance(spec, dict) or not isinstance(spec.get("ops"), list):
        return spec
    ops = []
    for item in spec["ops"]:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            factors = item[1]
            if not factors or not all(isinstance(f, int) and not isinstance(f, bool) for f in factors):
                return spec  # leave malformed; the executor reports unparsed
            prod = 1
            for f in factors:
                prod *= f
            item = [item[0], prod]
        ops.append(item)
    return {**spec, "ops": ops}


def _execute(task: Task, spec: object) -> SolveResult:
    if task.task_type == "math":
        return math_solver.execute(_collapse_products(spec))  # dispatches steps / ops
    if task.task_type == "data_analysis":
        return data_solver.execute(spec, task.workspace, task.check)
    if task.task_type == "search":
        return search_solver.execute(spec, task.workspace)
    return SolveResult("unparsed", note=f"no JSON route for {task.task_type}")


async def solve_group(task_type: str, template: str, members: list[tuple[Task, dict[str, str]]],
                      client: LLMClient, spec: dict | None = None) -> dict[str, SolveResult]:
    """`spec` is a spec already obtained for this template (a batched reply): the first request is skipped, every
    check after it is the same."""
    def all_members(result: SolveResult) -> dict[str, SolveResult]:
        return {task.task_id: result for task, _ in members}

    if _system_prompt(task_type) is None:
        return all_members(SolveResult("unparsed", note=f"no JSON route for {task_type}"))
    reverse = _reverse_relation(template) if task_type == "search" else None
    if reverse is not None:
        return all_members(SolveResult("unparsed", note=f"reverse relation: {reverse!r}"))
    batched = spec is not None
    if spec is None:
        try:
            spec = await request_spec(task_type, template, client)
        except BudgetExhausted:
            raise
        except LLMUnavailable as e:
            return all_members(SolveResult("unparsed", note=f"llm unavailable: {e}"))
    if spec is None:
        return all_members(SolveResult("unparsed", note="model returned no valid JSON spec"))
    log.info("2a spec %s template=%s tasks=%s: %s", "batch" if batched else "single",
             hashlib.sha1(template.encode("utf-8")).hexdigest()[:8], [t.task_id for t, _ in members],
             json.dumps(spec, ensure_ascii=False)[:300])
    if spec.get("impossible") is True:
        return all_members(SolveResult("unparsed", note="model: spec impossible"))
    out = _execute_all(members, spec)
    steps_why = _step_mismatch(task_type, template, spec)
    if steps_why is not None:  # the spec cannot be right for any member of the group
        bad = {t.task_id: f"step count: {steps_why}" for t, _ in members}
    else:
        bad = {tid: why for tid, why in ((t.task_id, _shape_error(t, out[t.task_id])) for t, _ in members) if why}
    if not bad:
        return out
    # one retry on MODEL_ALT for the members whose answer has the wrong shape (or the whole group when the step
    # count is off); then unparsed
    first = next(t for t, _ in members if t.task_id in bad)
    if steps_why is not None:
        feedback = (f"{steps_why}. Re-read the text, count its operation sentences again and reply with only a "
                    "corrected JSON object.")
    else:
        shape = expected_shape(first)
        feedback = (f"With that specification the answer has the wrong kind: the question asks for "
                    f"{DESCRIBE[shape]}, but the result was {str(out[first.task_id].answer)[:60]!r}. Re-read the question "
                    + ("and count its relation words again: one hop per relation word, the last hop is the asked field. "
                       "Reply " if task_type == "search" else "and reply ")
                    + "with only a corrected JSON object.")
    history = [{"role": "assistant", "content": _as_reply(task_type, spec)},
               {"role": "user", "content": feedback}]
    exhausted = None
    try:
        retry = await request_spec(task_type, template, client, MODEL_ALT, history)
    except LLMUnavailable:
        retry = None
    except BudgetExhausted as e:  # keep the good members' answers: the caller settles them, then stops
        retry, exhausted = None, e
    redo = [(t, s) for t, s in members if t.task_id in bad]
    again = (_execute_all(redo, retry) if isinstance(retry, dict) and retry.get("impossible") is not True
             else {})
    for task, _ in redo:
        res = again.get(task.task_id)
        retry_counts = (_step_mismatch(task_type, template, retry) is None
                        and (steps_why is None or _step_count(retry) is not None))
        if res is not None and res.status == "solved" and _shape_error(task, res) is None and retry_counts:
            out[task.task_id] = res
        else:
            # not final, but the latest wrong-shape value stays a best guess for the scheduler's last pass
            guess = res.answer if res is not None and res.status == "solved" else out[task.task_id].answer
            out[task.task_id] = SolveResult("unparsed", note=(bad[task.task_id] if steps_why is not None else f"answer shape: {bad[task.task_id]}"),
                                            extra={"best_guess": guess, "shape_mismatch": True})
    if exhausted is not None:
        exhausted.results = out
        raise exhausted
    return out


_DECIDED = ("reverse relation", "model: spec impossible", "no JSON route")


def _worth_reask(res: SolveResult) -> bool:
    """A batch member that ended unparsed and whose verdict a fresh single call could still change."""
    return (res.status == "unparsed" and not (res.extra or {}).get("shape_mismatch")
            and not (res.note or "").startswith(_DECIDED))


def _shape_error(task: Task, res: SolveResult) -> str | None:
    if res.status != "solved" or res.answer is None:
        return None
    return mismatch(task, str(res.answer))


def _execute_all(members: list[tuple[Task, dict[str, str]]], spec: dict) -> dict[str, SolveResult]:
    out: dict[str, SolveResult] = {}
    for task, slots in members:
        try:
            filled = fill(spec, slots)
            res = _execute(task, filled)
        except Exception as e:  # KeyError for an unknown placeholder, ValueError for a bad number, ...
            out[task.task_id] = SolveResult("unparsed", note=f"json route failed: {e!r}")
            continue
        if (task.task_type == "search" and res.status == "infeasible"
                and (res.note or "").startswith("start entity not found")
                and not _task_value(filled.get("entity"), slots, task.instruction)):
            # the model's (or the templater's) name, not the task's: let 2b look, do not close the task
            res = SolveResult("unparsed", note=f"start entity not found, not a task value: {filled.get('entity')!r}")
        out[task.task_id] = res
    return out


def _task_value(entity: object, slots: dict[str, str], instruction: str) -> bool:
    return isinstance(entity, str) and entity in slots.values() and entity in (instruction or "")


# ------------------------------------------------------------------ batching of small groups (round 6, task 32)
BATCH_MAX = 4
BATCH_MAX_MEMBERS = 2

_BATCH_NOTE = (
    "\n\nThe user message holds several numbered task texts, each with its own placeholders ({N1}, {E1}, ... "
    "restart at 1 in every text). Reply with one JSON object {\"specs\": [{\"i\": 1, \"spec\": <spec 1>}, "
    "{\"i\": 2, \"spec\": <spec 2>}, ...]}: one element per text, in the same order, \"i\" is the number of the text, "
    "each spec exactly as described above (or {\"impossible\": true} for that text). Use only the placeholders "
    "that appear in that text. Write it compactly on one line. Output only that JSON object."
)


_HOP_WORDS = {
    "Mentor": re.compile(r"\bmentor(?:s|ing)?\b", re.I),  # not "mentorship"
    "Employer": re.compile(r"\bemploy(?:er|ers|s|ed)?\b|\bworks? for\b|\bpayroll\b", re.I),
}


def hop_count_mismatch(template: str, spec: object) -> bool:
    """A search spec whose Mentor or Employer hops are not as many as the relation words of its text. The prompt
    asks for one hop per relation word; in a batched reply the model at times copies a hop from a neighbour text
    (expert pass 8: 1 of 6 batched replies added a second Mentor, 6 of 6 single replies were right). Used only to
    send the text to its own call, never to reject a reply to a single call."""
    if not isinstance(spec, dict) or spec.get("impossible") is True:
        return False
    path, field = spec.get("path"), spec.get("field")
    if not isinstance(path, list) or not isinstance(field, str):
        return False
    hops = [h for h in path if isinstance(h, str)] + [field]
    # counted only when the relation words sit in one sentence: a text that lists the steps and then asks the
    # question, or adds a lead-in ("trace the mentorship chain"), tells the chain twice; no such sentence: not counted
    sents = [s for s in split_sentences(template) if any(rx.search(s) for rx in _HOP_WORDS.values())]
    if len(sents) != 1:
        return False
    # a relation the text never names by these words is worded another way ("pays the salary of"): not counted
    counts = {name: len(rx.findall(sents[0])) for name, rx in _HOP_WORDS.items()}
    return any(n and hops.count(name) != n for name, n in counts.items())


def batchable(task_type: str, template: str, members: list) -> bool:
    """Small groups of a type with a spec prompt; reverse-relation search texts never get a call at all."""
    if len(members) > BATCH_MAX_MEMBERS or _system_prompt(task_type) is None:
        return False
    return not (task_type == "search" and _reverse_relation(template) is not None)


def _placeholders(value: object) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_placeholders(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_placeholders(v) for v in value)) if value else set()
    return set(_ANY.findall(value)) if isinstance(value, str) else set()


async def _request_batch(task_type: str, templates: list[str], client: LLMClient) -> list[dict | None] | None:
    """One call for several templates; element k is the spec for template k or None when that element is
    broken: not an object, its echoed index "i" is not k+1 (or is repeated), or it uses a placeholder its own
    template does not have. None for the whole reply when it cannot be read at all."""
    system = _system_prompt(task_type) + _BATCH_NOTE
    user = "\n".join(f"{i}. {t}" for i, t in enumerate(templates, 1))
    result = await chat_or_alt(client, MODEL, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                               MAX_TOKENS * len(templates), response_format={"type": "json_object"})
    value, _ = _parse_json(result.text)
    elements = None if value is None else value.get("specs")
    if not isinstance(elements, list) or len(elements) != len(templates):
        return None
    indexes = [e.get("i") if isinstance(e, dict) else None for e in elements]
    out: list[dict | None] = []
    for k, (element, template) in enumerate(zip(elements, templates)):
        spec = element.get("spec") if isinstance(element, dict) else None
        if indexes[k] != k + 1 or indexes.count(k + 1) != 1 or isinstance(indexes[k], bool):
            spec = None
        if isinstance(spec, dict) and task_type == "search":
            spec, _ = _from_hops(spec)
        if not isinstance(spec, dict) or not _placeholders(spec) <= set(_ANY.findall(template)):
            spec = None
        out.append(spec)
    return out


async def solve_batch(task_type: str, items: list[tuple[str, list[tuple[Task, dict[str, str]]]]],
                      client: LLMClient, on_group=None) -> dict[str, SolveResult]:
    """Groups (template, members) of one task type through one call. A group whose element is broken (or a reply
    that cannot be read) is solved by solve_group with its own call, as without batching. Every returned spec goes
    through the same checks (impossible, execution, shape, step count) as a single-call spec.

    The groups finish concurrently; `on_group(members, results)` is called as soon as one is ready, so a
    cancellation (deadline) loses only the groups still running. The full dict is returned as well."""
    if len(items) == 1:
        try:
            res = await solve_group(task_type, items[0][0], items[0][1], client)
        except BudgetExhausted as e:  # settle what the retry produced, then stop
            if on_group is not None and getattr(e, "results", None):
                on_group(items[0][1], e.results)
            raise
        if on_group is not None:
            on_group(items[0][1], res)
        return res
    out: dict[str, SolveResult] = {}
    try:
        specs = await _request_batch(task_type, [t for t, _ in items], client)
    except BudgetExhausted:
        raise
    except LLMUnavailable:
        specs = None
    specs = specs or [None] * len(items)
    exhausted: list[BudgetExhausted] = []

    def publish(part, res):
        out.update(res)
        if on_group is not None:
            on_group(part, res)

    async def one(template, members, spec):
        if task_type == "search" and spec is not None and hop_count_mismatch(template, spec):
            # the batched reply mixed this text up with a neighbour (a relation too many or too few): its own
            # call, as without batching
            log.info("2a batch spec dropped, hop count differs from the text: %s", json.dumps(spec)[:200])
            spec = None
        try:
            res = await solve_group(task_type, template, members, client, spec=spec)
        except BudgetExhausted as e:  # keep what this group's retry produced; the others finish first
            publish(members, getattr(e, "results", None) or {})
            exhausted.append(e)
            return
        again = [(t, s) for t, s in members if spec is not None and t.task_id in res and _worth_reask(res[t.task_id])]
        if not again:
            publish(members, res)
            return
        ids = {t.task_id for t, _ in again}
        # what the re-ask will not change is handed over first: a deadline cancel during it loses nothing
        publish([m for m in members if m[0].task_id not in ids], {k: v for k, v in res.items() if k not in ids})
        # the batched spec was wrong in a way the checks cannot call broken: its own call, once
        try:
            res2 = await solve_group(task_type, template, again, client)
        except BudgetExhausted as e:
            res2 = {**{k: v for k, v in res.items() if k in ids}, **(getattr(e, "results", None) or {})}
            exhausted.append(e)
        publish(again, res2)

    outcomes = await asyncio.gather(*(one(t, m, s) for (t, m), s in zip(items, specs)),
                                    return_exceptions=True)
    for o in outcomes:  # a non-budget failure of one group surfaces after its siblings have finished
        if isinstance(o, BaseException) and not isinstance(o, BudgetExhausted):
            raise o
    if exhausted:
        exhausted[0].results = out
        raise exhausted[0]
    return out
