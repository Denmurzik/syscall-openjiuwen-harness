"""One cheap model call per template group returns a JSON spec; deterministic executors do the rest."""
import json
import re

from harness.config import MODEL_ALT, MODEL_FAST
from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable
from harness.result import SolveResult
from harness.shape import DESCRIBE, expected_shape, mismatch
from harness.solvers import data_solver, math_solver, search_solver
from harness.tasks import Task

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
        "'ships N per day for M days' is \"x-{N2}*{N3}\"; 'receives N deliveries of M boxes each' is "
        "\"x+{N2}*{N3}\". Full example on one line: "
        "{\"start\":\"{N1}\",\"steps\":[\"x-{N2}*{N3}\",\"x-x//4\",\"x*2\"]}.\n"
    ),
    "data_analysis": (
        "You turn a question about a CSV file into a JSON specification.\n"
        "Schema: {\"file\": <csv file name string>, \"metric\": <metric>, "
        "\"product\": <string or null>, \"region\": <string or null>, "
        "\"month_from\": <integer or null>, \"month_to\": <integer or null>}\n"
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
    "search": (
        "You turn a question about a markdown knowledge base into a JSON specification.\n"
        "Schema: {\"entity\": <name string>, \"path\": [<field>, ...], \"field\": <field>}\n"
        "Each knowledge-base page describes one person or company and has 'Key: value' lines. "
        "Person fields: Born (year), Employer (a company name), Mentor (a person name). "
        "Company fields: Founded (year), Headquarters (city). "
        "entity is the name the question starts from. path lists link fields (Employer, Mentor) followed from the "
        "entity, in order, each yielding another page; use [] when none is needed. field is the final field "
        "read on the last page. Example: 'In which city is the headquarters of X's employer?' is "
        "{\"entity\": \"X\", \"path\": [\"Employer\"], \"field\": \"Headquarters\"}.\n"
    ),
}


def _system_prompt(task_type: str) -> str | None:
    schema = _SCHEMAS.get(task_type)
    return None if schema is None else schema + _RULES


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


def _parse_json(text: str) -> tuple[dict | None, str]:
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S | re.I)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except ValueError as e:
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
        result = await client.chat(model, messages, MAX_TOKENS, response_format={"type": "json_object"})
        spec, error = _parse_json(result.text)
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
                      client: LLMClient) -> dict[str, SolveResult]:
    def all_members(result: SolveResult) -> dict[str, SolveResult]:
        return {task.task_id: result for task, _ in members}

    if _system_prompt(task_type) is None:
        return all_members(SolveResult("unparsed", note=f"no JSON route for {task_type}"))
    try:
        spec = await request_spec(task_type, template, client)
    except BudgetExhausted:
        raise
    except LLMUnavailable as e:
        return all_members(SolveResult("unparsed", note=f"llm unavailable: {e}"))
    if spec is None:
        return all_members(SolveResult("unparsed", note="model returned no valid JSON spec"))
    if spec.get("impossible") is True:
        return all_members(SolveResult("unparsed", note="model: spec impossible"))
    out = _execute_all(members, spec)
    bad = {tid: why for tid, why in ((t.task_id, _shape_error(t, out[t.task_id])) for t, _ in members) if why}
    if not bad:
        return out
    # one retry on MODEL_ALT for the members whose answer has the wrong shape; then unparsed
    first = next(t for t, _ in members if t.task_id in bad)
    shape = expected_shape(first)
    feedback = (f"With that specification the answer has the wrong kind: the question asks for "
                f"{DESCRIBE[shape]}, but the result was {str(out[first.task_id].answer)[:60]!r}. Re-read the question "
                "and reply with only a corrected JSON object.")
    history = [{"role": "assistant", "content": json.dumps(spec, ensure_ascii=False)},
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
        if res is not None and res.status == "solved" and _shape_error(task, res) is None:
            out[task.task_id] = res
        else:
            out[task.task_id] = SolveResult("unparsed", note=f"answer shape: {bad[task.task_id]}")
    if exhausted is not None:
        exhausted.results = out
        raise exhausted
    return out


def _shape_error(task: Task, res: SolveResult) -> str | None:
    if res.status != "solved" or res.answer is None:
        return None
    return mismatch(task, str(res.answer))


def _execute_all(members: list[tuple[Task, dict[str, str]]], spec: dict) -> dict[str, SolveResult]:
    out: dict[str, SolveResult] = {}
    for task, slots in members:
        try:
            out[task.task_id] = _execute(task, fill(spec, slots))
        except Exception as e:  # KeyError for an unknown placeholder, ValueError for a bad number, ...
            out[task.task_id] = SolveResult("unparsed", note=f"json route failed: {e!r}")
    return out
