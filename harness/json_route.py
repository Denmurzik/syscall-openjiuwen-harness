"""One cheap model call per template group returns a JSON spec; deterministic executors do the rest."""
import json
import re

from harness.llm import BudgetExhausted, LLMClient, LLMUnavailable
from harness.result import SolveResult
from harness.solvers import data_solver, math_solver, search_solver
from harness.tasks import Task

MODEL = "qwen3.6-35b-a3b-noreason"
MAX_TOKENS = 250

_WHOLE = re.compile(r"\{([SEN]\d+)\}")
_ANY = re.compile(r"\{([A-Za-z]+\d+)\}")

_RULES = (
    "Use placeholders {N1}, {S1}, {E1} (exactly as they appear in the task text) instead of concrete values: "
    "{N#} stands for a whole number, {S#} for a quoted string, {E#} for a person or company name. "
    "A placeholder is a JSON string such as \"{N1}\", used as the whole value. "
    "If the task text does not contain enough data to build the specification, return {\"impossible\": true}. "
    "String-valued fields (product, region, entity, file) take {S#} or {E#} placeholders or literal text, never {N#}. "
    "Output only the JSON object, nothing else."
)

_SCHEMAS = {
    "math": (
        "You turn a short arithmetic word problem into a JSON specification.\n"
        "Schema: {\"start\": <integer>, \"ops\": [[<op>, <integer>], ...]}\n"
        "start is the initial quantity. ops are applied in order to the running value; op is one of "
        "\"+\", \"-\", \"*\", \"//\" (integer division rounding down). Each operand is one integer "
        "(or one placeholder), or a list of integers/placeholders meaning their product; no other expressions. Examples: 'it loses N boxes' is [\"-\", N]; 'receives N boxes' is "
        "[\"+\", N]; 'doubled' is [\"*\", 2]; 'split equally into N parts, keeps one share rounded down' is "
        "[\"//\", N]. 'ships N per day for M days' is [\"-\", [\"{N2}\", \"{N3}\"]] (a list operand is a product); "
        "'receives N deliveries of M boxes each' is [\"+\", [N, M]].\n"
    ),
    "data_analysis": (
        "You turn a question about a CSV file into a JSON specification.\n"
        "Schema (exactly these six keys): {\"file\": <csv file name string>, \"metric\": <metric>, "
        "\"product\": <string or null>, \"region\": <string or null>, "
        "\"month_from\": <integer or null>, \"month_to\": <integer or null>}\n"
        "metric is one of: \"sum_units\" (total units sold or ordered), \"sum_revenue\" (total of units*price), "
        "\"count_orders\" (number of rows), \"avg_price\" (average price), \"avg_units\" (average units per order). "
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
        return _ANY.sub(lambda mm: slots[mm.group(1)], spec)
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


async def request_spec(task_type: str, template: str, client: LLMClient) -> dict | None:
    system = _system_prompt(task_type)
    if system is None:
        return None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": template}]
    for attempt in range(2):
        result = await client.chat(MODEL, messages, MAX_TOKENS, response_format={"type": "json_object"})
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
        return math_solver.execute(_collapse_products(spec))
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
    out: dict[str, SolveResult] = {}
    for task, slots in members:
        try:
            out[task.task_id] = _execute(task, fill(spec, slots))
        except Exception as e:  # KeyError for an unknown placeholder, ValueError for a bad number, ...
            out[task.task_id] = SolveResult("unparsed", note=f"json route failed: {e!r}")
    return out
