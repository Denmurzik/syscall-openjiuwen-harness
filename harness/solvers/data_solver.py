import csv
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from harness.result import SolveResult
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

_FLAGS = re.IGNORECASE
_FILE = re.compile(r"The file (\S+\.csv) has columns ([\w,]+)(?: \(\d+ rows\))?\.", _FLAGS)

# Filter fragments, removed from the question text before the metric is matched.
_PRODUCT = re.compile(r"\s+of product '(\w+)'", _FLAGS)
_REGION = re.compile(r"\s+in region '(\w+)'", _FLAGS)
_MONTHS = re.compile(r"\s+in months? (\d+) through (\d+) inclusive", _FLAGS)
_MONTH = re.compile(r"\s+in month (\d+)(?!\d)", _FLAGS)

_METRICS = [
    ("sum_units", re.compile(r"how many units were sold\?", _FLAGS)),
    ("sum_revenue", re.compile(r"what (?:is|was) the total revenue\?", _FLAGS)),
    ("count_orders", re.compile(r"how many orders were (?:placed|made|recorded)\?", _FLAGS)),
    ("avg_price", re.compile(r"what (?:is|was) the average price\?", _FLAGS)),
    ("avg_units", re.compile(r"what (?:is|was) the average units(?: per order)?\?", _FLAGS)),
]
_METRIC_NAMES = {name for name, _ in _METRICS}
_REQUIRED = {
    "sum_units": ("units",),
    "sum_revenue": ("units", "price"),
    "count_orders": (),
    "avg_price": ("price",),
    "avg_units": ("units",),
}
_KEYS = {"file", "metric", "product", "region", "month_from", "month_to"}


def _unparsed(note: str) -> SolveResult:
    return SolveResult("unparsed", note=note)


def _take(rx: re.Pattern, text: str):
    """Remove exactly one occurrence of rx: (match, rest); (None, text) if absent, (False, text) if repeated."""
    found = list(rx.finditer(text))
    if not found:
        return None, text
    if len(found) > 1:
        return False, text
    m = found[0]
    return m, text[:m.start()] + text[m.end():]


def parse(instruction: str) -> dict | SolveResult:
    sentences = [" ".join(s.split()) for s in split_sentences(strip_boilerplate(instruction))]
    if len(sentences) != 2:
        return _unparsed(f"expected file sentence and question, got {len(sentences)} sentences")
    m = _FILE.fullmatch(sentences[0])
    if not m:
        return _unparsed(f"unrecognized file sentence: {sentences[0]!r}")
    file = m.group(1)
    text = sentences[1]

    product = region = month_from = month_to = None
    for name, rx in (("product", _PRODUCT), ("region", _REGION), ("months", _MONTHS), ("month", _MONTH)):
        found, text = _take(rx, text)
        if found is False:
            return _unparsed(f"repeated {name} filter")
        if found is None:
            continue
        if name == "product":
            product = found.group(1)
        elif name == "region":
            region = found.group(1)
        elif name == "months":
            month_from, month_to = int(found.group(1)), int(found.group(2))
        else:
            if month_from is not None:
                return _unparsed("conflicting month filters")
            month_from = month_to = int(found.group(1))
    text = " ".join(text.split())
    metrics = [name for name, rx in _METRICS if rx.fullmatch(text)]
    if len(metrics) != 1:
        return _unparsed(f"unrecognized question: {sentences[1]!r}")
    return {"file": file, "metric": metrics[0], "product": product, "region": region,
            "month_from": month_from, "month_to": month_to}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_spec(spec) -> str | None:
    """Return an error description, or None if the spec is well formed."""
    if not isinstance(spec, dict) or set(spec) != _KEYS:
        return "bad spec keys"
    f = spec["file"]
    if (not isinstance(f, str) or not f or f in (".", "..") or "/" in f or "\\" in f
            or ":" in f or "\0" in f):
        return f"bad file: {f!r}"
    if spec["metric"] not in _METRIC_NAMES:
        return f"bad metric: {spec['metric']!r}"
    for key in ("product", "region"):
        v = spec[key]
        if v is not None and (not isinstance(v, str) or not v):
            return f"bad {key}: {v!r}"
    lo, hi = spec["month_from"], spec["month_to"]
    if (lo is None) != (hi is None):
        return "month_from and month_to must be given together"
    if lo is not None and not (_is_int(lo) and _is_int(hi)):
        return "bad months"
    return None


def _fmt(value: Decimal, check: str | None) -> str:
    if check == "exact_int":
        return str(int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def execute(spec: dict, workspace: Path, check: str | None) -> SolveResult:
    try:
        err = _valid_spec(spec)
        if err:
            return _unparsed(err)
        metric = spec["metric"]
        path = Path(workspace) / spec["file"]
        if not path.is_file():
            return SolveResult("infeasible", note=f"no such file: {spec['file']}")
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = [c.strip() for c in (reader.fieldnames or [])]
            reader.fieldnames = columns
            needed = list(_REQUIRED[metric])
            if spec["product"] is not None:
                needed.append("product")
            if spec["region"] is not None:
                needed.append("region")
            if spec["month_from"] is not None:
                needed.append("month")
            missing = [c for c in needed if c not in columns]
            if missing:
                return SolveResult("infeasible", note=f"missing columns: {missing}")
            product = spec["product"].casefold() if spec["product"] is not None else None
            region = spec["region"].casefold() if spec["region"] is not None else None
            lo, hi = spec["month_from"], spec["month_to"]
            total = Decimal(0)
            count = 0
            for row in reader:
                if product is not None and (row["product"] or "").strip().casefold() != product:
                    continue
                if region is not None and (row["region"] or "").strip().casefold() != region:
                    continue
                if lo is not None:
                    month = int(row["month"].strip())
                    if not lo <= month <= hi:
                        continue
                count += 1
                if metric in ("sum_units", "avg_units"):
                    total += Decimal(row["units"].strip())
                elif metric == "avg_price":
                    total += Decimal(row["price"].strip())
                elif metric == "sum_revenue":
                    total += Decimal(row["units"].strip()) * Decimal(row["price"].strip())
        if metric == "count_orders":
            return SolveResult("solved", str(count))
        if metric in ("avg_price", "avg_units"):
            if count == 0:
                return SolveResult("infeasible", note="average of empty selection")
            total = total / count
        return SolveResult("solved", _fmt(total, check))
    except (ValueError, InvalidOperation, AttributeError, KeyError, TypeError, csv.Error) as e:
        return _unparsed(f"bad data: {e!r}")
    except OSError as e:
        return SolveResult("infeasible", note=f"cannot read file: {e!r}")


def solve(task: Task) -> SolveResult:
    try:
        spec = parse(task.instruction)
        if isinstance(spec, SolveResult):
            return spec
        return execute(spec, task.workspace, task.check)
    except Exception as e:
        return _unparsed(f"unexpected error: {e!r}")
