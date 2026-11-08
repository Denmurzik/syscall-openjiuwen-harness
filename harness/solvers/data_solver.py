import csv
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from harness.result import SolveResult
from harness.solvers.common import split_sentences, strip_boilerplate
from harness.tasks import Task

_FLAGS = re.IGNORECASE
_FILE = re.compile(r"The file (\S+\.csv) has columns ([\w,]+)(?: \(\d[\d,]* rows\))?\.", _FLAGS)

# Filter fragments, removed from the question text before the metric is matched.
_PRODUCT = re.compile(r"\s+(?:of|for) product '(\w+)'", _FLAGS)
# (pattern, value is quoted); an unquoted region must be confirmed against the CSV.
_REGIONS = [
    (re.compile(r"\s+in region '(\w+)'", _FLAGS), True),
    (re.compile(r"\s+in the '(\w+)' region", _FLAGS), True),
    (re.compile(r"\s+for region '(\w+)'", _FLAGS), True),
    (re.compile(r"\s+for the '(\w+)' region", _FLAGS), True),
    (re.compile(r"\s+in region (\w+)", _FLAGS), False),
    (re.compile(r"\s+for region (\w+)", _FLAGS), False),
    (re.compile(r"\s+in the (\w+) region", _FLAGS), False),
]
_RANGES = [
    re.compile(r"\s+in months? (\d+) through (\d+) inclusive", _FLAGS),
    re.compile(r"\s+in months? (\d+) to (\d+)(?: inclusive)?", _FLAGS),
    re.compile(r"\s+between months? (\d+) and (\d+)(?: inclusive)?", _FLAGS),
    re.compile(r"\s+from months? (\d+) to (?:month )?(\d+)(?: inclusive)?", _FLAGS),
]
_MONTH = re.compile(r"\s+in month (\d+)(?!\d)", _FLAGS)

_METRICS = [
    ("sum_units", re.compile(
        r"(?:how many units were (?:sold|ordered)|what (?:is|was) the total (?:number of )?units(?: sold)?)[?.!]?", _FLAGS)),
    ("sum_revenue", re.compile(
        r"what (?:is|was) (?:the )?(?:total )?revenue(?: \(units multiplied by price\))?[?.!]?", _FLAGS)),
    ("count_orders", re.compile(r"how many orders were (?:placed|made|recorded)[?.!]?", _FLAGS)),
    ("avg_price", re.compile(r"what (?:is|was) the average price[?.!]?", _FLAGS)),
    ("avg_units", re.compile(
        r"what (?:is|was) the average (?:number of )?units(?: per order| in an order)?[?.!]?", _FLAGS)),
    ("max_units", re.compile(
        r"what (?:is|was) the (?:largest|biggest|highest|maximum) (?:number of )?units?"
        r"(?: in| for| ordered in)? (?:a|an|one|any) (?:single )?order[?.!]?", _FLAGS)),
    ("min_units", re.compile(
        r"what (?:is|was) the (?:smallest|lowest|minimum) (?:number of )?units?"
        r"(?: in| for| ordered in)? (?:a|an|one|any) (?:single )?order[?.!]?", _FLAGS)),
]
# ", rounded to N decimal places" (only for averages and revenue); N=2 is the default.
_ROUND = re.compile(
    r",?\s+rounded to (?:(\d+) decimal places?|the nearest (?:integer|whole number))(?=\s*(?:\?|$))", _FLAGS)
# "What is the total discount given?": a quantity no sales column can provide.
_ABSENT = re.compile(
    r"what (?:is|was) the total (discount|tax|shipping|profit|margin|commission|rebate|refund|tip)s?"
    r"(?: given| charged| paid| applied| collected| amount)?\?", _FLAGS)
_ROUNDABLE = {"sum_revenue", "avg_price", "avg_units"}
_MAX_DECIMALS = 6
_TOP_COLUMNS = ("product", "region")
_TOP_BASES = ("sum_units", "sum_revenue", "count_orders")
_TOP_NAMES = {f"top_{c}_{b}" for c in _TOP_COLUMNS for b in _TOP_BASES}
_TOP_MEASURE = {"units": "sum_units", "revenue": "sum_revenue", "orders": "count_orders",
                "number of orders": "count_orders"}
_TOP_Q = re.compile(
    r"which (product|region) (?:sold the most (units)|(?:had|has|generated|made|recorded) the "
    r"(?:most|highest|greatest) (?:total )?(units|revenue|orders|number of orders)(?: sold)?)(?: overall)?[?.!]?",
    _FLAGS)
_METRIC_NAMES = {name for name, _ in _METRICS} | _TOP_NAMES
_REQUIRED = {
    "sum_units": ("units",),
    "sum_revenue": ("units", "price"),
    "count_orders": (),
    "avg_price": ("price",),
    "avg_units": ("units",),
    "max_units": ("units",),
    "min_units": ("units",),
}
_KEYS = {"file", "metric", "product", "region", "month_from", "month_to"}
_OPTIONAL_KEYS = {"decimals"}


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


def _column_values(path: Path, column: str) -> set[str] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            names = {(c or "").strip().casefold(): c for c in (reader.fieldnames or [])}
            if column not in names:
                return None
            return {(row[names[column]] or "").strip().casefold() for row in reader}
    except (OSError, ValueError, csv.Error):
        return None


def parse(instruction: str, workspace: Path | None = None) -> dict | SolveResult:
    sentences = [" ".join(s.split()) for s in split_sentences(strip_boilerplate(instruction))]
    if len(sentences) != 2:
        return _unparsed(f"expected file sentence and question, got {len(sentences)} sentences")
    m = _FILE.fullmatch(sentences[0])
    if not m:
        return _unparsed(f"unrecognized file sentence: {sentences[0]!r}")
    file = m.group(1)
    text = sentences[1]

    product = region = month_from = month_to = None
    found, text = _take(_PRODUCT, text)
    if found is False:
        return _unparsed("repeated product filter")
    if found:
        product = found.group(1)

    regions = []
    for rx, quoted in _REGIONS:
        found, text = _take(rx, text)
        if found is False:
            return _unparsed("repeated region filter")
        if found:
            regions.append((found.group(1), quoted))
    if len(regions) > 1:
        return _unparsed("repeated region filter")
    if regions:
        region, quoted = regions[0]
        if not quoted:
            values = _column_values(Path(workspace) / file, "region") if workspace is not None else None
            if values is None or region.casefold() not in values:
                return _unparsed(f"unquoted region {region!r} not confirmed by the data")

    ranges = []
    for rx in _RANGES:
        found, text = _take(rx, text)
        if found is False:
            return _unparsed("repeated month filter")
        if found:
            ranges.append((int(found.group(1)), int(found.group(2))))
    found, text = _take(_MONTH, text)
    if found is False:
        return _unparsed("repeated month filter")
    if found:
        ranges.append((int(found.group(1)),) * 2)
    if len(ranges) > 1:
        return _unparsed("conflicting month filters")
    if ranges:
        month_from, month_to = ranges[0]

    found, text = _take(_ROUND, text)
    if found is False:
        return _unparsed("repeated rounding clause")
    decimals = int(found.group(1) or 0) if found else None
    text = " ".join(text.split())
    metrics = [name for name, rx in _METRICS if rx.fullmatch(text)]
    top = _TOP_Q.fullmatch(text)
    if top is not None:
        metrics.append(f"top_{top.group(1).lower()}_{_TOP_MEASURE[(top.group(2) or top.group(3)).lower()]}")
    if len(metrics) != 1:
        absent = _ABSENT.fullmatch(text)
        if absent is not None and workspace is not None:
            cols = _csv_header_and_values(Path(workspace) / file)
            if cols is not None and absent.group(1).casefold() not in cols[0]:
                return SolveResult("infeasible", note=f"the file has no {absent.group(1).casefold()} column")
        return _unparsed(f"unrecognized question: {sentences[1]!r}")
    return _with_decimals({"file": file, "metric": metrics[0], "product": product, "region": region,
                           "month_from": month_from, "month_to": month_to}, decimals)


def _with_decimals(spec: dict, decimals: int | None) -> dict | SolveResult:
    if decimals is None:
        return spec
    if spec["metric"] not in _ROUNDABLE or decimals > _MAX_DECIMALS:
        return _unparsed(f"unsupported rounding to {decimals} places for {spec['metric']}")
    return {**spec, "decimals": decimals}


def _split_metric(metric: str) -> tuple[str | None, str]:
    """(group column or None, base metric)."""
    if metric in _TOP_NAMES:
        _, col, base = metric.split("_", 2)
        return col, base
    return None, metric

# ------------------------------------------------------------------ loose parser

_LOOSE_STOP = {
    "what", "is", "was", "are", "were", "the", "a", "an", "of", "in", "for", "total", "number", "how", "many",
    "sold", "placed", "made", "recorded", "overall", "all", "both", "included", "inclusive", "between", "from",
    "to", "through", "and", "s", "which", "did", "there", "had", "has", "have", "been", "we", "by", "that",
}
_LOOSE_MONTHS = re.compile(
    r"(?:\b(between|from|through|thru|until|till|up to|by)\s+)?\bmonths?\s+(\d+)"
    r"(?:\s*(?:through|to|-)\s*(?:months?\s+)?(\d+)|\s+and\s+(?:months?\s+)?(\d+))?(?!\d)"
    r"(?:\s+(onwards?)\b)?", _FLAGS)
_MONTH_NAMES = {n: i for i, n in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_MONTH_NAMES["sept"] = 9
_NAMED_MONTH = re.compile(r"\bin\s+(" + "|".join(_MONTH_NAMES) + r")\b", _FLAGS)
_ORDINAL_MONTH = re.compile(r"\bin\s+the\s+(\d+)(?:st|nd|rd|th)\s+month\b", _FLAGS)
_UPTO = {"through", "thru", "until", "till", "up to"}
# "units multiplied by price" / "price times units": the definition of revenue, not a second measure.
_REVENUE_DEF = re.compile(
    r"\(?\b(?:units?\s+(?:multiplied by|times)\s+(?:the\s+)?price"
    r"|price\s+(?:multiplied by|times)\s+(?:the\s+)?units?)\b\)?", _FLAGS)
_WORD = re.compile(r"[a-z0-9]+")
_MAX_WORDS = {"largest", "biggest", "maximum", "max", "highest"}
_MIN_WORDS = {"smallest", "lowest", "minimum", "min"}


def _csv_header_and_values(path: Path) -> tuple[list[str], dict[str, set[str]]] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = [(c or "").strip().casefold() for c in (reader.fieldnames or [])]
            reader.fieldnames = cols
            vals = {c: set() for c in _TOP_COLUMNS if c in cols}
            for row in reader:
                for c in vals:
                    vals[c].add((row[c] or "").strip().casefold())
            return cols, vals
    except (OSError, ValueError, csv.Error):
        return None


def parse_loose(instruction: str, workspace: Path | None) -> dict | SolveResult:
    """Fallback after the strict `parse`: closed keyword vocabulary plus product/region values that really
    occur in the CSV. Any word outside the vocabulary, or any ambiguity, gives `unparsed`."""
    if workspace is None:
        return _unparsed("loose parse needs a workspace")
    sentences = [" ".join(s.split()) for s in split_sentences(strip_boilerplate(instruction))]
    if len(sentences) != 2:
        return _unparsed("loose: expected file sentence and question")
    m = _FILE.fullmatch(sentences[0])
    if not m:
        return _unparsed("loose: unrecognized file sentence")
    file = m.group(1)
    data = _csv_header_and_values(Path(workspace) / file)
    if data is None:
        return _unparsed("loose: cannot read the file")
    _, values = data
    text = sentences[1].casefold()
    found, text = _take(_ROUND, text)
    if found is False:
        return _unparsed("loose: repeated rounding clause")
    decimals = int(found.group(1) or 0) if found else None

    text = _REVENUE_DEF.sub(" ", text)
    named = list(_NAMED_MONTH.finditer(text))
    ordinal = list(_ORDINAL_MONTH.finditer(text))
    ranges = list(_LOOSE_MONTHS.finditer(text))
    if len(ranges) + len(named) + len(ordinal) > 1:
        return _unparsed("loose: several month expressions")
    month_from = month_to = None
    if named or ordinal:
        mm = (named or ordinal)[0]
        if named:
            name = mm.group(1).casefold()
            if any(name in vals for vals in values.values()):
                return _unparsed("loose: month name is also a product/region")
            month_from = month_to = _MONTH_NAMES[name]
        else:
            month_from = month_to = int(mm.group(1))
        text = text[:mm.start()] + " " + text[mm.end():]
    elif ranges:
        mm = ranges[0]
        lead, first, second, third, onward = mm.groups()
        if third is not None and lead not in ("between", "from"):
            return _unparsed("loose: 'and' joins months only after between/from")
        month_from = int(first)
        if second is not None or third is not None:
            if onward or lead in _UPTO or lead == "by":
                return _unparsed("loose: month bounds mixed with an open bound")
            month_to = int(second or third)
        elif onward:
            if lead not in (None, "from"):
                return _unparsed("loose: open month bound")
            month_to = 12
        elif lead in _UPTO:
            month_from, month_to = 1, int(first)
        elif lead is None:
            month_to = month_from
        else:
            return _unparsed("loose: month bound on one side only")
        text = text[:mm.start()] + " " + text[mm.end():]

    if month_from is not None and not (1 <= month_from <= 12 and 1 <= month_to <= 12):
        return _unparsed("loose: month outside 1..12")

    words = _WORD.findall(text)
    found = {c: set() for c in _TOP_COLUMNS}
    rest = []
    for w in words:
        owners = [c for c in _TOP_COLUMNS if w in values.get(c, ())]
        if len(owners) > 1:
            return _unparsed(f"loose: {w!r} is both a product and a region")
        if owners:
            found[owners[0]].add(w)
        else:
            rest.append(w)
    if any(len(v) > 1 for v in found.values()):
        return _unparsed("loose: several values for one column")
    product = next(iter(found["product"]), None)
    region = next(iter(found["region"]), None)

    kw = set(rest)
    topic = {"product", "region"} & kw
    kw -= {"product", "region"}
    top_col = None
    if "which" in kw:
        if len(topic) != 1:
            return _unparsed("loose: cannot tell what to rank")
        top_col = next(iter(topic))
        if (product if top_col == "product" else region) is not None:
            return _unparsed("loose: ranked column is also filtered")
    elif (("product" in topic and product is None) or ("region" in topic and region is None)):
        return _unparsed("loose: column word without a value")
    if month_from is not None and kw & {"from", "through", "to", "between"}:
        return _unparsed("loose: month bound words left over")
    kw -= _LOOSE_STOP
    kw -= {"most", "highest", "greatest", "generated"} if top_col else set()

    has_orders = bool(kw & {"order", "orders"})
    kw -= {"order", "orders"}
    extreme = None
    if kw & _MAX_WORDS:
        extreme = "max_units"
    if kw & _MIN_WORDS:
        if extreme:
            return _unparsed("loose: both largest and smallest")
        extreme = "min_units"
    kw -= _MAX_WORDS | _MIN_WORDS
    single = bool(kw & {"single", "one"})
    kw -= {"single", "one", "per"} if extreme else set()
    has_revenue = "revenue" in kw
    has_avg = bool(kw & {"average", "avg", "mean"})
    if has_revenue:
        if kw & {"units", "unit", "price"}:
            return _unparsed("loose: revenue together with units or price as a second measure")
        kw -= {"revenue"}
    if has_avg:
        kw -= {"average", "avg", "mean"}
    has_units = bool(kw & {"units", "unit"})
    has_price = "price" in kw
    kw -= {"units", "unit", "price"}
    if kw:
        return _unparsed(f"loose: unknown words {sorted(kw)!r}")

    if extreme is not None:
        if (top_col is not None or has_avg or has_revenue or has_price or not has_units
                or not has_orders or not single):
            return _unparsed("loose: unclear largest/smallest order")
        metric = extreme
    elif top_col is not None:
        measure = [x for x, ok in (("sum_units", has_units), ("sum_revenue", has_revenue),
                                   ("count_orders", has_orders)) if ok]
        if len(measure) != 1 or has_avg or has_price:
            return _unparsed("loose: unclear ranking measure")
        metric = f"top_{top_col}_{measure[0]}"
    elif has_avg:
        if has_revenue or has_orders or has_units == has_price:
            return _unparsed("loose: unclear average")
        metric = "avg_price" if has_price else "avg_units"
    elif has_revenue:
        if has_orders:
            return _unparsed("loose: revenue and orders")
        metric = "sum_revenue"
    elif has_orders:
        if has_units or has_price:
            return _unparsed("loose: orders and units")
        metric = "count_orders"
    elif has_units and not has_price:
        metric = "sum_units"
    else:
        return _unparsed("loose: no metric")
    return _with_decimals({"file": file, "metric": metric, "product": product, "region": region,
                           "month_from": month_from, "month_to": month_to}, decimals)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_spec(spec) -> str | None:
    """Return an error description, or None if the spec is well formed."""
    if not isinstance(spec, dict) or not _KEYS <= set(spec) <= _KEYS | _OPTIONAL_KEYS:
        return "bad spec keys"
    if "decimals" in spec and spec["decimals"] is None:
        spec = {k: v for k, v in spec.items() if k != "decimals"}
    d = spec.get("decimals", 2)
    if not _is_int(d) or not 0 <= d <= _MAX_DECIMALS:
        return f"bad decimals: {d!r}"
    if "decimals" in spec and spec["metric"] not in _ROUNDABLE:
        return "decimals not applicable to this metric"
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


def _fmt(value: Decimal, check: str | None, decimals: int | None = None) -> str:
    """`decimals=None`: two places with trailing zeros stripped; an explicit "rounded to N decimal places"
    in the question gives exactly N places ("56.00")."""
    if check == "exact_int":
        return str(int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    value = value.quantize(Decimal(1).scaleb(-(2 if decimals is None else decimals)), rounding=ROUND_HALF_UP)
    text = format(value, "f")
    if decimals is None and "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("-") and not text.strip("-0."):
        text = text[1:]
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
            columns = [c.strip().casefold() for c in (reader.fieldnames or [])]
            reader.fieldnames = columns
            top_col, base = _split_metric(metric)
            needed = list(_REQUIRED[base])
            if top_col is not None:
                needed.append(top_col)
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
            best: Decimal | None = None
            groups: dict[str, list] = {}
            seen_products: set[str] = set()
            seen_regions: set[str] = set()
            for row in reader:
                if product is not None:
                    value = (row["product"] or "").strip().casefold()
                    seen_products.add(value)
                    if value != product:
                        continue
                if region is not None:
                    value = (row["region"] or "").strip().casefold()
                    seen_regions.add(value)
                    if value != region:
                        continue
                if lo is not None:
                    month = int(row["month"].strip())
                    if not lo <= month <= hi:
                        continue
                count += 1
                if top_col is not None:
                    raw = (row[top_col] or "").strip()
                    if not raw:
                        continue
                    share = (Decimal(1) if base == "count_orders" else Decimal(row["units"].strip())
                             if base == "sum_units" else
                             Decimal(row["units"].strip()) * Decimal(row["price"].strip()))
                    entry = groups.setdefault(raw.casefold(), [raw, Decimal(0)])
                    entry[1] += share
                    continue
                if metric in ("max_units", "min_units"):
                    units = Decimal(row["units"].strip())
                    if best is None or (units > best if metric == "max_units" else units < best):
                        best = units
                elif metric in ("sum_units", "avg_units"):
                    total += Decimal(row["units"].strip())
                elif metric == "avg_price":
                    total += Decimal(row["price"].strip())
                elif metric == "sum_revenue":
                    total += Decimal(row["units"].strip()) * Decimal(row["price"].strip())
        if top_col is not None:
            if not groups:
                return _unparsed("no rows to rank")
            ranked = sorted(groups.values(), key=lambda e: e[1], reverse=True)
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                return SolveResult("unparsed", note="tie for the top value",
                                   extra={"best_guess": ranked[0][0]})  # first in file order
            return SolveResult("solved", ranked[0][0])
        if metric == "count_orders":
            return SolveResult("solved", str(count))
        if metric in ("max_units", "min_units"):
            if best is None:
                if ((product is not None and product not in seen_products)
                        or (region is not None and region not in seen_regions)):
                    return SolveResult("infeasible", note="the product/region does not occur in the file")
                return _unparsed("no rows for the extreme value")
            return SolveResult("solved", _fmt(best, check))
        if metric in ("avg_price", "avg_units"):
            if count == 0:
                return SolveResult("solved", "0", note="average of empty selection")
            total = total / count
        return SolveResult("solved", _fmt(total, check, spec.get("decimals")))
    except UnicodeDecodeError as e:
        return SolveResult("infeasible", note=f"cannot decode file: {e!r}")
    except (ValueError, InvalidOperation, AttributeError, KeyError, TypeError, csv.Error) as e:
        return _unparsed(f"bad data: {e!r}")
    except OSError as e:
        return SolveResult("infeasible", note=f"cannot read file: {e!r}")


def solve(task: Task) -> SolveResult:
    try:
        spec = parse(task.instruction, task.workspace)
        if isinstance(spec, SolveResult):
            loose = parse_loose(task.instruction, task.workspace)
            if isinstance(loose, SolveResult):
                return spec
            result = execute(loose, task.workspace, task.check)
            if result.status == "solved":
                return result
            return result if result.extra and not spec.extra else spec
        return execute(spec, task.workspace, task.check)
    except Exception as e:
        return _unparsed(f"unexpected error: {e!r}")
