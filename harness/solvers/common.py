import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_SKIP = re.compile(r"(compute it with|use bash|use the file tools)\b", re.IGNORECASE)
_STOP = re.compile(r"when you are done\b", re.IGNORECASE)
_WORK_ONLY = re.compile(r"work only inside\b", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text.strip())) if s]


def strip_boilerplate(instruction: str) -> str:
    kept = []
    for sentence in split_sentences(instruction):
        if _STOP.match(sentence):
            break
        if _SKIP.match(sentence) or _WORK_ONLY.match(sentence):
            continue
        kept.append(sentence)
    return " ".join(kept)


# ============================================================================ phase-1 normalizer
#
# `normalize_instruction` only rewrites the FORM of an instruction and is deliberately conservative: it saves tokens,
# a wrong silent answer costs points. Rules: (1) bullet lines become sentences; (2) a sentence is dropped only if it
# fully matches a short CLOSED list of content-free patterns (no domain word, number, quote or name can be in it);
# (3) a file description is rewritten to the canonical file sentence; (4) a polite/imperative question lead is
# rewritten to a wh-question. Anything else, and any text after the question sentence, leaves the instruction as
# `strip_boilerplate` gives it: the solver then parses it strictly or answers `unparsed`.

_I = re.IGNORECASE
_BULLET = re.compile(r"^(\s*)(?:[-*•‣◦●]|\d{1,2}[.)])\s+")
_INITIAL_END = re.compile(r"\s[A-Z]\.$")
_MAX_LEN = 20000


def _flatten(text: str) -> str | None:
    """Bullet lines become sentences (a missing final stop is added). None if a bullet is nested / indented."""
    parts = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _BULLET.match(line)
        if m:
            if m.group(1):
                return None
            s = line[m.end():].strip()
            if not s:
                continue
            if s[-1] not in ".?!":
                s += "."
        parts.append(s)
    return " ".join(parts)


# Content-free whole sentences (case-insensitive, full match). No digits, quotes, names or domain words.
_GENERIC_FILLER = [re.compile(p, _I) for p in (
    r"(?:please )?(?:review|read|consider|look at) the following(?: task| problem| question| scenario)?[.:]?",
    r"here is (?:a|the) (?:task|problem|question|scenario)[.:]?",
    r"you are given the following(?: task| problem| scenario)?[.:]?",
    r"let['’]s look at the following[.:]?",
)]
_DATA_FILLER = [re.compile(p, _I) for p in (
    r"here is a brief overview of the dataset structure\.",
)]
_SEARCH_FILLER = [re.compile(p, _I) for p in (
    r"(?:the )?(?:directory )?kb/ (?:directory )?(?:contains|holds) a small knowledge base of markdown files\.",
    r"the kb/ directory holds a small knowledge base in markdown format\.",
    r"a small knowledge base of markdown files is (?:available|located|stored) in the kb/ directory\.",
    r"(?:you can find|we have|we maintain) a small knowledge base of markdown files in the kb/ directory\.",
    r"(?:refer to|locate|access) the small knowledge base of markdown files (?:located )?in the kb/ directory\.",
    r"check out the kb/ folder for some markdown files\.",
)]
_FILLER = {"math": _GENERIC_FILLER, "data": _GENERIC_FILLER + _DATA_FILLER, "search": _GENERIC_FILLER + _SEARCH_FILLER}
# Sentence-initial words that carry nothing: "FYI, ...", "Quick question: ...", "Given this context, ...".
_PREFIX = re.compile(
    r"(?:fyi|quick (?:note|question|query)|just a heads-up|heads up|note|here is (?:a|the) (?:task|problem|question|scenario)"
    r"|(?:given|based on|using|considering|with) (?:this|that|the above) (?:context|information|data|resource))\s*[:,]\s+",
    _I)
_PRE_FILE = {"", "looking at", "regarding", "quick query on", "given that", "here is", "please check", "please review",
             "please look at", "reviewing", "checking", "we have", "we need to analyze", "heads up", "fyi", "note",
             "i am looking at a dataset from", "looking at the file"}


def _peel_prefix(sentence: str) -> str:
    while True:
        m = _PREFIX.match(sentence)
        if not m or not sentence[m.end():].strip():
            return sentence
        sentence = sentence[m.end():].strip()


# ---------------------------------------------------------------------------- data: file description
_F = r"(?P<f>[A-Za-z0-9][\w.-]*\.csv)"
_STOPS = (r"(?!(?:could|can|would|please|what|whats|how|which|who|tell|calculate|compute|determine|find|identify|"
          r"provide|give|let|we|i|you|the|a|an|then|so|now|based|given|using|for|in|on|is|are|do|does)\b)")
_C = rf"(?P<c>[A-Za-z_]\w*(?:\s*,\s*(?:and\s+)?{_STOPS}[A-Za-z_]\w*)+)"
_R = r"(?P<r>\d[\d,]*)\s+rows?"
_FP = rf"(?:(?:the|this|that)\s+)?(?:(?:file|dataset|table)\s+)?{_F}(?:\s+(?:file|dataset))?"
_COLW = r"(?:the\s+)?(?:columns?|cols?)"
_FILE_PATTERNS = [re.compile(p, _I) for p in (
    rf"{_FP}\s+(?:has|have)\s+{_COLW}\s+{_C}(?:\s*\(\s*{_R}\s*\))?",
    rf"{_FP}\s*\(\s*{_COLW}\s*:?\s*{_C}\s*[;,]\s*{_R}\s*\)",
    rf"{_FP}\s*\(\s*{_C}\s*;\s*{_R}\s*\)",
    rf"{_FP}\s*\(\s*{_R}\s*,\s*{_COLW}\s*:?\s*{_C}\s*\)",
    rf"{_FP}(?:\s*,\s*which)?\s+(?:contains|includes|has|have|with|containing)\s+{_R}\s+"
    rf"(?:with|and|containing|having|including)\s+{_COLW}\s+{_C}",
    rf"\bfile:\s*{_F}\.?\s+columns?:\s*{_C}\.?\s+rows?(?:\s+count)?:\s*(?P<r>\d[\d,]*)",
)]
_CANON_FILE = re.compile(r"The file (\S+\.csv) has columns ([\w,]+)(?: \(\d[\d,]* rows\))?\.")


def _file_sentence(m: re.Match) -> str:
    cols = [c for c in re.split(r"\s*,\s*(?:and\s+)?", m.group("c").strip()) if c]
    rows = m.groupdict().get("r")
    out = f"The file {m.group('f')} has columns {','.join(cols)}"
    return out + (f" ({rows} rows)." if rows else ".")


def _rewrite_file(text: str) -> str:
    """Replace the one file description in `text` with the canonical file sentence; unchanged if there are
    several, or if something other than a clause boundary follows it."""
    found = []
    for rx in _FILE_PATTERNS:
        for m in rx.finditer(text):
            found.append(m)
    if not found:
        return text
    found.sort(key=lambda m: (m.start(), -m.end()))
    best = found[0]
    for m in found[1:]:
        if m.start() < best.end():
            if _file_sentence(m) != _file_sentence(best) and m.end() > best.end():
                return text
            continue
        return text  # two separate descriptions
    pre, post = text[:best.start()].rstrip(), text[best.end():]
    if post.strip() and not re.match(r"\s*(?:[,;:.]|and\s)", post, _I):
        return text
    post = re.sub(r"^\s*(?:and\s+)?[,;:.]*", "", post, flags=_I).strip()
    parts = []
    if pre:
        head, tail = (pre.rsplit(". ", 1) + [""])[:2] if ". " in pre else ("", pre)
        if head:
            tail_sentences = head + "."
            parts.append(tail_sentences)
        if _peel_prefix(tail.strip() + " \0").rstrip("\0").strip(" :,").lower() not in _PRE_FILE:
            return text
    parts.append(_file_sentence(best))
    if post:
        parts.append(post)
    return " ".join(parts)


# ---------------------------------------------------------------------------- per-domain question forms
_START_A = re.compile(r"(?:initial|starting|beginning)\s+(?:stock|inventory)(?:\s+is|\s*:)\s+(\d[\d,]*)\s+boxes\.?", _I)
_START_B = re.compile(r"(?:the|our|a|this|one)\s+warehouse\s+(?:starts|begins)\s+with\s+(\d[\d,]*)\s+boxes\.?", _I)


def _math_sentence(s: str) -> str:
    for rx in (_START_A, _START_B):
        m = rx.fullmatch(s)
        if m:
            return f"A warehouse starts with {m.group(1)} boxes."
    return s


_POLITE = r"(?:(?:could|can|would) you(?: please)?|please)"
_REQ_VERB = r"(?:calculate|compute|determine|find|identify|locate|tell me|tell us|let me know|provide|give me)"
_DATA_LEAD = re.compile(rf"(?:{_POLITE}(?:\s+{_REQ_VERB})?|{_REQ_VERB})\s+(?P<r>.+?)\s*[?.!]?", _I)
_WH = re.compile(r"(?:what|how|which|in which)\b", _I)


def _data_question(s: str) -> str:
    s = re.sub(r"\bwhat['’]s\b", "what is", s, flags=_I)
    s = re.sub(r"\bmean\b", "average", s, flags=_I)
    s = re.sub(r"\bunits\s*(?:\*|x|×|times)\s*price\b", "units multiplied by price", s, flags=_I)
    m = _DATA_LEAD.fullmatch(s)
    if m:
        r = m.group("r")
        if _WH.match(r):
            return r[0].upper() + r[1:] + "?"
        if re.match(r"the\b", r, _I):
            return "What is " + r + "?"
    return s


_SEARCH_LEAD = re.compile(rf"(?:{_POLITE}\s+{_REQ_VERB}|{_REQ_VERB}|{_POLITE})\s+(?P<r>.+?)\s*[?.!]?", _I)
_SEARCH_NP = [
    (re.compile(r"the mentor of (.+)", _I), "Who is the mentor of {}?"),
    (re.compile(r"the employer of (.+)", _I), "Who is the employer of {}?"),
    (re.compile(r"the (?:company|firm) that employs (.+)", _I), "Which company employs {}?"),
    (re.compile(r"(?:the )?(?:birth year|year of birth) of (.+)", _I), "What year was {} born?"),
    (re.compile(r"the (?:founding year|year of founding) (?:of|for) (.+)", _I), "What year was {} founded?"),
    (re.compile(r"the person who is the mentor of (.+)", _I), "Who is the mentor of {}?"),
    (re.compile(r"the company where (.+) is employed", _I), "Which company employs {}?"),
    (re.compile(r"the city where (the headquarters of .+) is located", _I), "In which city is {}?"),
    (re.compile(r"the city where (.+) is headquartered", _I), "In which city is {} headquartered?"),
    (re.compile(r"the city where (.+) has its headquarters", _I), "In which city does {} have its headquarters?"),
]


def _search_question(s: str) -> str:
    m = _SEARCH_LEAD.fullmatch(s)
    if not m:
        return s
    r = m.group("r")
    if _WH.match(r) or re.match(r"(?:who|where|when)\b", r, _I):
        return r[0].upper() + r[1:] + "?"
    for rx, template in _SEARCH_NP:
        n = rx.fullmatch(r)
        if n:
            return template.format(n.group(1))
    return s


# ---------------------------------------------------------------------------- entry point
def _merge_initials(sentences: list[str]) -> list[str]:
    out: list[str] = []
    for s in sentences:  # "Jonas T. Dubois": a single-letter initial does not end a sentence
        if out and _INITIAL_END.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def normalize_instruction(instruction: str, domain: str) -> str:
    """Instruction without the boilerplate tail, in canonical wording where the rewrite is safe.
    `domain` is "math", "data" or "search". Never raises; on doubt returns `strip_boilerplate(instruction)`."""
    try:
        if len(instruction) > _MAX_LEN:
            return strip_boilerplate(instruction)
        out = _normalize(instruction, domain)
        return strip_boilerplate(instruction) if out is None else out
    except Exception:
        return strip_boilerplate(instruction)


def _normalize(instruction: str, domain: str) -> str | None:
    flat = _flatten(instruction)
    if flat is None:
        return None
    text = strip_boilerplate(flat)
    if domain == "data":
        text = _rewrite_file(text)
    sents = [" ".join(s.split()) for s in _merge_initials(split_sentences(text))]
    sents = [_peel_prefix(s) for s in sents if s]
    if not sents:
        return None
    protected = {i for i, s in enumerate(sents) if domain == "data" and _CANON_FILE.fullmatch(s)}
    others = [i for i in range(len(sents)) if i not in protected]
    if not others:
        return None
    asking = [i for i in others if sents[i].endswith("?")]
    q = asking[-1] if asking else others[-1]
    if q != len(sents) - 1:
        return None  # something follows the question: never guess
    out = []
    for i, s in enumerate(sents):
        if i != q and i not in protected and not s.endswith("?") and any(rx.fullmatch(s) for rx in _FILLER[domain]):
            continue
        if i == q:
            if domain == "data":
                s = _data_question(s)
            elif domain == "search":
                s = _search_question(s)
        elif domain == "math":
            s = _math_sentence(s)
        out.append(s)
    return " ".join(out)
