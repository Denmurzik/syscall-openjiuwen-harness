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
# `normalize_instruction` only rewrites the FORM of an instruction (bullets, filler preambles, file-description
# variants, polite question leads) into the canonical wording the solvers parse. It never decides anything:
# a sentence is dropped only when it is made of filler words alone and every number, quoted literal, file name and
# capitalised name in it also stands in the question or the file sentence (so it cannot carry a filter the solver
# would miss). Everything else stays as written and the solver answers `unparsed`, as before.

_I = re.IGNORECASE
_BULLET = re.compile(r"^\s*(?:[-*•‣◦●]|\d{1,2}[.)])\s+")
_INITIAL_END = re.compile(r"\s[A-Z]\.$")


def _flatten(text: str) -> str:
    """Bullet lines become sentences (a missing final stop is added); other lines are joined as they are."""
    parts = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _BULLET.match(line)
        if m:
            s = line[m.end():].strip()
            if not s:
                continue
            if s[-1] not in ".?!":
                s += "."
        parts.append(s)
    text = " ".join(parts)
    # "(Note: The directory kb/ holds ... files.)": a parenthesised whole sentence is just a sentence
    return re.sub(r"\(\s*([A-Z][^()]*[.?!])\s*\)", r"\1", text)


# Words that carry no task content. PREFIX words may open a sentence before a "," or ":" (no question words, no
# measures); the sentence-level sets add words used by redundant restatements ("Calculate the total volume ...").
_BASE = set("""
a an the this that these those it its their his her our your my we you i us me they them he she
is are was were be being been am do does did has have having can could would will should may might shall must
of in on at for with from by as to into within about regarding concerning during and or through
please quick note notes fyi heads up headsup just here there now then also some small following below above
before proceeding proceed continue given based context information info resource using use considering consider
imagine suppose assume let's lets let look looking take check out review reviewing see read open access consult
refer examine study analyze analyzing analyse inspect task involves involving overview brief
dataset structure data file files folder directory kb/ markdown markdown-based knowledge base collection set stored
located contains contain contained comprising comprises consists consisting holds hold available provided
maintain maintains maintained format question query request problem puzzle scenario situation update status story
inventory stock level levels warehouse warehouses logistics center centre distribution busy managing manages
tracking track changes specifically heads-up finally next lastly
""".split())
_DATA = set("""
orders order units unit price revenue region product month months sales sale year quarter total number count
value mean average sum result answer volume period specified specific entries entry rows row columns column
calculate compute determine identify locate focus focusing focused compile get inclusive between
which who where that need needed want wanted know max maximum min minimum largest smallest highest lowest
single per
""".split())
_SEARCH = set("""
employer employers mentor mentors mentorship chain company companies person people individual headquarters city
founding founded year born birth headquartered employs employed find identify determine locate trace
second third final first which who where that when need needed want know
""".split())
_MATH = set()
_VOCAB = {"math": _BASE | _MATH, "data": _BASE | _DATA, "search": _BASE | _SEARCH}

_QUOTED = re.compile(r"(?<![\w])'([^'\n]{1,40})'(?![\w])")
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./’'-]*")
_MAX_FLUFF_WORDS = 40


def _clean_token(tok: str) -> str:
    tok = tok.rstrip(".'’-")
    for suffix in ("'s", "’s"):
        if tok.endswith(suffix):
            tok = tok[:-2]
    return tok


def _contained(token: str, ref: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(token.casefold())}(?![\w])", ref.casefold()) is not None


def _is_fluff(text: str, ref: str, vocab: set[str]) -> bool:
    """True if `text` holds only filler words, plus numbers / quoted literals / file names / names that `ref` repeats."""
    quoted = _QUOTED.findall(text)
    if any(f"'{q.casefold()}'" not in ref.casefold() for q in quoted):
        return False
    rest = _QUOTED.sub(" ", text)
    tokens = [_clean_token(t) for t in _TOKEN.findall(rest)]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > _MAX_FLUFF_WORDS:
        return False
    for tok in tokens:
        low = tok.casefold()
        if low in vocab:
            continue
        if any(c.isdigit() for c in tok) or low.endswith(".csv") or "/" in tok or tok[0].isupper():
            if _contained(tok, ref):
                continue
        return False
    return True


def _peel_prefix(sentence: str, ref: str) -> str:
    """Drop leading filler clauses ("Quick note on inventory: ...", "Given this context, ...")."""
    while True:
        for i, ch in enumerate(sentence):
            if ch not in ",:":
                continue
            prefix, rest = sentence[:i].strip(), sentence[i + 1:].strip()
            if prefix and rest and _is_fluff(prefix, rest + " " + ref, _BASE):
                sentence = rest
                break
        else:
            return sentence


def _unwrap(sentence: str) -> str:
    if sentence.startswith("(") and sentence.endswith(")") and sentence.count("(") == sentence.count(")") == 1:
        return sentence[1:-1].strip()
    return sentence


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
        parts.append(pre if pre[-1] in ".?!" else pre + ".")
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
    `domain` is "math", "data" or "search"."""
    try:
        return _normalize(instruction, domain)
    except Exception:
        return strip_boilerplate(instruction)


def _normalize(instruction: str, domain: str) -> str:
    vocab = _VOCAB[domain]
    text = strip_boilerplate(_flatten(instruction))
    if domain == "data":
        text = _rewrite_file(text)
    sents = [_unwrap(" ".join(s.split())) for s in _merge_initials(split_sentences(text))]
    sents = [s for s in sents if s]
    if not sents:
        return ""
    protected = {i for i, s in enumerate(sents) if domain == "data" and _CANON_FILE.fullmatch(s)}
    file_text = " ".join(sents[i] for i in protected)
    asking = [i for i, s in enumerate(sents) if s.endswith("?") and i not in protected]
    others = [i for i in range(len(sents)) if i not in protected]
    if not others:
        return " ".join(sents)
    q = asking[-1] if asking else others[-1]
    sents[q] = _peel_prefix(sents[q], file_text)
    for i in range(len(sents)):
        if i != q and i not in protected:
            sents[i] = _peel_prefix(sents[i], sents[q] + " " + file_text)
    ref = sents[q] + " " + file_text
    keep = []
    for i, s in enumerate(sents):
        if i != q and i not in protected and not s.endswith("?") and _is_fluff(s, ref, vocab):
            continue
        keep.append(i)
    out = []
    for i in keep:
        s = sents[i]
        if i == q:
            if domain == "data":
                s = _data_question(s)
            elif domain == "search":
                s = _search_question(s)
        elif domain == "math":
            s = _math_sentence(s)
        out.append(s)
    return " ".join(out)
