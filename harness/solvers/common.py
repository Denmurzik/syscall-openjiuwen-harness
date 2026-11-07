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
