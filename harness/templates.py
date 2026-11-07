"""Instruction templates: values become placeholders so many tasks share one template."""
import re

from harness.solvers.common import strip_boilerplate
from harness.tasks import Task

_STOP_WORDS = {"A", "An", "The", "In", "How", "What", "Which", "Who", "Where", "When", "Then", "It", "Use",
               "Compute"}
_QUOTED = re.compile(r"""(?<![\w])'([^'\n]+)'(?!\w)|"([^"\n]+)\"""")
_ENTITY = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}\b")
_PLACEHOLDER = re.compile(r"\{[SEN]\d+\}")
# whole integers only: not part of a word, a decimal, or an existing placeholder
_NUMBER = re.compile(r"(?<![\w.{])\d+(?![\w}]|\.\d)")


class _Slots:
    def __init__(self, kind: str):
        self.kind = kind
        self.by_value: dict[str, str] = {}

    def name(self, value: str) -> str:
        if value not in self.by_value:
            self.by_value[value] = f"{self.kind}{len(self.by_value) + 1}"
        return "{" + self.by_value[value] + "}"


def _skip_placeholders(text: str, sub) -> str:
    """Apply `sub` (a regex-sub callable) to text between existing placeholders only."""
    out, pos = [], 0
    for m in _PLACEHOLDER.finditer(text):
        out.append(sub(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(sub(text[pos:]))
    return "".join(out)


def templatize(instruction: str) -> tuple[str, dict[str, str]]:
    text = strip_boilerplate(instruction)
    strings, entities, numbers = _Slots("S"), _Slots("E"), _Slots("N")

    text = _QUOTED.sub(lambda m: strings.name(m.group(1) if m.group(1) is not None else m.group(2)), text)

    def entity_sub(m: re.Match) -> str:
        words = m.group(0).split(" ")
        lead = 0
        while lead < len(words) and words[lead] in _STOP_WORDS:
            lead += 1
        rest = words[lead:]
        if len(rest) < 2:
            return m.group(0)
        return " ".join(words[:lead] + [entities.name(" ".join(rest))]).strip()

    text = _skip_placeholders(text, lambda s: _ENTITY.sub(entity_sub, s))
    text = _skip_placeholders(text, lambda s: _NUMBER.sub(lambda m: numbers.name(m.group(0)), s))
    slots: dict[str, str] = {}
    for table in (strings, entities, numbers):
        for value, name in table.by_value.items():
            slots[name] = value
    return text, slots


def group(tasks: list[Task]) -> dict[tuple[str, str], list[tuple[Task, dict[str, str]]]]:
    groups: dict[tuple[str, str], list[tuple[Task, dict[str, str]]]] = {}
    for task in tasks:
        template, slots = templatize(task.instruction)
        groups.setdefault((task.task_type, template), []).append((task, slots))
    return groups
