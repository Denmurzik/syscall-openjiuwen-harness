"""Round 11, task 64: a quantity named without a number makes a math task infeasible."""
from harness.solvers import math_solver
from harness.tasks import Task

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
W = "A warehouse starts with "
END = " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)." + TAIL


def _math(text):
    return Task("m", "math", text, 3.0, "exact_int", None, None)


def test_quantity_without_a_number_is_infeasible():
    for body in ("1669 boxes. It ships 36 boxes per day for several days. It receives 2 deliveries of 87 boxes each.",
                 "1669 boxes. It ships 36 boxes per day for 3 days. It receives some deliveries of 87 boxes each.",
                 "N boxes. It ships 36 boxes per day for 3 days.",
                 "1669 boxes. It ships an unknown number of boxes per day for 3 days.",
                 "1669 boxes. Then the stock is split equally into several warehouses and it keeps one share (rounded down)."):
        res = math_solver.solve(_math(W + body + END))
        assert res.status == "infeasible" and res.note.startswith("quantity not given"), body


def test_readable_and_unfamiliar_texts_are_untouched():
    ok = W + "1669 boxes. It ships 36 boxes per day for 3 days. It receives 2 deliveries of 87 boxes each." + END
    res = math_solver.solve(_math(ok))
    assert (res.status, res.answer) == ("solved", "1735")
    # an unfamiliar sentence stays with the model even when a vague word stands in it
    odd = W + "1669 boxes. It ships some boxes on Monday and 36 boxes per day after that for 3 days." + END
    assert math_solver.solve(_math(odd)).status == "unparsed"


def test_vague_word_in_a_parsed_sentence_does_not_trigger():
    # "some" is an ordinary word of a sentence that parses as written; another sentence is unknown: not judged
    assert math_solver.missing_quantity(
        W + "1669 boxes. It ships 36 boxes per day for 3 days. Then something strange happens." + END) is None
    # nothing vague at all
    assert math_solver.missing_quantity(
        W + "1669 boxes. It ships 36 boxes per day for 3 days." + END) is None
