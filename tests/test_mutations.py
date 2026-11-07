from harness.solvers.mutations import mutants


def _all(src):
    return list(mutants(src))


def test_mutants_include_inverse_of_known_bugs():
    src = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
    assert any("s += x" in m for m in mutants(src))


def test_compare_and_constant_and_call_mutants():
    ms = _all("def f(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n")
    assert any("== 0" in m for m in ms)
    ms = _all("def g(a, b):\n    if a < b:\n        return sorted(a)\n    return b\n")
    assert any("a > b" in m for m in ms)
    assert any("reversed(a)" in m for m in ms)
    assert any("not a < b" in m for m in ms)
    assert any("reverse=True" in m for m in ms)


def test_slice_bool_and_minmax_mutants():
    ms = _all("def f(xs, n):\n    return xs[:n - 1]\n")
    assert any("xs[:n - 2]" in m for m in ms) and any("xs[:n - 0]" in m for m in ms)
    ms = _all("def f(xs):\n    return xs[::-1]\n")
    assert any(m.strip().endswith("return xs") for m in ms)
    assert any("max(" in m for m in _all("def f(a, b):\n    return min(a, b)\n"))
    assert any("False" in m for m in _all("def f():\n    return True\n"))
    assert any(" or " in m for m in _all("def f(a, b):\n    return a and b\n"))


def test_operators_come_first_and_no_duplicates_or_identity():
    src = "def f(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
    ms = _all(src)
    assert ms[0].count("+=") == 1
    assert len(ms) == len(set(ms))
    import ast
    assert ast.unparse(ast.parse(src)) not in ms


def test_indented_and_broken_source():
    assert _all("    def f(a):\n        return a + 1\n")
    assert _all("def f(:\n") == []
