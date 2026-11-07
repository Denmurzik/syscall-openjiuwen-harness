from harness.solvers.probe import behaviour, inputs_from_call, visible_call


def test_behaviour_reprs_and_exceptions():
    src = "def f(x):\n    return 10 // x\n"
    assert behaviour(src, "f", [(2,), (0,), (5,)]) == ["5", "<exc:ZeroDivisionError>", "2"]


def test_behaviour_does_not_mutate_inputs_between_calls():
    src = "def f(xs):\n    xs.append(1)\n    return len(xs)\n"
    assert behaviour(src, "f", [([],), ([],)]) == ["1", "1"]


def test_behaviour_survives_print_and_missing_function():
    assert behaviour("def f():\n    print('x')\n    return 1\n", "f", [()]) == ["1"]
    assert behaviour("def f():\n    return 1\n", "g", [(), ()]) == ["<exc:KeyError>"] * 2


def test_behaviour_timeout_fills_rest():
    src = "def f(x):\n    while x:\n        pass\n    return 1\n"
    out = behaviour(src, "f", [(0,), (1,), (0,)], timeout=3)
    assert out[0] == "1" and len(out) == 3 and out[2].startswith("<exc:")


def test_inputs_from_call_shapes():
    ins = inputs_from_call(([3, 1, 2], "ab", 5), n=6)
    assert ins[0] == ([3, 1, 2], "ab", 5) and 2 <= len(ins) <= 6
    assert len({repr(i) for i in ins}) == len(ins)
    for a in ins:
        assert isinstance(a[0], list) and isinstance(a[1], str) and isinstance(a[2], int)
    assert inputs_from_call(([3, 1, 2],), seed=1) == inputs_from_call(([3, 1, 2],), seed=1)


def test_visible_call_finds_literal_call_and_statement():
    t = ("import unittest\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
         "    def test_a(self):\n        self.assertEqual(m.f([3, 1]), [1, 3])\n")
    args, stmt = visible_call(t, "f")
    assert args == ([3, 1],) and "assertEqual" in stmt
    assert visible_call(t, "g") is None
