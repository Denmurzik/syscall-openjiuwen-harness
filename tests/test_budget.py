from harness.budget import TokenLedger


def test_cost_counts_input_at_one_tenth():
    led = TokenLedger(budget=1000, reserve=100)
    assert led.charge(prompt=500, completion=40) == 90
    assert led.remaining() == 910


def test_can_afford_respects_reserve_and_exhaustion():
    led = TokenLedger(budget=1000, reserve=100)
    assert led.can_afford(900) and not led.can_afford(901)
    led.mark_exhausted()
    assert not led.can_afford(1)
