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


def test_hold_full_amount_when_it_fits():
    led = TokenLedger(budget=1000, reserve=100)
    assert led.hold(50, 400) == 400
    assert led.held == 450
    led.release(450)
    assert led.held == 0


def test_hold_counts_earlier_holds_and_waits_while_they_are_in_flight():
    led = TokenLedger(budget=1000, reserve=100)
    assert led.hold(0, 800) == 800
    assert led.hold(0, 400) is None  # do not cut: the first call may spend far less
    led.release(800)
    assert led.hold(0, 400) == 400


def test_hold_cuts_when_nothing_is_in_flight():
    led = TokenLedger(budget=1000, reserve=100)
    led.charge(0, 500)
    assert led.hold(50, 3000) == 350
    assert led.held == 400


def test_hold_refuses_below_the_minimum():
    led = TokenLedger(budget=1000, reserve=100)
    led.charge(0, 850)
    assert led.hold(0, 3000, min_tokens=100) == 0
    assert led.held == 0
    assert led.hold(0, 50, min_tokens=100) == 50  # fits as asked: the minimum is only about cutting


def test_hold_after_overspend_refuses():
    led = TokenLedger(budget=1000, reserve=100)
    led.charge(0, 5000)
    assert led.hold(0, 1) == 0
