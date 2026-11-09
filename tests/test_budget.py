import random

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


def test_holds_of_different_sizes_release_to_exactly_zero():
    """Float sums of prompt costs left a residue in held: a later call took it for a call in flight."""
    rnd = random.Random(7)
    for _ in range(500):
        led = TokenLedger(100000, 1000)
        live = []
        for _ in range(30):
            if len(live) < 3 and (not live or rnd.random() < 0.6):
                cost = rnd.randint(300, 9000) // 4 / 10
                live.append(cost + led.hold(cost, rnd.choice([250, 400, 600, 3000])))
            else:
                led.release(live.pop(rnd.randrange(len(live))))
        for amount in live:
            led.release(amount)
        assert led.held == 0


def test_call_that_does_not_fit_is_cut_after_mixed_holds_end():
    led = TokenLedger(10000, 100)
    costs = [60.7, 60.3, 90.9, 0.3, 12.1]
    limits = [led.hold(c, 400) for c in costs]
    for c, limit in sorted(zip(costs, limits), reverse=True):
        led.release(c + limit)
    led.charge(0, 9700)
    assert led.hold(0.3, 3000) == 199
