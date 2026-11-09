import math
import threading


class TokenLedger:
    def __init__(self, budget: float, reserve: float):
        self.budget = budget
        self.reserve = reserve
        self.spent = 0.0
        # what the calls in flight can still cost, in whole tokens rounded up: a sum of floats would not
        # return to exactly 0, and hold() takes anything above 0 for a call in flight
        self.held = 0
        self.exhausted = False
        self._lock = threading.Lock()

    @staticmethod
    def cost(prompt: int, completion: int) -> float:
        return prompt / 10 + completion

    def charge(self, prompt: int, completion: int) -> float:
        c = self.cost(prompt, completion)
        with self._lock:
            self.spent += c
        return c

    def remaining(self) -> float:
        with self._lock:
            return self.budget - self.spent

    def can_afford(self, estimate: float) -> bool:
        with self._lock:
            return not self.exhausted and estimate <= self.budget - self.spent - self.reserve

    def hold(self, prompt_cost: float, max_tokens: int, min_tokens: int = 1) -> int | None:
        """Set aside the most a call can cost; returns the output limit to send it with.

        The full `max_tokens` if it fits. Otherwise None while other calls are in flight (they usually
        cost far less than held: ask again when they end), then the limit cut to what is left, or 0 when
        that is below `min_tokens` and the call must not be made. `release(prompt_cost + limit)` ends it.
        """
        with self._lock:
            left = self.budget - self.spent - self.reserve - self.held
            if prompt_cost + max_tokens <= left:
                limit = max_tokens
            elif self.held > 0:
                return None
            else:
                limit = int(left - prompt_cost)
                if limit < min_tokens:
                    return 0
            self.held += math.ceil(prompt_cost + limit)
            return limit

    def release(self, amount: float) -> None:
        with self._lock:
            self.held = max(0, self.held - math.ceil(amount))

    def mark_exhausted(self) -> None:
        with self._lock:
            self.exhausted = True
