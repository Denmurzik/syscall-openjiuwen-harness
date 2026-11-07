import threading


class TokenLedger:
    def __init__(self, budget: float, reserve: float):
        self.budget = budget
        self.reserve = reserve
        self.spent = 0.0
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

    def mark_exhausted(self) -> None:
        with self._lock:
            self.exhausted = True
