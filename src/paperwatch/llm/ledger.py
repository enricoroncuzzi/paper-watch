"""Token and cost ledger with a hard per-run cap.

Every provider call is charged here before and after it runs, so a bug that
loops over a paid API hits `BudgetExceeded` instead of an invoice.
"""


class BudgetExceeded(Exception):
    pass


class Ledger:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.tokens_used = 0
        self.entries: list[tuple[str, int]] = []

    def charge(self, model: str, tokens: int) -> None:
        if self.tokens_used + tokens > self.max_tokens:
            raise BudgetExceeded(f"{self.tokens_used}+{tokens} > {self.max_tokens}")
        self.tokens_used += tokens
        self.entries.append((model, tokens))
