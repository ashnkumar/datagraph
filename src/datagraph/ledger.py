"""A small double-entry ledger with escrow.

Every movement of credits is an :class:`Entry` whose postings sum to zero, so credits can be
created only at the boundary (the ``EXTERNAL`` account) and never by an arithmetic slip inside
the system. A query escrows the researcher's payment before any work happens and the escrow is
drained to exactly zero on settlement or refund — a query cannot end with money stranded.

This is deliberately in-memory and single-process. Persistence lives in
:mod:`datagraph.registry`; the ledger's job is to be obviously correct.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["EXTERNAL", "Entry", "Ledger", "LedgerError"]

#: The boundary account. It is the only account allowed to hold a negative balance: credits
#: entering the system are a transfer *from* the outside world, so ``EXTERNAL`` runs negative by
#: exactly the number of credits in circulation.
EXTERNAL = "external"


class LedgerError(Exception):
    """Raised when an operation would violate a ledger invariant."""


@dataclass(frozen=True)
class Entry:
    """One balanced movement of credits. ``postings`` maps account -> signed delta."""

    memo: str
    postings: Mapping[str, int]


class Ledger:
    """An append-only double-entry ledger."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._balances: defaultdict[str, int] = defaultdict(int)
        self._escrows: dict[str, int] = {}

    # -- core ---------------------------------------------------------------------------

    def post(self, memo: str, postings: Mapping[str, int]) -> Entry:
        """Record a balanced entry.

        Raises:
            LedgerError: If the postings do not sum to zero, or if applying them would drive
                any account other than ``EXTERNAL`` negative.
        """
        if not postings:
            raise LedgerError("an entry must have at least one posting")

        total = sum(postings.values())
        if total != 0:
            raise LedgerError(f"entry {memo!r} does not balance: postings sum to {total}, not 0")

        for account, delta in postings.items():
            if account == EXTERNAL:
                continue
            if self._balances[account] + delta < 0:
                raise LedgerError(
                    f"entry {memo!r} would overdraw {account!r}: "
                    f"balance {self._balances[account]}, delta {delta}"
                )

        for account, delta in postings.items():
            self._balances[account] += delta

        entry = Entry(memo=memo, postings=dict(postings))
        self._entries.append(entry)
        return entry

    def balance(self, account: str) -> int:
        return self._balances[account]

    def balances(self) -> dict[str, int]:
        """All non-zero balances, including ``EXTERNAL``."""
        return {a: b for a, b in self._balances.items() if b != 0}

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def credits_in_circulation(self) -> int:
        """Credits held by real accounts — the mirror of the ``EXTERNAL`` balance."""
        return -self._balances[EXTERNAL]

    def check_invariants(self) -> None:
        """Assert the ledger's global invariants. Cheap enough to call after every operation.

        Raises:
            LedgerError: If any invariant is broken.
        """
        if (total := sum(self._balances.values())) != 0:
            raise LedgerError(f"ledger does not balance: all accounts sum to {total}, not 0")

        for account, bal in self._balances.items():
            if account != EXTERNAL and bal < 0:
                raise LedgerError(f"account {account!r} is negative: {bal}")

        for escrow, amount in self._escrows.items():
            if self._balances[escrow] != amount:
                raise LedgerError(
                    f"escrow {escrow!r} holds {self._balances[escrow]}, expected {amount}"
                )

    # -- escrow -------------------------------------------------------------------------

    def fund(self, account: str, amount: int) -> Entry:
        """Bring ``amount`` credits into the system and place them in ``account``."""
        if amount <= 0:
            raise LedgerError(f"funding amount must be positive, got {amount}")
        return self.post(f"fund {account}", {account: amount, EXTERNAL: -amount})

    def open_escrow(self, escrow: str, payer: str, amount: int) -> Entry:
        """Move ``amount`` from ``payer`` into a held ``escrow`` account."""
        if amount <= 0:
            raise LedgerError(f"escrow amount must be positive, got {amount}")
        if escrow in self._escrows:
            raise LedgerError(f"escrow {escrow!r} is already open")

        entry = self.post(f"escrow {escrow}", {payer: -amount, escrow: amount})
        self._escrows[escrow] = amount
        return entry

    def settle_escrow(self, escrow: str, payouts: Mapping[str, int]) -> Entry:
        """Distribute a held escrow to ``payouts``, which must exhaust it exactly.

        Raises:
            LedgerError: If the escrow is not open, any payout is negative, or the payouts do
                not sum to the escrowed amount. The last case is the one that matters: it is
                what stops an attribution bug from quietly creating or destroying credits.
        """
        held = self._require_open(escrow)

        for account, amount in payouts.items():
            if amount < 0:
                raise LedgerError(f"payout to {account!r} is negative: {amount}")

        total = sum(payouts.values())
        if total != held:
            raise LedgerError(f"payouts for {escrow!r} sum to {total} but the escrow holds {held}")

        postings: defaultdict[str, int] = defaultdict(int)
        postings[escrow] -= held
        for account, amount in payouts.items():
            postings[account] += amount

        entry = self.post(f"settle {escrow}", dict(postings))
        del self._escrows[escrow]
        return entry

    def refund_escrow(self, escrow: str, payee: str) -> Entry:
        """Return a held escrow to ``payee`` in full."""
        held = self._require_open(escrow)
        entry = self.post(f"refund {escrow}", {escrow: -held, payee: held})
        del self._escrows[escrow]
        return entry

    def open_escrows(self) -> dict[str, int]:
        return dict(self._escrows)

    def _require_open(self, escrow: str) -> int:
        if escrow not in self._escrows:
            raise LedgerError(f"escrow {escrow!r} is not open")
        return self._escrows[escrow]
