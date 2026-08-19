"""Integer money and exact apportionment.

Every amount in this system is an integer count of **credits** (minor units). Attribution
produces real-valued weights; this module is the single place those weights become money, and
the only place rounding happens.

Computing payouts with floating-point division and rounding only for display can leave the
reported payouts short of the payment. Nothing here uses a float to represent money.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

__all__ = ["AllocationError", "allocate"]


class AllocationError(ValueError):
    """Raised when an allocation cannot be performed as requested."""


def allocate(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` credits across ``weights`` so the result sums to exactly ``total``.

    Uses the largest-remainder (Hamilton) method: floor each proportional share, then hand the
    leftover credits one at a time to the largest fractional remainders. Ties are broken by
    index, so the result is deterministic for a given input.

    Two edge cases have deliberate, documented behavior:

    * **All weights zero** — splits equally. This is the natural limit and it preserves the sum
      invariant, but it is rarely the right *policy*: if nothing contributed, nobody has earned
      anything. Callers deciding how to settle a payment should check for this case themselves
      rather than relying on the split. :func:`datagraph.marketplace.Marketplace.query` refunds
      instead of calling this.
    * **Negative weights** — rejected. A negative contribution has no meaning here, and silently
      clamping one to zero would hide a bug in an attribution engine.

    **On dividing by the pool.** Each quota is ``total * weight / sum(weights)``, which is
    proportional allocation — the same arithmetic that, applied to weights which do not sum to
    the whole, is the silent transfer this project exists to refuse. It is safe here only
    because of what the caller guarantees about the vector it passes: the Shapley weights sum
    to ``v(N)``, the payment buys ``v(N)``, and so the pool is one and the division changes
    nothing. That is a property of the *attribution*, not of this function, and this function
    cannot check it — the weights arrive with no record of what they were meant to exhaust.
    :meth:`datagraph.marketplace.Marketplace.query` is where it is enforced: it refuses to
    settle a vector that clamping has pushed above ``v(N)`` rather than letting the line below
    quietly scale it back down. Hand this function a vector that does not exhaust the value it
    was measured against and it will balance the books by moving money between recipients.

    Args:
        total: Credits to divide. Must be non-negative.
        weights: Relative shares, one per recipient. Must be non-negative and finite.

    Returns:
        One integer per weight, summing to exactly ``total``.

    Raises:
        AllocationError: If ``total`` is negative, ``weights`` is empty, or any weight is
            negative, NaN, or infinite.
    """
    if total < 0:
        raise AllocationError(f"total must be non-negative, got {total}")
    if not weights:
        raise AllocationError("weights must not be empty")

    for i, w in enumerate(weights):
        # NaN fails every comparison, so test it as `not (w >= 0)` rather than `w < 0`.
        if not (w >= 0) or w == float("inf"):
            raise AllocationError(f"weight[{i}] must be non-negative and finite, got {w!r}")

    n = len(weights)

    # Exact rational arithmetic, not floating point. Every float is exactly representable as a
    # Fraction, so this loses nothing on input and cannot overflow or round on the way through.
    # Floats were wrong here in two ways that only show up at the edges of the accepted range:
    # a weight sum large enough to overflow to inf drove every share to zero, and a `total`
    # beyond 2^53 lost integer precision, so the floors summed short by more than the recipient
    # count and the largest-remainder pass could not make it up.
    pool = sum((Fraction(w) for w in weights), Fraction(0))

    # All-zero weights split equally — the documented limit case; see the docstring for why
    # callers should decide the policy themselves rather than leaning on it.
    if pool == 0:
        quotas = [Fraction(total, n)] * n
    else:
        quotas = [Fraction(total) * Fraction(w) / pool for w in weights]

    floors = [q.numerator // q.denominator for q in quotas]
    remainder = total - sum(floors)

    # Hand out the leftover credits to the largest fractional parts, ties broken by index.
    # With exact quotas each floor is within 1 of its quota, so 0 <= remainder < n and the
    # slice below always covers the whole shortfall.
    order = sorted(range(n), key=lambda i: (floors[i] - quotas[i], i))
    for i in order[:remainder]:
        floors[i] += 1

    return floors
