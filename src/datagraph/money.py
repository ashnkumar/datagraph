"""Integer money and exact apportionment.

Every amount in this system is an integer count of **credits** (minor units). Attribution
produces real-valued weights; this module is the single place those weights become money, and
the only place rounding happens.

The reference implementation this project reacts to computed payouts as
``payment_amount / row_count`` in floating point and reported the result with ``.toFixed(2)``.
Payouts did not sum to payments, and the difference silently vanished. Nothing here uses a
float to represent money.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["AllocationError", "allocate"]


class AllocationError(ValueError):
    """Raised when an allocation cannot be performed as requested."""


def allocate(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` credits across ``weights`` so the result sums to exactly ``total``.

    Uses the largest-remainder (Hamilton) method: floor each proportional share, then hand the
    leftover credits one at a time to the largest fractional remainders. Ties are broken by
    index, so the result is deterministic for a given input.

    Two edge cases have deliberate, documented behaviour:

    * **All weights zero** — splits equally. This is the natural limit and it preserves the sum
      invariant, but it is rarely the right *policy*: if nothing contributed, nobody has earned
      anything. Callers deciding how to settle a payment should check for this case themselves
      rather than relying on the split. :func:`datagraph.marketplace.Marketplace.query` refunds
      instead of calling this.
    * **Negative weights** — rejected. A negative contribution has no meaning here, and silently
      clamping one to zero would hide a bug in an attribution engine.

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
    pool = float(sum(weights))

    # All-zero weights split equally — the documented limit case; see the docstring for why
    # callers should decide the policy themselves rather than leaning on it.
    shares = [1.0 / n] * n if pool == 0.0 else [w / pool for w in weights]

    exact = [total * s for s in shares]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)

    # Hand out the leftover credits to the largest fractional parts, ties broken by index.
    order = sorted(range(n), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:remainder]:
        floors[i] += 1

    return floors
