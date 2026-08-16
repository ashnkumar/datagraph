"""Measuring which sources actually changed the answer.

Paying providers for one answer is a **cost-allocation problem**, not a ranking problem. Framed
as a cooperative game:

* the **players** are the providers whose records reached the model — providers rather than
  records, because the Shapley value is not replication-proof and per-record players let a
  provider inflate its cut by cloning a row (see :class:`CoalitionValue`),
* the **characteristic function** ``v(S)`` is how much of the full answer is recoverable from
  the subset ``S``, normalized so ``v(∅) = 0`` and ``v(N) = 1``,
* the **payout** is each player's share of ``v(N)``.

Two engines are implemented against that frame, and the contrast between them is the point of
this module.

``leave_one_out`` is the intuitive one — "how much worse is the answer without you?" — and it
is **not an efficient allocation**: the weights do not sum to ``v(N)``. Its failure mode is
redundancy, which is the common case in a marketplace: two providers supply the same fact,
removing either changes nothing, so both score zero and the money has nowhere to go.

``shapley`` is the unique allocation satisfying efficiency, symmetry, null player, and
additivity. Efficiency is what makes settlement sound — the weights sum to ``v(N)`` exactly,
so the escrow is exhausted with no normalization fudge, and redundant providers split the
credit for the fact they jointly supply instead of both being zeroed.

Estimation follows Castro, Gómez & Tejada (2009), *Computers & Operations Research* 36(5),
1726–1730: sample random permutations and average each player's marginal contribution. Because
every permutation's marginals telescope to ``v(N) - v(∅)``, the estimator is efficient
*exactly*, not just in expectation — sampling error moves credit between players but never
creates or destroys any.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

from datagraph.models import ModelClient, Source
from datagraph.text import tokenize

__all__ = [
    "Attribution",
    "CoalitionValue",
    "Similarity",
    "TokenF1",
    "ValueFn",
    "attribute",
    "exact_shapley",
    "leave_one_out",
    "shapley",
]

#: ``v(S)`` — the value of the answer obtainable from a subset of the players.
ValueFn = Callable[[frozenset[str]], float]

#: Guard on exact Shapley, which is 2^n evaluations.
MAX_EXACT_PLAYERS = 12

#: How far a weight sum may drift from ``v(N)`` and still count as exhausting it. Efficiency is
#: exact in the mathematics; this covers floating-point accumulation over many permutations.
EFFICIENCY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Attribution:
    """The result of one attribution run."""

    engine: str
    weights: dict[str, float]
    grand_value: float
    """``v(N)`` — the value being allocated."""

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    @property
    def is_efficient(self) -> bool:
        """Whether the weights exhaust ``v(N)``, which is what settlement requires."""
        return abs(self.total_weight - self.grand_value) < EFFICIENCY_TOLERANCE

    @property
    def clamped_excess(self) -> float:
        """How much weight :meth:`clamped` adds to the total. Zero unless a marginal is negative.

        When this is not zero the clamped vector claims more than ``v(N)``, so no split of the
        escrow can pay every positive contributor its efficient share.
        """
        return sum(-w for w in self.weights.values() if w < 0)

    def clamped(self) -> dict[str, float]:
        """Weights with negatives floored at zero, ready for :func:`datagraph.money.allocate`.

        A negative marginal contribution means removing a source *improved* the answer. That
        is meaningful signal but it is not a debt, so it becomes a zero payout rather than a
        charge.

        **Clamping breaks efficiency, and that is not a rounding detail.** The raw weights sum
        to ``v(N)``; flooring a negative one lifts the sum above it, so the providers that are
        left claim more than the whole payment between them. Handing that vector to
        :func:`datagraph.money.allocate` would divide it back down to fit — the same silent
        transfer this project refuses in leave-one-out, performed on the engine that is
        supposed to be immune to it. :attr:`clamped_excess` is how a caller detects the case,
        and :meth:`datagraph.marketplace.Marketplace.query` refunds rather than settle it.
        """
        return {k: max(0.0, w) for k, w in self.weights.items()}


class Similarity:
    """Scores how much of ``reference`` survives in ``candidate``. Symmetric, in ``[0, 1]``."""

    def __call__(self, candidate: str, reference: str) -> float:  # pragma: no cover - protocol
        raise NotImplementedError


class TokenF1(Similarity):
    """F1 over content-word sets.

    Deterministic and dependency-free, which is what lets the offline suite exercise the real
    attribution path. It measures whether the same *content* survived, not whether the same
    *meaning* did — an honest proxy for the extractive question-answering this system does,
    where an answer is grounded in retrieved records and a source's contribution shows up as
    content appearing or disappearing. Swap in an embedding-backed implementation for semantic
    agreement; no engine below has to change.
    """

    def __call__(self, candidate: str, reference: str) -> float:
        a, b = set(tokenize(candidate)), set(tokenize(reference))
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        overlap = len(a & b)
        if overlap == 0:
            return 0.0

        precision = overlap / len(a)
        recall = overlap / len(b)
        return 2 * precision * recall / (precision + recall)


@dataclass
class CoalitionValue:
    """Builds ``v(S)`` by regenerating the answer from ``S`` and scoring it against ``v(N)``.

    **The players are providers, not records.** A coalition names a set of providers, and the
    answer for that coalition is generated from *all* of their retrieved records at once. This
    is not a detail — it is what stops the obvious attack.

    The Shapley value is not replication-proof. If each record were its own player and a
    provider's payout were the sum of their records' shares, a provider could split one record
    into four identical copies and take a larger cut for contributing nothing new: measured on
    the demo data, one provider went from 446 to 612 credits out of 1000 by cloning a single
    record four times. Grouping by provider makes the payout invariant to how a provider
    happens to divide its data into rows, because duplicating a row changes neither the player
    set nor any coalition's content.

    It is also cheaper. The coalition space is ``2^providers`` rather than ``2^records``, and
    providers are never more numerous than their records.

    Memoized on the coalition, because ``v(S)`` depends only on the *set* — the same subset
    reached by two different permutations costs one model call, not two.
    """

    question: str
    sources: Sequence[Source]
    model: ModelClient
    similarity: Similarity = field(default_factory=TokenF1)

    _cache: dict[frozenset[str], float] = field(default_factory=dict, init=False, repr=False)
    _reference: str | None = field(default=None, init=False, repr=False)
    _floor: float | None = field(default=None, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)

    @property
    def players(self) -> list[str]:
        """The distinct providers behind the retrieved records."""
        return sorted({s.provider_id for s in self.sources})

    @property
    def calls(self) -> int:
        """Model calls made. Distinct coalitions evaluated, thanks to memoization."""
        return self._calls

    @property
    def reference_answer(self) -> str:
        """The answer from all sources — generated once, and what the user is shown."""
        if self._reference is None:
            self._reference = self._generate(frozenset(self.players))
        return self._reference

    @property
    def floor(self) -> float:
        """Similarity between the *no-source* answer and the reference answer.

        Every answer shares some boilerplate with every other — "the records show…" against
        "the records do not support an answer". That shared vocabulary is a constant floor
        under the raw similarity, and it is not evidence that anything contributed. Measuring
        from it rather than from zero is what makes a source that changes nothing score
        exactly zero instead of inheriting the boilerplate as earnings.
        """
        if self._floor is None:
            self._floor = self.similarity(self._generate(frozenset()), self.reference_answer)
        return self._floor

    def __call__(self, coalition: frozenset[str]) -> float:
        """``v(S)``, rescaled so that ``v(∅) = 0`` and ``v(N) = 1``."""
        if not coalition:
            return 0.0

        if coalition in self._cache:
            return self._cache[coalition]

        # The grand coalition reuses the reference answer rather than regenerating it. That
        # saves a call, and it also makes v(N) exactly 1: with a live model a second
        # generation would return different text and the value being allocated would drift.
        generated = (
            self.reference_answer
            if coalition == frozenset(self.players)
            else self._generate(coalition)
        )
        raw = self.similarity(generated, self.reference_answer)
        value = _rescale(raw, self.floor)
        self._cache[coalition] = value
        return value

    def _generate(self, coalition: frozenset[str]) -> str:
        self._calls += 1
        # Every record belonging to a provider in the coalition, so a provider is present or
        # absent as a whole and cannot gain by splitting its data across more rows.
        subset = [s for s in self.sources if s.provider_id in coalition]
        return self.model.answer(self.question, subset)


def _rescale(raw: float, floor: float) -> float:
    """Map a raw similarity onto ``[0, 1]`` with the boilerplate floor removed."""
    headroom = 1.0 - floor
    if headroom <= 1e-9:
        # Degenerate: the answer is the same with no sources as with all of them, so nothing
        # was contributed by anything. v ≡ 0, every weight is zero, and the marketplace
        # refunds rather than paying for an answer the data did not shape.
        return 0.0
    return min(1.0, max(0.0, (raw - floor) / headroom))


def leave_one_out(players: Sequence[str], value: ValueFn) -> Attribution:
    """φᵢ = v(N) − v(N∖{i}). Cheap, intuitive, and not an efficient allocation.

    Costs ``n + 1`` calls to ``value``: the grand coalition plus one per player. Note that
    *evaluations of v* and *model calls* are not the same count — :class:`CoalitionValue` also
    generates a no-source answer to establish its floor, so the real cost through the shipped
    path is ``n + 2`` generations, six for the four-provider demo. See the module docstring for
    why the resulting weights should not be used to settle a payment without understanding what
    they do on redundant sources — :func:`datagraph.attribution.shapley` is the default for
    exactly that reason.
    """
    grand = frozenset(players)
    total = value(grand)
    weights = {p: total - value(grand - {p}) for p in players}
    return Attribution(engine="leave_one_out", weights=weights, grand_value=total)


def shapley(
    players: Sequence[str],
    value: ValueFn,
    permutations: int = 2000,
    seed: int = 0,
) -> Attribution:
    """Monte-Carlo estimate of the Shapley value.

    Walks ``permutations`` random arrival orders, accumulating each player's marginal
    contribution as they join. Seeded, so a given query is reproducible.

    Sampling affects *how credit is divided*, never *how much*: each permutation's marginals
    telescope to ``v(N) − v(∅)``, so the average sums to ``v(N)`` for any number of
    permutations, including one.

    Args:
        players: Source ids.
        value: The characteristic function.
        permutations: Random orders to sample. More reduces variance between players.
            The default is high because memoization makes it nearly free: distinct coalitions
            are bounded by 2^n, so past the point where the cache saturates, extra
            permutations cost dictionary lookups rather than model calls.
        seed: Seed for the permutation sampler.

    Note:
        At small ``n`` that bound cuts both ways — once sampling has visited most of the
        coalition space it has paid for the whole space anyway, and :func:`exact_shapley` costs
        the same with no variance. Sampling is the right tool when you deliberately keep
        ``permutations`` below saturation, or when ``n`` is large enough that 2^n is out of
        reach.
    """
    if permutations < 1:
        raise ValueError(f"permutations must be at least 1, got {permutations}")

    ordered = list(players)
    grand = value(frozenset(ordered))
    if not ordered:
        return Attribution(engine="shapley", weights={}, grand_value=grand)

    rng = random.Random(seed)
    totals = dict.fromkeys(ordered, 0.0)

    for _ in range(permutations):
        rng.shuffle(ordered)
        coalition: set[str] = set()
        running = 0.0  # v(∅)
        for player in ordered:
            coalition.add(player)
            nxt = value(frozenset(coalition))
            totals[player] += nxt - running
            running = nxt

    weights = {p: t / permutations for p, t in totals.items()}
    return Attribution(engine="shapley", weights=weights, grand_value=grand)


def exact_shapley(players: Sequence[str], value: ValueFn) -> Attribution:
    """Exact Shapley value by enumeration. ``2^n`` evaluations — for small games and tests."""
    ordered = list(players)
    n = len(ordered)
    if n > MAX_EXACT_PLAYERS:
        raise ValueError(
            f"exact Shapley over {n} players needs 2^{n} evaluations; "
            f"use shapley() for more than {MAX_EXACT_PLAYERS}"
        )

    grand = value(frozenset(ordered))
    weights = dict.fromkeys(ordered, 0.0)

    for player in ordered:
        others = [p for p in ordered if p != player]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                s = frozenset(subset)
                weights[player] += weight * (value(s | {player}) - value(s))

    return Attribution(engine="exact_shapley", weights=weights, grand_value=grand)


_ENGINES: dict[str, Callable[..., Attribution]] = {
    "shapley": shapley,
    "exact_shapley": exact_shapley,
    "leave_one_out": leave_one_out,
}


def attribute(engine: str, players: Iterable[str], value: ValueFn, **kwargs: object) -> Attribution:
    """Run a named engine. ``kwargs`` are passed through to it."""
    try:
        fn = _ENGINES[engine]
    except KeyError:
        raise ValueError(
            f"unknown engine {engine!r}; choose from {', '.join(sorted(_ENGINES))}"
        ) from None
    return fn(list(players), value, **kwargs)
