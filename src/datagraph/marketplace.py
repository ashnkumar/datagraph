"""The whole loop: escrow, retrieve, redact, answer, attribute, settle.

The ordering matters and is deliberate. Payment is escrowed before any work happens, the
cohort floor is checked before the model is ever called, and settlement happens only after
attribution has produced weights that exhaust the payment. Every path out of
:meth:`Marketplace.query` either settles the escrow or refunds it in full.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from datagraph.attribution import (
    EFFICIENCY_TOLERANCE,
    Attribution,
    CoalitionValue,
    Similarity,
    TokenF1,
    attribute,
)
from datagraph.ledger import Ledger
from datagraph.models import ModelClient, ModelRefusal, ModelSubstituted
from datagraph.money import allocate
from datagraph.policy import DEFAULT_COHORT_FLOOR, CohortTooSmall, enforce_cohort_floor
from datagraph.registry import Registry, SourceView, provider_ids

__all__ = ["DEFAULT_MAX_SOURCES", "Marketplace", "QueryResult"]

#: Cap on records used to answer one query. This bounds the attribution player set, and so
#: the coalition space: measuring contribution honestly costs up to 2^n model calls, and n is
#: the only lever on that. Six keeps a live query under 64 generations.
DEFAULT_MAX_SOURCES = 6


@dataclass(frozen=True)
class QueryResult:
    """Everything one query produced, including how it was paid for."""

    query_id: str
    question: str
    answer: str
    sources: Sequence[SourceView]
    """Disclosure-only views. A caller of :meth:`Marketplace.query` never receives raw values —
    including on a refunded query, where the records were retrieved but never answered from."""

    payouts: Mapping[str, int]
    """Provider id -> credits earned. Sums to the payment unless the query was refunded."""

    attribution: Attribution | None = None
    provider_weights: Mapping[str, float] = field(default_factory=dict)
    """Provider id -> measured share of the answer."""

    model_calls: int = 0
    refunded: bool = False
    refund_reason: str | None = None

    @property
    def total_paid(self) -> int:
        return sum(self.payouts.values())


def account(kind: str, entity_id: str) -> str:
    return f"{kind}:{entity_id}"


class Marketplace:
    """Runs queries against a registry, paying providers by measured contribution.

    Args:
        registry: Where providers, datasets, and records live.
        ledger: Where credits move.
        model: Answer generator.
        engine: ``"shapley"`` (default), ``"exact_shapley"``, or ``"leave_one_out"``.
        similarity: How answers are compared. Defaults to :class:`~datagraph.attribution.TokenF1`.
        cohort_floor: Minimum distinct providers behind an answer.
        max_sources: Cap on records used per query.
        permutations: Sample size for the Shapley estimator.
        seed: Seed for the Shapley estimator, so a query is reproducible.
    """

    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        model: ModelClient,
        engine: str = "shapley",
        similarity: Similarity | None = None,
        cohort_floor: int = DEFAULT_COHORT_FLOOR,
        max_sources: int = DEFAULT_MAX_SOURCES,
        permutations: int = 2000,
        seed: int = 0,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.model = model
        self.engine = engine
        self.similarity = similarity or TokenF1()
        self.cohort_floor = cohort_floor
        self.max_sources = max_sources
        self.permutations = permutations
        self.seed = seed
        self._query_seq = 0

    def fund_researcher(self, researcher_id: str, credits: int) -> None:
        self.ledger.fund(account("researcher", researcher_id), credits)

    def balance_of(self, kind: str, entity_id: str) -> int:
        return self.ledger.balance(account(kind, entity_id))

    def query(self, researcher_id: str, question: str, payment: int) -> QueryResult:
        """Answer ``question`` and pay the providers whose records shaped the answer.

        The escrow is opened first, so the payment is committed before any work is done. Every
        exit path settles or refunds — including exits by exception. Retrieval, generation,
        attribution, apportionment, and settlement can all fail for reasons that have nothing
        to do with the researcher (a timeout, a rate limit, a bad weight vector), and any of
        those leaving the escrow open would debit the payer and strand the credits, so the
        whole post-escrow body runs under a cleanup guard.
        """
        self._query_seq += 1
        query_id = f"q{self._query_seq}"
        escrow = account("escrow", query_id)
        researcher = account("researcher", researcher_id)

        self.ledger.open_escrow(escrow, researcher, payment)

        try:
            return self._run(query_id, question, payment, escrow, researcher)
        except BaseException:
            # Refund only if the escrow survived: after a successful settlement there is
            # nothing held, and refunding again would create credits. Re-raise either way —
            # the caller still needs to see the failure.
            if escrow in self.ledger.open_escrows():
                self.ledger.refund_escrow(escrow, researcher)
                self.ledger.check_invariants()
            raise

    def _run(
        self, query_id: str, question: str, payment: int, escrow: str, researcher: str
    ) -> QueryResult:
        # Cap any one provider at a fair fraction of the slots. Tying it to the cohort floor
        # keeps the two consistent: a full result set can always satisfy the floor, so a
        # provider padding its dataset cannot force a refusal.
        records = self.registry.search(
            question,
            limit=self.max_sources,
            max_per_provider=max(1, self.max_sources // self.cohort_floor),
        )
        # Convert at the boundary: nothing downstream — prompts, results, the CLI — is handed
        # an object carrying suppressed values.
        sources = [r.view() for r in records]

        try:
            enforce_cohort_floor(provider_ids(sources), self.cohort_floor)
        except CohortTooSmall as exc:
            return self._refund(query_id, question, escrow, researcher, sources, str(exc))

        value = CoalitionValue(
            question=question,
            sources=sources,
            model=self.model,
            similarity=self.similarity,
        )

        try:
            answer = value.reference_answer
            result = attribute(self.engine, value.players, value, **self._engine_kwargs())
        except (ModelRefusal, ModelSubstituted) as exc:
            return self._refund(query_id, question, escrow, researcher, sources, str(exc))

        # Weights are already per-provider: the players in the game are providers, so there is
        # no roll-up step and no way for a provider's share to depend on its row count.
        weights = result.clamped()

        # Nothing measurably contributed. Paying anyway would mean paying for retrieval rather
        # than for contribution, which is the failure this project exists to avoid — so the
        # researcher gets their credits back instead.
        if sum(weights.values()) <= 0:
            return self._refund(
                query_id,
                question,
                escrow,
                researcher,
                sources,
                f"{self.engine} attributed no contribution to any provider",
                attribution=result,
                answer=answer,
                model_calls=value.calls,
            )

        # The one way an efficient engine can stop being efficient. A negative marginal means a
        # provider's records made the answer worse; clamping it to zero is right — it is not a
        # debt — but it lifts the remaining weights above v(N), so together they now claim more
        # than the payment. `allocate` would divide them back down to fit, which is exactly the
        # silent transfer this project refuses in leave-one-out. There is no split of the escrow
        # that pays every positive contributor its measured share, so the honest move is to
        # decline to price the answer rather than to shade everyone down and say nothing.
        if result.clamped_excess > EFFICIENCY_TOLERANCE:
            return self._refund(
                query_id,
                question,
                escrow,
                researcher,
                sources,
                f"{self.engine} scored at least one provider below zero; the shares that "
                f"remain claim more than the payment, and settling would mean scaling them "
                f"to fit",
                attribution=result,
                answer=answer,
                model_calls=value.calls,
            )

        recipients = sorted(weights)
        shares = allocate(payment, [weights[p] for p in recipients])
        payouts = dict(zip(recipients, shares, strict=True))

        self.ledger.settle_escrow(
            escrow, {account("provider", p): amount for p, amount in payouts.items()}
        )
        self.ledger.check_invariants()

        return QueryResult(
            query_id=query_id,
            question=question,
            answer=answer,
            sources=sources,
            payouts=payouts,
            attribution=result,
            provider_weights=weights,
            model_calls=value.calls,
        )

    # -- internals ----------------------------------------------------------------------

    def _engine_kwargs(self) -> dict[str, object]:
        if self.engine == "shapley":
            return {"permutations": self.permutations, "seed": self.seed}
        return {}

    def _refund(
        self,
        query_id: str,
        question: str,
        escrow: str,
        researcher: str,
        sources: Sequence[SourceView],
        reason: str,
        attribution: Attribution | None = None,
        answer: str = "",
        model_calls: int = 0,
    ) -> QueryResult:
        self.ledger.refund_escrow(escrow, researcher)
        self.ledger.check_invariants()
        return QueryResult(
            query_id=query_id,
            question=question,
            answer=answer,
            sources=sources,
            payouts={},
            attribution=attribution,
            model_calls=model_calls,
            refunded=True,
            refund_reason=reason,
        )
