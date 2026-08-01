"""End-to-end tests for the query loop, offline against the deterministic FakeModel."""

import pytest

from datagraph.ledger import Ledger
from datagraph.marketplace import Marketplace, account
from datagraph.models import FakeModel, ModelRefusal
from datagraph.policy import Disclosure, DisclosurePolicy
from datagraph.registry import Registry
from datagraph.sample_data import DEMO_QUESTION, seed_demo

PAYMENT = 1000


@pytest.fixture
def market():
    registry = seed_demo(Registry())
    ledger = Ledger()
    # Default permutation count: memoisation caps model calls at 2^n regardless, so there is
    # no reason to sample sparsely and inherit the variance.
    mkt = Marketplace(registry, ledger, FakeModel(), seed=11)
    mkt.fund_researcher("rachel", 100_000)
    yield mkt
    registry.close()


def test_query_answers_pays_and_conserves_credits(market):
    before = market.ledger.credits_in_circulation()
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    assert not result.refunded
    assert result.answer
    assert result.total_paid == PAYMENT
    assert market.ledger.credits_in_circulation() == before
    assert market.ledger.open_escrows() == {}
    market.ledger.check_invariants()


def test_payouts_land_in_provider_accounts(market):
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    for provider_id, amount in result.payouts.items():
        assert market.ledger.balance(account("provider", provider_id)) == amount
    assert market.balance_of("researcher", "rachel") == 100_000 - PAYMENT


def test_hidden_fields_never_reach_the_answer(market):
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    # Both the answer and every rendered source must be free of suppressed values.
    rendered = result.answer + " ".join(s.render() for s in result.sources)
    assert "synthetic-participant" not in rendered
    assert "1992-02-29" not in rendered

    # ...while the raw values are still in the store, which is the point of the distinction.
    raw = {s.id: s.values for s in result.sources}
    assert raw["rec-01"]["participant_ref"] == "synthetic-participant-0001"


def test_derived_fields_reach_the_model_only_as_bands(market):
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)
    rec01 = next(s for s in result.sources if s.id == "rec-01")

    assert rec01.disclosed["age"] == "30-39"
    assert "34" not in rec01.render()


def test_redundant_providers_are_paid_equally_and_not_zero(market):
    """The headline behaviour, through the real loop.

    'borealis' and 'cascade' disclose identical records. Under leave-one-out they would each
    score zero. Under Shapley they split the credit for the fact they jointly supply, and
    symmetry means they split it evenly — up to the estimator's sampling error.
    """
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    assert result.payouts["borealis"] > 0
    assert result.payouts["cascade"] > 0
    assert abs(result.payouts["borealis"] - result.payouts["cascade"]) <= 0.02 * PAYMENT


def test_exact_shapley_pays_symmetric_providers_identically(market):
    """With no sampling error, symmetry is exact rather than approximate."""
    market.engine = "exact_shapley"
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    # Largest-remainder allocation can differ by a single credit when a share is not a whole
    # number; the underlying weights are equal.
    assert abs(result.payouts["borealis"] - result.payouts["cascade"]) <= 1
    assert result.source_weights["rec-02"] == pytest.approx(result.source_weights["rec-03"])


def test_leave_one_out_pays_redundant_providers_nothing_and_reassigns_their_share(market):
    """The defect, made operational on real data.

    Leave-one-out does not fail loudly here — it fails *quietly*, which is worse. The two
    providers holding corroborated data score exactly zero, and because the weights have to be
    normalised to settle the payment, the credits they should have earned are silently handed
    to the providers whose data happened to be unique.
    """
    market.engine = "leave_one_out"
    loo = market.query("rachel", DEMO_QUESTION, PAYMENT)

    assert loo.payouts["borealis"] == 0
    assert loo.payouts["cascade"] == 0
    assert loo.total_paid == PAYMENT  # the money still moves — just to the wrong people

    assert loo.attribution is not None
    assert not loo.attribution.is_efficient
    assert loo.attribution.total_weight < loo.attribution.grand_value

    # Shapley, on the identical query, pays them.
    market.engine = "shapley"
    fair = market.query("rachel", DEMO_QUESTION, PAYMENT)

    assert fair.payouts["borealis"] > 0
    assert fair.payouts["cascade"] > 0
    assert fair.payouts["aurora"] < loo.payouts["aurora"]  # aurora was over-credited
    market.ledger.check_invariants()


def test_shapley_attribution_is_efficient_in_a_real_run(market):
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)
    assert result.attribution is not None
    assert result.attribution.is_efficient


def test_a_narrow_query_is_refused_before_the_model_is_called():
    class ExplodingModel:
        def answer(self, question, sources):  # pragma: no cover - must never run
            raise AssertionError("the cohort floor must block generation")

    registry = Registry()
    registry.add_provider("solo", "Solo Provider")
    registry.add_dataset(
        "solo-ds", "solo", "ds", DisclosurePolicy(levels={"region": Disclosure.OPEN})
    )
    registry.add_record("only", "solo-ds", {"region": "northern"})

    market = Marketplace(registry, Ledger(), ExplodingModel())
    market.fund_researcher("rachel", 5_000)

    result = market.query("rachel", "northern region", 500)

    assert result.refunded
    assert "cohort floor" in (result.refund_reason or "")
    assert market.balance_of("researcher", "rachel") == 5_000
    registry.close()


def test_a_query_matching_nothing_is_refunded(market):
    result = market.query("rachel", "zzz nonexistent vocabulary", PAYMENT)

    assert result.refunded
    assert result.sources == []
    assert market.balance_of("researcher", "rachel") == 100_000


def test_a_model_refusal_refunds_rather_than_charging(market):
    class RefusingModel:
        def answer(self, question, sources):
            raise ModelRefusal(category="bio", explanation="declined")

    market.model = RefusingModel()
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)

    assert result.refunded
    assert "declined" in (result.refund_reason or "")
    assert market.balance_of("researcher", "rachel") == 100_000
    market.ledger.check_invariants()


def test_a_researcher_cannot_spend_credits_they_do_not_have(market):
    from datagraph.ledger import LedgerError

    with pytest.raises(LedgerError, match="would overdraw"):
        market.query("rachel", DEMO_QUESTION, 500_000)


def test_repeated_queries_are_reproducible(market):
    a = market.query("rachel", DEMO_QUESTION, PAYMENT)
    b = market.query("rachel", DEMO_QUESTION, PAYMENT)
    assert a.payouts == b.payouts


def test_memoisation_keeps_the_call_count_at_the_coalition_bound(market):
    result = market.query("rachel", DEMO_QUESTION, PAYMENT)
    n = len(result.sources)

    # 200 permutations x n players would be 800 naive evaluations; the distinct-coalition
    # bound is 2^n - 1 non-empty subsets plus the no-source answer used to calibrate v.
    assert result.model_calls <= 2**n
