"""Tests for the attribution engines.

Most of these run against synthetic characteristic functions with known closed-form Shapley
values, so they check the mathematics rather than a model's behavior. The last section wires
the real path together with the deterministic FakeModel.
"""

from dataclasses import dataclass

import pytest

from datagraph.attribution import (
    Attribution,
    CoalitionValue,
    TokenF1,
    attribute,
    exact_shapley,
    leave_one_out,
    shapley,
)
from datagraph.models import FakeModel

# --- synthetic games -----------------------------------------------------------------------


def additive(contributions: dict[str, float]):
    """Every player contributes independently. Both engines should agree here."""
    total = sum(contributions.values())

    def v(coalition):
        return sum(contributions[p] for p in coalition) / total

    return v


def redundant_pair():
    """Players 'a' and 'b' supply the same fact; 'c' supplies the other one.

    Each fact is worth half the answer. This is the shape a marketplace actually accumulates —
    several providers holding overlapping data — and it is where leave-one-out breaks.
    """
    facts = {"a": {"f1"}, "b": {"f1"}, "c": {"f2"}}

    def v(coalition):
        covered = set().union(*(facts[p] for p in coalition)) if coalition else set()
        return len(covered) / 2

    return v


def fully_redundant():
    """Both players supply the only fact. The extreme case: leave-one-out yields all zeros."""

    def v(coalition):
        return 1.0 if coalition else 0.0

    return v


def complementary_players():
    """Every player is required; removing any one collapses the value."""

    required = {"a", "b", "c"}

    def v(coalition):
        return 1.0 if required <= set(coalition) else 0.0

    return v


# --- properties that make settlement sound -------------------------------------------------


def test_shapley_is_efficient_so_the_escrow_is_exhausted():
    v = redundant_pair()
    result = shapley(["a", "b", "c"], v, permutations=32)

    assert result.grand_value == 1.0
    assert result.total_weight == pytest.approx(1.0)
    assert result.is_efficient


@pytest.mark.parametrize("permutations", [1, 2, 7, 64])
def test_shapley_efficiency_holds_at_any_sample_size(permutations):
    # Marginals telescope within each permutation, so sampling moves credit between players
    # but never creates or destroys any. This is why sampling is safe for settlement.
    v = redundant_pair()
    result = shapley(["a", "b", "c"], v, permutations=permutations)
    assert result.total_weight == pytest.approx(1.0)


def test_shapley_pays_a_null_player_exactly_zero():
    # 'z' never changes any answer. The property that makes attribution mean anything.
    v = additive({"a": 3.0, "b": 1.0, "z": 0.0})
    result = exact_shapley(["a", "b", "z"], v)

    assert result.weights["z"] == pytest.approx(0.0)
    assert result.weights["a"] == pytest.approx(0.75)
    assert result.weights["b"] == pytest.approx(0.25)


def test_shapley_treats_identical_contributors_identically():
    v = redundant_pair()
    result = exact_shapley(["a", "b", "c"], v)
    assert result.weights["a"] == pytest.approx(result.weights["b"])


def test_shapley_splits_redundant_credit_instead_of_zeroing_it():
    # Closed form: 'a' and 'b' share the fact worth 1/2, so each earns 1/4; 'c' earns 1/2.
    v = redundant_pair()
    result = exact_shapley(["a", "b", "c"], v)

    assert result.weights["a"] == pytest.approx(0.25)
    assert result.weights["b"] == pytest.approx(0.25)
    assert result.weights["c"] == pytest.approx(0.50)


# --- the contrast: where leave-one-out fails ------------------------------------------------


def test_leave_one_out_is_not_efficient_on_redundant_sources():
    """The headline defect.

    Two providers supply an indispensable fact between them. Remove either and the answer is
    unchanged, so each scores zero — and half the payment has nowhere to go.
    """
    v = redundant_pair()
    loo = leave_one_out(["a", "b", "c"], v)

    assert loo.weights["a"] == pytest.approx(0.0)
    assert loo.weights["b"] == pytest.approx(0.0)
    assert loo.weights["c"] == pytest.approx(0.5)

    assert loo.total_weight == pytest.approx(0.5)
    assert not loo.is_efficient  # only half the payment is accounted for

    # Shapley, on the same game, pays them and exhausts the payment.
    exact = exact_shapley(["a", "b", "c"], v)
    assert exact.weights["a"] > 0
    assert exact.weights["b"] > 0
    assert exact.is_efficient


def test_leave_one_out_collapses_entirely_when_everything_is_redundant():
    """All weights zero, so normalizing divides by zero — there is no payout to compute."""
    v = fully_redundant()
    loo = leave_one_out(["a", "b"], v)

    assert loo.total_weight == pytest.approx(0.0)
    assert not loo.is_efficient

    exact = exact_shapley(["a", "b"], v)
    assert exact.weights == pytest.approx({"a": 0.5, "b": 0.5})
    assert exact.is_efficient


def test_leave_one_out_can_overshoot_on_complementary_sources():
    """Every necessary provider claims the full value, so normalization dilutes them all."""
    loo = leave_one_out(["a", "b", "c"], complementary_players())

    assert loo.weights == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0})
    assert loo.total_weight == pytest.approx(3.0)
    assert not loo.is_efficient


def test_both_engines_agree_when_contributions_are_independent():
    # Leave-one-out is not wrong in general — it is wrong when sources overlap.
    v = additive({"a": 2.0, "b": 1.0, "c": 1.0})
    loo = leave_one_out(["a", "b", "c"], v)
    exact = exact_shapley(["a", "b", "c"], v)

    for p in ("a", "b", "c"):
        assert loo.weights[p] == pytest.approx(exact.weights[p])
    assert loo.is_efficient


# --- estimator behavior --------------------------------------------------------------------


def test_sampled_shapley_converges_to_the_exact_value():
    v = redundant_pair()
    exact = exact_shapley(["a", "b", "c"], v)
    sampled = shapley(["a", "b", "c"], v, permutations=4000, seed=7)

    for p in ("a", "b", "c"):
        assert sampled.weights[p] == pytest.approx(exact.weights[p], abs=0.02)


def test_sampled_shapley_is_reproducible_for_a_given_seed():
    v = redundant_pair()
    a = shapley(["a", "b", "c"], v, permutations=16, seed=3)
    b = shapley(["a", "b", "c"], v, permutations=16, seed=3)
    c = shapley(["a", "b", "c"], v, permutations=16, seed=4)

    assert a.weights == b.weights
    assert a.weights != c.weights


def test_single_player_takes_the_whole_value():
    v = additive({"solo": 1.0})
    assert shapley(["solo"], v).weights["solo"] == pytest.approx(1.0)
    assert leave_one_out(["solo"], v).weights["solo"] == pytest.approx(1.0)


def test_empty_player_set_is_handled():
    v = fully_redundant()
    assert shapley([], v).weights == {}
    assert leave_one_out([], v).weights == {}


def test_invalid_engine_configuration_is_rejected():
    v = fully_redundant()
    with pytest.raises(ValueError, match="permutations must be at least 1"):
        shapley(["a"], v, permutations=0)
    with pytest.raises(ValueError, match="unknown engine"):
        attribute("nope", ["a"], v)
    with pytest.raises(ValueError, match="2\\^13"):
        exact_shapley([str(i) for i in range(13)], v)


def test_attribute_dispatches_and_passes_through_kwargs():
    v = redundant_pair()
    result = attribute("shapley", ["a", "b", "c"], v, permutations=8, seed=1)
    assert result.engine == "shapley"
    assert result.total_weight == pytest.approx(1.0)


def test_clamped_weights_floor_negatives_at_zero():
    result = Attribution(engine="x", weights={"a": 0.8, "b": -0.3}, grand_value=0.5)
    assert result.clamped() == {"a": 0.8, "b": 0.0}


def test_clamping_a_negative_marginal_breaks_efficiency_and_says_so():
    """The one path on which an efficient engine stops exhausting exactly ``v(N)``.

    Flooring a negative weight is right — a provider that made the answer worse is not in
    debt — but it lifts the remaining weights above the value being allocated. Paying that
    vector out means dividing it back down to fit, which is the transfer this project refuses
    everywhere else, so the excess has to be visible rather than absorbed.
    """
    result = Attribution(engine="x", weights={"a": 0.8, "b": -0.3, "c": 0.5}, grand_value=1.0)

    assert result.is_efficient  # the raw weights do exhaust v(N)
    assert result.clamped_excess == pytest.approx(0.3)
    assert sum(result.clamped().values()) == pytest.approx(1.3)  # ...the clamped ones overshoot


def test_clamped_excess_is_zero_when_every_marginal_is_positive():
    result = Attribution(engine="x", weights={"a": 0.6, "b": 0.4}, grand_value=1.0)
    assert result.clamped_excess == 0.0
    assert result.clamped() == result.weights


def test_a_negative_shapley_marginal_is_reachable_from_a_real_value_function():
    """Not a hypothetical: ``v`` is a similarity score, so it is not monotone in the players.

    A provider whose records pull the answer away from what the rest of them produce scores a
    negative marginal, and the raw weights still sum to ``v(N)`` exactly.
    """
    v = {
        frozenset(): 0.0,
        frozenset({"a"}): 1.0,
        frozenset({"b"}): 0.0,
        frozenset({"c"}): 0.5,
        frozenset({"a", "b"}): 0.5,  # b displaces what a supplied
        frozenset({"a", "c"}): 1.0,
        frozenset({"b", "c"}): 0.5,
        frozenset({"a", "b", "c"}): 1.0,
    }
    result = exact_shapley(["a", "b", "c"], lambda s: v[frozenset(s)])

    assert result.weights["b"] < 0
    assert result.is_efficient
    assert result.clamped_excess > 0


# --- similarity ------------------------------------------------------------------------------


def test_token_f1_scores_identity_and_disjointness():
    sim = TokenF1()
    assert sim("sleep hours seven", "sleep hours seven") == 1.0
    assert sim("sleep hours seven", "unrelated vocabulary entirely") == 0.0
    assert 0 < sim("sleep hours seven", "sleep hours eight") < 1


def test_token_f1_is_symmetric_and_ignores_stopwords():
    sim = TokenF1()
    assert sim("a b", "b a") == sim("b a", "a b")
    assert sim("the sleep", "sleep") == 1.0


def test_token_f1_handles_empty_strings():
    sim = TokenF1()
    assert sim("", "") == 1.0
    assert sim("", "something") == 0.0


# --- end to end through the real value function ---------------------------------------------


@dataclass(frozen=True)
class StubSource:
    """A source with its own provider, so one stub is one player."""

    id: str
    body: str
    provider_id: str = ""

    def __post_init__(self) -> None:
        # Default each stub to its own provider; tests that care about grouping set it.
        object.__setattr__(self, "provider_id", self.provider_id or self.id)

    def render(self) -> str:
        return f"[{self.id}] {self.body}"


def test_coalition_value_defines_the_empty_set_as_zero():
    sources = [StubSource("s1", "region: north"), StubSource("s2", "region: south")]
    v = CoalitionValue(question="which regions?", sources=sources, model=FakeModel())

    assert v(frozenset()) == 0.0
    assert v(frozenset({"s1", "s2"})) == pytest.approx(1.0)


def test_coalition_value_memoizes_so_each_subset_costs_one_call():
    sources = [StubSource(f"s{i}", f"field{i}: value{i}") for i in range(3)]
    v = CoalitionValue(question="what do the records show?", sources=sources, model=FakeModel())

    shapley(v.players, v, permutations=50)

    # 50 permutations x 3 players = 150 evaluations, but only 8 distinct coalitions exist.
    # The full accounting: 6 proper non-empty subsets, plus the reference answer (which the
    # grand coalition reuses), plus the no-source answer that calibrates the floor.
    assert v.calls == 8


def test_real_path_reproduces_the_redundancy_result():
    """The synthetic result, but through FakeModel and TokenF1 rather than a hand-written v.

    's1' and 's2' disclose the same fact; 's3' discloses a different one.
    """
    sources = [
        StubSource("s1", "region: north"),
        StubSource("s2", "region: north"),
        StubSource("s3", "cohort: beta"),
    ]
    v = CoalitionValue(question="what do the records show?", sources=sources, model=FakeModel())

    loo = leave_one_out(v.players, v)
    assert loo.weights["s1"] == pytest.approx(0.0)
    assert loo.weights["s2"] == pytest.approx(0.0)
    assert not loo.is_efficient

    exact = exact_shapley(v.players, v)
    assert exact.weights["s1"] == pytest.approx(exact.weights["s2"])
    assert exact.weights["s1"] > 0
    assert exact.is_efficient


def test_a_source_disclosing_nothing_earns_nothing():
    sources = [
        StubSource("s1", "region: north"),
        StubSource("s2", "cohort: beta"),
        StubSource("empty", ""),
    ]
    v = CoalitionValue(question="what do the records show?", sources=sources, model=FakeModel())

    exact = exact_shapley(v.players, v)
    assert exact.weights["empty"] == pytest.approx(0.0)
    assert exact.is_efficient
