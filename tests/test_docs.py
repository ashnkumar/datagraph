"""Pins the claims the README makes about this code, so they cannot rot silently.

A README breaks every time the code changes underneath it and nothing fails. Every claim that
is cheap to check mechanically and expensive to notice going stale is one assertion here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from datagraph.attribution import CoalitionValue, exact_shapley, leave_one_out, shapley
from datagraph.ledger import Ledger
from datagraph.marketplace import DEFAULT_COHORT_FLOOR, DEFAULT_MAX_SOURCES, Marketplace
from datagraph.models import FakeModel
from datagraph.money import allocate
from datagraph.registry import Registry
from datagraph.sample_data import seed_demo

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SPEC = (ROOT / "SPEC.md").read_text()
QUESTION = (
    "What do the records show about sleep hours and resting heart rate in the northern region?"
)
PAYMENT = 1000


def _payouts(engine: str) -> dict[str, int]:
    registry = Registry(":memory:")
    seed_demo(registry)
    market = Marketplace(registry=registry, ledger=Ledger(), model=FakeModel(), engine=engine)
    market.fund_researcher("r1", PAYMENT)
    return dict(market.query("r1", QUESTION, PAYMENT).payouts)


def test_readme_states_the_current_length_of_the_file_it_says_to_read_first():
    claimed = int(
        re.search(r"attribution\.py`\. It holds both engines and it is (\d+) lines", README).group(
            1
        )
    )
    actual = len((ROOT / "src" / "datagraph" / "attribution.py").read_text().splitlines())
    assert claimed == actual


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        ("shapley", {"aurora": 243, "borealis": 195, "cascade": 203, "delta": 359}),
        ("exact_shapley", {"aurora": 241, "borealis": 197, "cascade": 197, "delta": 365}),
        ("leave_one_out", {"aurora": 287, "borealis": 0, "cascade": 0, "delta": 713}),
    ],
)
def test_the_numbers_quoted_in_the_docs_are_what_the_engines_produce(engine, expected):
    assert _payouts(engine) == expected


def test_readme_quotes_the_redundant_pair_correctly():
    """The 'two providers hold the same fact' row: 195 and 203 sampled, 197 each exactly."""
    assert "`195` and `203` of 1000" in README
    assert "`197` each computed exactly" in README
    sampled, exact = _payouts("shapley"), _payouts("exact_shapley")
    assert (sampled["borealis"], sampled["cascade"]) == (195, 203)
    assert exact["borealis"] == exact["cascade"] == 197


def test_leave_one_out_weights_do_not_sum_to_the_whole():
    """The 0.2744 in the hero caption, and the reason the project exists."""
    assert "0.2744" in README
    registry = Registry(":memory:")
    seed_demo(registry)
    sources = registry.search(QUESTION, limit=DEFAULT_MAX_SOURCES, max_per_provider=2)
    value = CoalitionValue(model=FakeModel(), question=QUESTION, sources=sources)
    loo = leave_one_out(value.players, value)
    assert round(loo.total_weight, 4) == 0.2744
    assert not loo.is_efficient
    for engine in (shapley(value.players, value), exact_shapley(value.players, value)):
        assert engine.is_efficient


@dataclass(frozen=True)
class _Rec:
    id: str
    provider_id: str
    body: str

    def render(self) -> str:
        return f"[{self.id}] {self.body}"


def _base_records() -> list[_Rec]:
    registry = Registry(":memory:")
    seed_demo(registry)
    views = registry.search(QUESTION, limit=DEFAULT_MAX_SOURCES, max_per_provider=2)
    return [_Rec(v.id, v.provider_id, v.render().partition("] ")[2]) for v in views]


def _split(weights: dict[str, float]) -> dict[str, int]:
    keys = sorted(weights)
    return dict(zip(keys, allocate(PAYMENT, [weights[k] for k in keys]), strict=True))


@pytest.mark.parametrize("copies", [0, 3, 10])
def test_payout_is_invariant_to_row_count(copies):
    """The README's 'padding your dataset' row: 10 extra copies leave delta on 365."""
    assert "leave it on `365` credits" in README
    records = list(_base_records())
    owned = [r for r in records if r.provider_id == "delta"]
    for k in range(copies):
        records += [replace(r, id=f"{r.id}-c{k}") for r in owned]
    value = CoalitionValue(model=FakeModel(), question=QUESTION, sources=records)
    assert _split(exact_shapley(value.players, value).clamped())["delta"] == 365


def test_registering_twice_pays_what_records_as_players_paid():
    """Both figures in the Limitations bullet, and the reason they are the same number."""
    assert "from `365` to `446`" in README
    records = _base_records()
    owned = [r.id for r in records if r.provider_id == "delta"]
    assert len(owned) == 2

    split_records = [
        replace(r, provider_id=f"delta-{owned.index(r.id) + 1}") if r.provider_id == "delta" else r
        for r in records
    ]
    value = CoalitionValue(model=FakeModel(), question=QUESTION, sources=split_records)
    payouts = _split(exact_shapley(value.players, value).clamped())
    assert sum(v for k, v in payouts.items() if k.startswith("delta")) == 446


def test_retrieval_caps_slots_per_provider_at_two():
    """`max(1, 6 // 3)` — the cap is tied to the floor, so a full result set can satisfy it."""
    assert DEFAULT_MAX_SOURCES == 6
    assert DEFAULT_COHORT_FLOOR == 3
    assert max(1, DEFAULT_MAX_SOURCES // DEFAULT_COHORT_FLOOR) == 2


def test_sampling_and_enumerating_cost_the_same_number_of_calls():
    """The README claims 16 model calls either way on the demo query."""
    assert "the same 16 model calls" in README
    for engine in ("shapley", "exact_shapley"):
        registry = Registry(":memory:")
        seed_demo(registry)
        market = Marketplace(registry=registry, ledger=Ledger(), model=FakeModel(), engine=engine)
        market.fund_researcher("r1", PAYMENT)
        assert market.query("r1", QUESTION, PAYMENT).model_calls == 16


def test_leave_one_out_costs_what_the_documents_say_it_costs():
    """Its cost is one of the four lessons, and it drifted: three artifacts, three numbers.

    ``leave_one_out`` makes ``n + 1`` calls to ``v``, which is not the same as ``n + 1`` model
    calls — ``CoalitionValue`` also generates a no-source answer for the floor. Six on the demo,
    not four and not five.
    """
    assert "that is 6 calls against 16" in README

    registry = Registry(":memory:")
    seed_demo(registry)
    market = Marketplace(
        registry=registry, ledger=Ledger(), model=FakeModel(), engine="leave_one_out"
    )
    market.fund_researcher("r1", PAYMENT)
    result = market.query("r1", QUESTION, PAYMENT)

    players = len(result.attribution.weights)
    assert players == 4
    assert result.model_calls == players + 2 == 6


BRITISH = re.compile(
    r"behaviour|colour|favour|organis|recognis|centre|whilst|amongst|learnt|cancelled"
    r"|labelled|catalogue|defence|offence|practis|programme|sceptic|utilis|initialis"
    r"|afterwards|normalis|decentralis|memois|judgement",
    re.IGNORECASE,
)


@pytest.mark.parametrize(("name", "text"), [("README.md", README), ("SPEC.md", SPEC)])
def test_documents_use_american_spellings(name, text):
    assert not BRITISH.findall(text), f"{name}: {set(BRITISH.findall(text))}"


def test_source_and_tests_use_american_spellings():
    here = Path(__file__).resolve()  # this file holds the pattern itself
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py")):
        if path.resolve() == here:
            continue
        found = BRITISH.findall(path.read_text())
        assert not found, f"{path.relative_to(ROOT)}: {set(found)}"


def test_thinking_stays_enabled_and_cost_is_controlled_with_effort():
    """Anthropic's documented mitigation: disabling thinking is what leaks internal XML tags,
    and a leaked tag is spurious tokens in the similarity score, so a wrong payout."""
    source = (ROOT / "src" / "datagraph" / "models.py").read_text()
    assert '"thinking": {"type": "adaptive", "display": "omitted"}' in source
    assert '"disabled"' not in source
    # And no instruction telling the model not to reason, which increases tag leakage.
    from datagraph.models import SYSTEM_PROMPT

    assert not re.search(r"do not (think|reason)", SYSTEM_PROMPT, re.IGNORECASE)
    assert "Do not include internal or system XML tags in your response." in SYSTEM_PROMPT
