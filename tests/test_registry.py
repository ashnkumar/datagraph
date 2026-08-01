import pytest

from datagraph.policy import Disclosure, DisclosurePolicy
from datagraph.registry import Registry, RegistryError, provider_ids, render_sources

POLICY = DisclosurePolicy(
    levels={
        "region": Disclosure.OPEN,
        "sleep_hours": Disclosure.OPEN,
        "age": Disclosure.DERIVED,
        "participant": Disclosure.HIDDEN,
    },
    buckets={"age": 10},
)


@pytest.fixture
def registry():
    with Registry() as reg:
        reg.add_provider("p1", "Northern Cohort")
        reg.add_dataset("d1", "p1", "sleep study", POLICY)
        yield reg


def test_records_carry_both_raw_and_disclosed_values(registry):
    record = registry.add_record(
        "r1", "d1", {"region": "north", "sleep_hours": 7, "age": 34, "participant": "synthetic-01"}
    )

    assert record.values["participant"] == "synthetic-01"
    assert "participant" not in record.disclosed
    assert record.disclosed["age"] == "30-39"
    assert record.provider_id == "p1"


def test_rendered_record_contains_no_hidden_value(registry):
    record = registry.add_record(
        "r1", "d1", {"region": "north", "age": 34, "participant": "synthetic-01"}
    )
    rendered = record.render()

    assert "synthetic-01" not in rendered
    assert "north" in rendered
    assert "30-39" in rendered


def test_records_round_trip_through_sqlite(registry):
    registry.add_record("r1", "d1", {"region": "north", "sleep_hours": 7, "participant": "x"})

    reloaded = registry.all_records()
    assert len(reloaded) == 1
    assert reloaded[0].disclosed == {"region": "north", "sleep_hours": 7}
    assert reloaded[0].values["participant"] == "x"


def test_policy_round_trips_through_sqlite(registry):
    dataset = registry.get_dataset("d1")
    assert dataset is not None
    assert dataset.policy.levels["age"] is Disclosure.DERIVED
    assert dataset.policy.buckets["age"] == 10
    assert dataset.policy.default is Disclosure.HIDDEN


def test_unknown_parents_are_rejected(registry):
    with pytest.raises(RegistryError, match="unknown provider"):
        registry.add_dataset("d2", "nope", "x", POLICY)
    with pytest.raises(RegistryError, match="unknown dataset"):
        registry.add_record("r9", "nope", {"region": "north"})


def test_search_matches_disclosed_content_and_is_bounded(registry):
    registry.add_record("r1", "d1", {"region": "north", "sleep_hours": 7})
    registry.add_record("r2", "d1", {"region": "south", "sleep_hours": 6})
    registry.add_record("r3", "d1", {"region": "north", "sleep_hours": 8})

    hits = registry.search("what is sleep like in the north region?", limit=2)

    assert len(hits) == 2
    assert {r.id for r in hits} == {"r1", "r3"}


def test_search_cannot_match_on_a_hidden_value(registry):
    # Searching for a suppressed value must not surface the record that holds it.
    registry.add_record("r1", "d1", {"region": "north", "participant": "kestrel"})
    assert registry.search("kestrel") == []


def test_search_is_stable_for_a_given_store(registry):
    for i in range(6):
        registry.add_record(f"r{i}", "d1", {"region": "north", "sleep_hours": i})

    assert [r.id for r in registry.search("north", limit=3)] == [
        r.id for r in registry.search("north", limit=3)
    ]


def test_search_with_no_content_words_returns_nothing(registry):
    registry.add_record("r1", "d1", {"region": "north"})
    assert registry.search("the and of") == []


def test_a_rejected_record_leaves_the_store_readable(registry):
    """A bad insert must fail *closed*, not poison every subsequent read.

    Committing before projecting meant one non-finite value persisted a row that then raised
    during redaction — so `all_records()` and `search()` failed forever after, from a single
    bad insert. Validation now happens first.
    """
    registry.add_record("good", "d1", {"region": "north"})

    with pytest.raises(RegistryError, match="not serialisable"):
        registry.add_record("bad", "d1", {"age": float("nan")})

    assert [r.id for r in registry.all_records()] == ["good"]
    assert [r.id for r in registry.search("north")] == ["good"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(registry, bad):
    with pytest.raises(RegistryError):
        registry.add_record("bad", "d1", {"age": bad})
    assert registry.all_records() == []


def test_search_suppresses_a_providers_duplicate_records(registry):
    """Identical disclosed content from one provider is one fact, not several."""
    for i in range(4):
        registry.add_record(f"dupe-{i}", "d1", {"region": "north", "sleep_hours": 7})

    hits = registry.search("north region", limit=6)
    assert len(hits) == 1


def test_search_caps_how_many_slots_one_provider_can_take(registry):
    registry.add_provider("p2", "Other Cohort")
    registry.add_dataset("d2", "p2", "other", POLICY)

    # One provider pads its dataset with distinct-but-similar rows.
    for i in range(10):
        registry.add_record(f"pad-{i}", "d1", {"region": "north", "sleep_hours": i})
    registry.add_record("other", "d2", {"region": "north", "sleep_hours": 99})

    hits = registry.search("north region", limit=6, max_per_provider=2)

    assert sum(1 for r in hits if r.provider_id == "p1") == 2
    assert any(r.provider_id == "p2" for r in hits)


def test_helpers(registry):
    registry.add_record("r2", "d1", {"region": "south"})
    registry.add_record("r1", "d1", {"region": "north"})
    records = registry.all_records()

    assert provider_ids(records) == ["p1", "p1"]
    # Rendering is ordered by record id so prompts are stable across runs.
    assert render_sources(records).splitlines()[0].startswith("[r1]")
