import pytest

from datagraph.policy import (
    CohortTooSmall,
    Disclosure,
    DisclosurePolicy,
    enforce_cohort_floor,
)

POLICY = DisclosurePolicy(
    levels={
        "region": Disclosure.OPEN,
        "sleep_hours": Disclosure.OPEN,
        "age": Disclosure.DERIVED,
        "recorded_on": Disclosure.DERIVED,
        "name": Disclosure.HIDDEN,
    },
    buckets={"age": 10},
)


def test_open_fields_pass_through_verbatim():
    out = POLICY.redact({"region": "north", "sleep_hours": 7.5})
    assert out == {"region": "north", "sleep_hours": 7.5}


def test_hidden_fields_are_absent_not_nulled():
    out = POLICY.redact({"region": "north", "name": "Wren Ashby"})
    assert "name" not in out
    assert out == {"region": "north"}


def test_hidden_value_appears_nowhere_in_the_serialised_projection():
    # The structural guarantee: a suppressed value cannot reach a prompt because it is not in
    # the object the prompt builder receives.
    out = POLICY.redact({"name": "Wren Ashby", "age": 34})
    assert "Wren" not in repr(out)


def test_numeric_derived_fields_are_banded():
    assert POLICY.redact({"age": 34}) == {"age": "30-39"}
    assert POLICY.redact({"age": 30}) == {"age": "30-39"}
    assert POLICY.redact({"age": 29}) == {"age": "20-29"}


def test_date_derived_fields_truncate_to_the_month():
    assert POLICY.redact({"recorded_on": "2026-03-17"}) == {"recorded_on": "2026-03"}


def test_unknown_fields_default_to_hidden():
    # Fail closed: adding a column to a dataset must not publish it by accident.
    assert POLICY.redact({"unlisted_column": "secret"}) == {}


def test_default_level_is_configurable_but_defaults_to_hidden():
    assert DisclosurePolicy().default is Disclosure.HIDDEN
    permissive = DisclosurePolicy(default=Disclosure.OPEN)
    assert permissive.redact({"anything": 1}) == {"anything": 1}


def test_uncoarsenable_derived_value_is_dropped_rather_than_leaked():
    policy = DisclosurePolicy(levels={"notes": Disclosure.DERIVED})
    assert policy.redact({"notes": "free text with no coarse form"}) == {}


def test_booleans_are_not_treated_as_numbers():
    # bool subclasses int; banding it would expose the value verbatim as "1-10".
    policy = DisclosurePolicy(levels={"flag": Disclosure.DERIVED})
    assert policy.redact({"flag": True}) == {}


def test_disclosed_fields_excludes_hidden():
    assert POLICY.disclosed_fields() == {"region", "sleep_hours", "age", "recorded_on"}


def test_cohort_floor_allows_a_wide_enough_cohort():
    enforce_cohort_floor(["p1", "p2", "p3"], floor=3)


def test_cohort_floor_refuses_a_narrow_cohort():
    with pytest.raises(CohortTooSmall) as excinfo:
        enforce_cohort_floor(["p1", "p2"], floor=3)
    assert excinfo.value.found == 2
    assert excinfo.value.required == 3


def test_cohort_floor_counts_distinct_providers_not_records():
    # Ten records from one provider is still one person.
    with pytest.raises(CohortTooSmall):
        enforce_cohort_floor(["p1"] * 10, floor=3)


def test_empty_result_set_fails_the_cohort_floor():
    with pytest.raises(CohortTooSmall):
        enforce_cohort_floor([], floor=3)
