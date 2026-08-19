"""Synthetic data for the demo and the integration tests.

All of it is invented. There are no real people here and no real measurements.

The shape is deliberate rather than arbitrary, because the demo has a point to make. It
contains, on purpose:

* a **redundancy pair** — two providers whose disclosed records are identical, so removing
  either changes nothing. This is where leave-one-out pays them both zero and Shapley splits
  the credit. Marketplaces accumulate this shape naturally as providers with overlapping data
  join, which is why it is the default demo rather than a footnote.
* a **distinctive record** no other provider duplicates, which earns the largest share.
* **hidden fields** (a participant reference and a date of birth) that must never appear in
  an answer, so the redaction path is visible in a real run rather than only in a unit test.
* a **derived field** (age) that reaches the model only as a decade band.
"""

from __future__ import annotations

from datagraph.policy import Disclosure, DisclosurePolicy
from datagraph.registry import Registry

__all__ = ["DEMO_QUESTION", "STUDY_POLICY", "seed_demo"]

#: The question the demo asks. Chosen to retrieve the redundancy pair alongside a distinctive
#: record, so the difference between Shapley and leave-one-out is visible in one run.
DEMO_QUESTION = (
    "What do the records show about sleep hours and resting heart rate in the northern region?"
)

STUDY_POLICY = DisclosurePolicy(
    levels={
        "region": Disclosure.OPEN,
        "sleep_hours": Disclosure.OPEN,
        "resting_hr": Disclosure.OPEN,
        "age": Disclosure.DERIVED,
        "recorded_on": Disclosure.DERIVED,
        "participant_ref": Disclosure.HIDDEN,
        "date_of_birth": Disclosure.HIDDEN,
    },
    buckets={"age": 10},
)

_PROVIDERS = [
    ("aurora", "Aurora Sleep Cohort"),
    ("borealis", "Borealis Health Collective"),
    ("cascade", "Cascade Wellness Panel"),
    ("delta", "Delta Metrics Group"),
]

# (record id, provider, values). The two `borealis`/`cascade` rows disclose identical content
# on purpose — that is the redundancy pair.
_RECORDS = [
    (
        "rec-01",
        "aurora",
        {
            "region": "northern",
            "sleep_hours": 5,
            "resting_hr": 78,
            "age": 34,
            "recorded_on": "2026-03-14",
            "participant_ref": "synthetic-participant-0001",
            "date_of_birth": "1992-02-29",
        },
    ),
    (
        "rec-02",
        "borealis",
        {
            "region": "northern",
            "sleep_hours": 8,
            "resting_hr": 55,
            "age": 41,
            "recorded_on": "2026-03-14",
            "participant_ref": "synthetic-participant-0002",
            "date_of_birth": "1985-07-11",
        },
    ),
    (
        "rec-03",
        "cascade",
        {
            "region": "northern",
            "sleep_hours": 8,
            "resting_hr": 55,
            "age": 44,
            "recorded_on": "2026-03-14",
            "participant_ref": "synthetic-participant-0003",
            "date_of_birth": "1982-11-30",
        },
    ),
    (
        "rec-04",
        "delta",
        {
            "region": "northern",
            "sleep_hours": 7,
            "resting_hr": 64,
            "age": 29,
            "recorded_on": "2026-03-15",
            "participant_ref": "synthetic-participant-0004",
            "date_of_birth": "1997-05-02",
        },
    ),
    (
        "rec-05",
        "delta",
        {
            "region": "southern",
            "sleep_hours": 6,
            "resting_hr": 71,
            "age": 52,
            "recorded_on": "2026-03-15",
            "participant_ref": "synthetic-participant-0005",
            "date_of_birth": "1974-01-19",
        },
    ),
]


def seed_demo(registry: Registry) -> Registry:
    """Populate ``registry`` with the synthetic study. Returns it for chaining."""
    for provider_id, name in _PROVIDERS:
        registry.add_provider(provider_id, name)
        registry.add_dataset(
            f"{provider_id}-study", provider_id, f"{name} sleep study", STUDY_POLICY
        )

    for record_id, provider_id, values in _RECORDS:
        registry.add_record(record_id, f"{provider_id}-study", values)

    return registry
