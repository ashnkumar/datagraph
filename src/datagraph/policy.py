"""Disclosure policies: what a provider is willing to expose, and how.

Redaction happens *before* records are assembled into a prompt, so a ``HIDDEN`` field is
absent from the object the prompt builder ever sees. That is a structural property of the data
path, not an instruction to the model — the prompt cannot contain what was never put in it.

**This is enforced by the application, not by cryptography.** An operator with database access
reads raw records, and a compromised process bypasses redaction entirely. Real guarantees would
require the raw values never to reach this tier: trusted execution, secure aggregation, or
local differential privacy on the provider's side. See the README for the same caveat stated
where a user will actually read it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

__all__ = [
    "DEFAULT_BUCKET",
    "DEFAULT_COHORT_FLOOR",
    "CohortTooSmall",
    "Disclosure",
    "DisclosurePolicy",
    "enforce_cohort_floor",
]

#: Minimum number of distinct providers behind an answer. Below this the query is refused.
DEFAULT_COHORT_FLOOR = 3

#: Band width used to coarsen a numeric ``DERIVED`` field when the policy doesn't specify one.
DEFAULT_BUCKET = 10

_ISO_DATE = re.compile(r"^(\d{4}-\d{2})-\d{2}")


class Disclosure(StrEnum):
    """How much of a field may leave the provider's record."""

    OPEN = "open"
    """Exposed verbatim."""

    DERIVED = "derived"
    """Exposed only coarsened — numbers into bands, dates truncated to the month."""

    HIDDEN = "hidden"
    """Never exposed under any query."""


class CohortTooSmall(Exception):
    """Raised when a query's result set spans too few providers to answer safely."""

    def __init__(self, found: int, required: int) -> None:
        super().__init__(
            f"query matched {found} distinct provider(s), which is below the cohort floor of "
            f"{required}; refusing to answer"
        )
        self.found = found
        self.required = required


@dataclass(frozen=True)
class DisclosurePolicy:
    """Per-field disclosure rules for a dataset.

    Args:
        levels: Field name -> disclosure level.
        buckets: Band width for numeric ``DERIVED`` fields, by field name. Defaults to
            :data:`DEFAULT_BUCKET`.
        default: Level applied to a field with no explicit rule. Defaults to ``HIDDEN`` —
            the policy fails closed, so adding a column to a dataset cannot accidentally
            publish it.
    """

    levels: Mapping[str, Disclosure] = field(default_factory=dict)
    buckets: Mapping[str, int] = field(default_factory=dict)
    default: Disclosure = Disclosure.HIDDEN

    def level_for(self, name: str) -> Disclosure:
        return self.levels.get(name, self.default)

    def redact(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Project ``values`` down to what the policy permits.

        ``HIDDEN`` fields are omitted from the result entirely rather than nulled, so a
        downstream consumer cannot distinguish "suppressed" from "absent" and cannot
        accidentally serialise a placeholder that hints at the real value.
        """
        out: dict[str, Any] = {}
        for name, value in values.items():
            level = self.level_for(name)
            if level is Disclosure.OPEN:
                out[name] = value
            elif level is Disclosure.DERIVED:
                coarsened = _coarsen(value, self.buckets.get(name, DEFAULT_BUCKET))
                # A value we cannot coarsen is treated as HIDDEN. Failing closed here matters:
                # the alternative is leaking a value verbatim through the "safer" level.
                if coarsened is not None:
                    out[name] = coarsened
        return out

    def disclosed_fields(self) -> set[str]:
        """Fields this policy can expose at all, in some form."""
        return {n for n, lvl in self.levels.items() if lvl is not Disclosure.HIDDEN}


def _coarsen(value: Any, bucket: int) -> str | None:
    """Coarsen one value, or return ``None`` if it has no safe coarse form."""
    if isinstance(value, bool):
        # bool is an int subclass; banding it would be nonsense and exposes it verbatim.
        return None

    if isinstance(value, int | float):
        if bucket <= 0:
            return None
        # NaN and the infinities have no meaningful band. Returning None drops the field,
        # which is the fail-closed answer; raising here would let one malformed value poison
        # every later read of the record that holds it.
        if isinstance(value, float) and not isfinite(value):
            return None
        low = int(value // bucket) * bucket
        return f"{low}-{low + bucket - 1}"

    if isinstance(value, str) and (m := _ISO_DATE.match(value)):
        return m.group(1)

    return None


def enforce_cohort_floor(provider_ids: Iterable[str], floor: int = DEFAULT_COHORT_FLOOR) -> None:
    """Refuse a query whose results come from fewer than ``floor`` distinct provider accounts.

    Without this, a researcher narrows a query until one provider's records are the whole answer
    and reads them straight out of it. The check runs *before* generation, so a refused query
    never reaches the model and never spends a credit.

    **This is a source-diversity floor, not k-anonymity.** It counts provider ids, which is the
    only identity in the schema; it has no notion of a data subject and no way to acquire one.
    Three accounts held by one operator satisfy it, and so do three providers whose records all
    describe the same person. It raises the cost of narrowing a query to a single source — it
    does not guarantee that three people stand behind an answer.

    Raises:
        CohortTooSmall: If the result set spans fewer than ``floor`` providers.
    """
    distinct = len(set(provider_ids))
    if distinct < floor:
        raise CohortTooSmall(found=distinct, required=floor)
