"""The one test that exercises the real main path against the real API.

Deselected by default — the whole point of :class:`~datagraph.models.FakeModel` is that the
rest of the suite runs offline with no key. Run this one deliberately:

    uv run pytest -m live

It costs up to 2^n generations for one query, which with the default source cap is 64.
"""

import os

import pytest

from datagraph.ledger import Ledger
from datagraph.marketplace import Marketplace
from datagraph.models import AnthropicModel
from datagraph.registry import Registry
from datagraph.sample_data import DEMO_QUESTION, seed_demo

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY is not set",
    ),
]


def test_a_real_query_settles_correctly_end_to_end():
    registry = seed_demo(Registry())
    market = Marketplace(registry, Ledger(), AnthropicModel(), max_sources=4)
    market.fund_researcher("rowan", 10_000)

    result = market.query("rowan", DEMO_QUESTION, 1_000)

    assert not result.refunded, result.refund_reason
    assert result.answer.strip()

    # The properties that have to hold whatever the model says.
    assert result.total_paid == 1_000
    assert result.attribution is not None
    assert result.attribution.is_efficient
    market.ledger.check_invariants()

    # Redaction is structural, so it holds for a real generation too.
    assert "synthetic-participant" not in result.answer
    assert "1992-02-29" not in result.answer

    registry.close()
