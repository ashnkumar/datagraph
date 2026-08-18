"""Offline tests for the Anthropic client wrapper.

The response-handling rules here are all about API behavior — a refusal arriving as a 200, a
fallback quietly changing which model served the request — and none of them were reachable
from the offline suite before, because the only test that built an :class:`AnthropicModel` was
the live one. A stub client makes them testable without a key, which matters: these are exactly
the paths that decide whether a payout is meaningful.
"""

from dataclasses import dataclass, field

import pytest

from datagraph.models import DEFAULT_MODEL, AnthropicModel, ModelRefusal, ModelSubstituted


@dataclass
class Block:
    text: str
    type: str = "text"


@dataclass
class StubResponse:
    content: list[Block]
    model: str = DEFAULT_MODEL
    stop_reason: str = "end_turn"
    stop_details: object | None = None


@dataclass
class StubMessages:
    response: StubResponse
    calls: list[dict] = field(default_factory=list)

    def create(self, **request):
        self.calls.append(request)
        return self.response


@dataclass
class StubClient:
    """Shaped like the bits of the SDK surface :class:`AnthropicModel` actually touches."""

    response: StubResponse

    def __post_init__(self):
        self.messages = StubMessages(self.response)
        self.beta = type("Beta", (), {"messages": StubMessages(self.response)})()


def test_a_plain_answer_comes_back_stripped():
    client = StubClient(StubResponse(content=[Block("  The records show a link.  ")]))
    model = AnthropicModel(client=client)

    assert model.answer("q", []) == "The records show a link."


def test_a_refusal_raises_rather_than_returning_empty_text():
    """A refusal is a 200 whose content may be empty, so it has to be checked before reading."""

    @dataclass
    class Details:
        category: str = "bio"
        explanation: str = "declined"

    client = StubClient(StubResponse(content=[], stop_reason="refusal", stop_details=Details()))
    model = AnthropicModel(client=client)

    with pytest.raises(ModelRefusal) as exc:
        model.answer("q", [])
    assert exc.value.category == "bio"


def test_a_coalition_served_by_a_different_model_stops_the_query():
    """The confound server-side fallbacks introduce, and the reason the check exists.

    A fallback re-runs a declined request on another model and returns an ordinary 200. Nothing
    in the answer says so; only ``response.model`` does. Scoring one coalition on a different
    model than the others means the difference between them is no longer just the records, and
    the query cannot say how much of a provider's share was the swap.
    """
    client = StubClient(
        StubResponse(content=[Block("An answer from somewhere else.")], model="claude-haiku-4-5")
    )
    model = AnthropicModel(client=client)

    with pytest.raises(ModelSubstituted) as exc:
        model.answer("q", [])
    assert exc.value.served == "claude-haiku-4-5"
    assert exc.value.requested == DEFAULT_MODEL


def test_the_requested_model_serving_the_request_is_not_a_substitution():
    client = StubClient(StubResponse(content=[Block("Fine.")], model=DEFAULT_MODEL))
    assert AnthropicModel(client=client).answer("q", []) == "Fine."


def test_thinking_stays_on_and_effort_carries_the_cost_control():
    """Disabling thinking is what leaks internal tags, and the leak lands in the scored string."""
    client = StubClient(StubResponse(content=[Block("Fine.")]))
    model = AnthropicModel(client=client, effort="low")
    model.answer("q", [])

    request = client.beta.messages.calls[0]
    assert request["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert request["output_config"] == {"effort": "low"}
    # The parameters that would return a 400 on this model must not be sent at all.
    assert "temperature" not in request
    assert "top_p" not in request
    assert "top_k" not in request
