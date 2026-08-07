"""Answer generation: the Anthropic client, and a deterministic stand-in for tests.

Attribution works by regenerating the answer from many different subsets of the retrieved
records, so this interface is called a lot — up to 2^n times per query before memoization.
Everything here is shaped by that: short answers and low effort.

**A note on determinism.** There is no way to ask for it. Setting ``temperature``, ``top_p``
or ``top_k`` to any non-default value returns a 400 on this model, there is no ``seed``
parameter, and Anthropic's migration guide notes that ``temperature = 0`` "never guaranteed
identical outputs on prior models" either. So two calls with identical inputs may return
different text, and that is a real source of noise in a measurement built on comparing
regenerated answers.

Two things bound it instead of pretending it away: coalition values are memoized, so each
distinct subset is generated exactly once per query and every comparison within a query is
against a fixed set of strings; and :class:`FakeModel` is exactly deterministic, so the test
suite measures the attribution mathematics rather than model variance.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
    "AnthropicModel",
    "FakeModel",
    "ModelClient",
    "ModelRefusal",
    "Source",
]

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You answer questions using only the records provided in the user message.

Answer in at most three sentences. State what the records show. If the records do not support \
an answer, say so plainly rather than reasoning from general knowledge — an answer that does \
not come from the records is worse than no answer here, because the whole system rests on \
knowing which record produced which claim.

Do not include internal or system XML tags in your response.\
"""
# That last sentence is the third clause of Anthropic's documented mitigation for running with
# thinking disabled, kept because disabling is one line away. The first two clauses are about
# tool calls and this workload has no tools, so they are omitted. It is deliberately generic:
# "Instructions that call out thinking tags by name are less effective than the general form."
# Note also what is *absent* — no instruction telling the model not to think or not to reason,
# because per the same page "that kind of instruction increases tag leakage."

# Defense in depth behind the instruction above and the adaptive-thinking default. A leaked
# internal tag would be spurious tokens in the similarity measure, and would therefore skew
# every provider's share in the query — the failure is a wrong payout, not an ugly answer.
_INTERNAL_TAG = re.compile(r"</?(?:thinking|internal|scratchpad)[^>]*>", re.IGNORECASE)


@runtime_checkable
class Source(Protocol):
    """Anything renderable into a prompt, with an identity and an owner.

    ``provider_id`` is part of the protocol because attribution groups by it: the players in
    the cooperative game are the parties who get paid, not the individual rows.
    """

    @property
    def id(self) -> str: ...

    @property
    def provider_id(self) -> str: ...

    def render(self) -> str: ...


class ModelClient(Protocol):
    """Produces an answer to ``question`` grounded in ``sources``."""

    def answer(self, question: str, sources: Sequence[Source]) -> str: ...


class ModelRefusal(Exception):
    """Raised when the model declined to answer.

    Current Claude models return a successful response with ``stop_reason == "refusal"``
    rather than an HTTP error, so this has to be checked explicitly before reading content.
    """

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"model declined to answer (category={category!r}): {explanation}")
        self.category = category
        self.explanation = explanation


def build_prompt(question: str, sources: Sequence[Source]) -> str:
    """Assemble the user message. Sources are ordered by id so the prompt is stable."""
    if not sources:
        return f"No records are available.\n\nQuestion: {question}"

    body = "\n".join(s.render() for s in sorted(sources, key=lambda s: s.id))
    return f"Records:\n{body}\n\nQuestion: {question}"


class FakeModel:
    """A deterministic stand-in that composes an answer from the facts in its sources.

    This is not a mock that bypasses the logic under test. It produces the union of the
    distinct facts its sources disclose, which means removing a source changes the answer
    exactly when that source supplied a fact nothing else supplies. That is precisely the
    structure attribution has to measure, so the offline suite exercises the real engines with
    exactly known ground truth.
    """

    def answer(self, question: str, sources: Sequence[Source]) -> str:
        facts = sorted({fact for s in sources for fact in _facts(s)})
        if not facts:
            return "The records do not support an answer."
        return "The records show " + "; ".join(facts) + "."


def _facts(source: Source) -> list[str]:
    """Extract comparable fact strings from a rendered source."""
    rendered = source.render()
    # Drop the "[id] " prefix so identity does not itself count as content — otherwise every
    # source would look unique and redundancy could never be detected. Test `sep`, not `body`:
    # a source that discloses nothing has an empty body, and falling back to the raw string
    # there would smuggle its id in as a fact.
    _, sep, body = rendered.partition("] ")
    text = body if sep else rendered
    return [part.strip() for part in text.split(",") if part.strip()]


class AnthropicModel:
    """Generates answers with the Anthropic API.

    Args:
        client: An ``anthropic.Anthropic`` instance. Constructed from the environment if
            omitted.
        model: Model id. Defaults to :data:`DEFAULT_MODEL`.
        effort: Reasoning effort. ``"low"`` suits this workload — the task is extraction from
            a short record list, and the call is repeated many times per query. This is the
            cost lever, in place of disabling thinking.
        max_tokens: Output cap. Answers are capped at three sentences by the system prompt,
            but this has to leave room for thinking too: ``max_tokens`` is a hard cap on
            "total output for the request, thinking and response text combined."
        use_fallbacks: Opt into server-side refusal fallbacks, so a request declined by a
            safety classifier is re-run on the model Anthropic recommends for that refusal
            category, inside the same call. Enabled by default. The feature is in beta and is
            unavailable on the Batches API and on the cloud-provider platforms, so if the
            request is rejected for it, the first call drops it and carries on unprotected
            rather than failing the whole query.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        effort: str = "low",
        max_tokens: int = 2048,
        use_fallbacks: bool = True,
    ) -> None:
        if client is None:
            import anthropic  # imported lazily so offline use needs no API key

            client = anthropic.Anthropic()

        self._client = client
        self._model = model or os.environ.get("DATAGRAPH_MODEL", DEFAULT_MODEL)
        self._effort = effort
        self._max_tokens = max_tokens
        self._use_fallbacks = use_fallbacks

    def answer(self, question: str, sources: Sequence[Source]) -> str:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            # Thinking stays on and cost is controlled with effort instead, which is what
            # Anthropic recommends: "for most tasks, thinking enabled at `low` effort performs
            # better than thinking disabled at similar cost". Disabling it is what makes the
            # model "occasionally emit tool calls as plain text or include internal XML tags
            # in its visible output" — and the visible output is the measurement here, so a
            # leaked tag is spurious tokens in a similarity score and a wrong payout.
            # `display: omitted` is already the default on this model; it is set explicitly
            # because it is load-bearing: thinking text must never reach the scored string.
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": self._effort},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(question, sources)}],
        }

        response = self._create(request)

        # Must be checked before reading content: a refusal is a successful HTTP response
        # whose content is empty or partial.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise ModelRefusal(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
            )

        text = "".join(b.text for b in response.content if b.type == "text")
        return _INTERNAL_TAG.sub("", text).strip()

    def _create(self, request: dict[str, Any]) -> Any:
        if not self._use_fallbacks:
            return self._client.messages.create(**request)

        try:
            return self._client.beta.messages.create(
                **request,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:
            # The beta is not available everywhere the API is. If this deployment rejects it,
            # drop it once and carry on unprotected rather than failing the whole query.
            if "fallback" not in str(exc).lower():
                raise
            self._use_fallbacks = False
            return self._client.messages.create(**request)
