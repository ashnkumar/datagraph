"""Answer generation: the Anthropic client, and a deterministic stand-in for tests.

Attribution works by regenerating the answer from many different subsets of the retrieved
records, so this interface is called a lot — up to 2^n times per query before memoisation.
Everything here is shaped by that: short answers, low effort, no thinking.

**A note on determinism.** ``temperature`` is not accepted on current Claude models, so two
calls with identical inputs may return different text. That is a real source of noise in a
measurement built on comparing regenerated answers. Two things keep it bounded: coalition
values are memoised, so each distinct subset is generated exactly once per query and the
comparison within a query is self-consistent; and :class:`FakeModel` is exactly deterministic,
so the test suite measures the attribution mathematics rather than model variance.
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

# Defence in depth for the tag instruction above: leaked internal tags would show up as
# spurious tokens in the similarity measure and skew every influence score in the query.
_INTERNAL_TAG = re.compile(r"</?(?:thinking|internal|scratchpad)[^>]*>", re.IGNORECASE)


@runtime_checkable
class Source(Protocol):
    """Anything renderable into a prompt with a stable identity."""

    @property
    def id(self) -> str: ...

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
            a short record list, and the call is repeated many times per query.
        max_tokens: Output cap. Answers are capped at three sentences by the system prompt.
        use_fallbacks: Opt into server-side refusal fallbacks, so a declined request is
            re-run on another model inside the same call. Enabled by default. If the beta is
            not available to your organisation the first call falls back to a plain request
            and logs nothing further.
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
            # Thinking off, low effort: this is short extractive work repeated many times.
            # Disabling thinking is permitted at effort "high" or below.
            "thinking": {"type": "disabled"},
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
            # The fallbacks beta is gated per organisation. If it is unavailable, drop it once
            # and carry on unprotected rather than failing the whole query.
            if "fallback" not in str(exc).lower():
                raise
            self._use_fallbacks = False
            return self._client.messages.create(**request)
