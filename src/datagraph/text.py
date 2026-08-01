"""Shared text handling.

Both retrieval and the default similarity measure need the same notion of "content word", so
it lives in one place rather than drifting apart in two.
"""

from __future__ import annotations

import re

__all__ = ["STOPWORDS", "tokenize"]

_WORD = re.compile(r"[a-z0-9]+")

#: Words carrying no topical signal. Short and deliberately unclever — a bigger list would be
#: guesswork, and the similarity measure is a documented proxy either way.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-case content words, in order, with stopwords removed."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
