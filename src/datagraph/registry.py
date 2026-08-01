"""Providers, datasets, records, and retrieval, over SQLite.

SQLite through the standard library keeps the quickstart to one command. The schema is created
on open, so there is no migration step and no service to start.

Retrieval here is a deliberately plain lexical matcher. It is *not* the interesting part of
this project, and it is kept simple so nobody mistakes it for the interesting part: the
attribution engines are indifferent to how sources were selected, and swapping in embeddings
would not change a line of :mod:`datagraph.attribution`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagraph.policy import Disclosure, DisclosurePolicy
from datagraph.text import tokenize

__all__ = [
    "Dataset",
    "Provider",
    "Record",
    "Registry",
    "RegistryError",
    "provider_ids",
    "render_sources",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id           TEXT PRIMARY KEY,
    provider_id  TEXT NOT NULL REFERENCES providers(id),
    name         TEXT NOT NULL,
    policy       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL REFERENCES datasets(id),
    values_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS records_by_dataset ON records(dataset_id);
"""


class RegistryError(Exception):
    """Raised when a registry operation refers to something that does not exist."""


@dataclass(frozen=True)
class Provider:
    id: str
    name: str


@dataclass(frozen=True)
class Dataset:
    id: str
    provider_id: str
    name: str
    policy: DisclosurePolicy


@dataclass(frozen=True)
class Record:
    """One row of provider data, plus the projection the policy permits.

    ``values`` is raw and never leaves this tier. ``disclosed`` is what may be shown, and is
    what the prompt builder receives.
    """

    id: str
    dataset_id: str
    provider_id: str
    values: Mapping[str, Any]
    disclosed: Mapping[str, Any]

    def render(self) -> str:
        """The record as it appears to the model — disclosed fields only."""
        body = ", ".join(f"{k}: {v}" for k, v in sorted(self.disclosed.items()))
        return f"[{self.id}] {body}"


class Registry:
    """Storage and retrieval for providers, datasets, and records."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -------------------------------------------------------------------------

    def add_provider(self, provider_id: str, name: str) -> Provider:
        self._conn.execute("INSERT INTO providers (id, name) VALUES (?, ?)", (provider_id, name))
        self._conn.commit()
        return Provider(id=provider_id, name=name)

    def add_dataset(
        self, dataset_id: str, provider_id: str, name: str, policy: DisclosurePolicy
    ) -> Dataset:
        if self.get_provider(provider_id) is None:
            raise RegistryError(f"unknown provider {provider_id!r}")

        self._conn.execute(
            "INSERT INTO datasets (id, provider_id, name, policy) VALUES (?, ?, ?, ?)",
            (dataset_id, provider_id, name, _dump_policy(policy)),
        )
        self._conn.commit()
        return Dataset(id=dataset_id, provider_id=provider_id, name=name, policy=policy)

    def add_record(self, record_id: str, dataset_id: str, values: Mapping[str, Any]) -> Record:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise RegistryError(f"unknown dataset {dataset_id!r}")

        self._conn.execute(
            "INSERT INTO records (id, dataset_id, values_json) VALUES (?, ?, ?)",
            (record_id, dataset_id, json.dumps(dict(values), sort_keys=True)),
        )
        self._conn.commit()
        return self._to_record(record_id, dataset, dict(values))

    # -- reads --------------------------------------------------------------------------

    def get_provider(self, provider_id: str) -> Provider | None:
        row = self._conn.execute(
            "SELECT id, name FROM providers WHERE id = ?", (provider_id,)
        ).fetchone()
        return Provider(id=row["id"], name=row["name"]) if row else None

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        row = self._conn.execute(
            "SELECT id, provider_id, name, policy FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        return Dataset(
            id=row["id"],
            provider_id=row["provider_id"],
            name=row["name"],
            policy=_load_policy(row["policy"]),
        )

    def providers(self) -> list[Provider]:
        rows = self._conn.execute("SELECT id, name FROM providers ORDER BY id").fetchall()
        return [Provider(id=r["id"], name=r["name"]) for r in rows]

    def datasets(self) -> list[Dataset]:
        rows = self._conn.execute("SELECT id FROM datasets ORDER BY id").fetchall()
        return [d for r in rows if (d := self.get_dataset(r["id"])) is not None]

    def all_records(self) -> list[Record]:
        rows = self._conn.execute(
            "SELECT id, dataset_id, values_json FROM records ORDER BY id"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(self, question: str, limit: int = 8) -> list[Record]:
        """Return up to ``limit`` records whose disclosed content best matches ``question``.

        Scoring is token overlap against each record's disclosed field names and values;
        records with no overlap are excluded. Ties break by record id, so results are stable
        for a given store — which matters, because an unstable source set would show up as
        noise in the attribution measurement.
        """
        wanted = set(tokenize(question))
        if not wanted:
            return []

        scored: list[tuple[int, str, Record]] = []
        for record in self.all_records():
            haystack = " ".join(
                [*record.disclosed.keys(), *(str(v) for v in record.disclosed.values())]
            )
            score = len(wanted & set(tokenize(haystack)))
            if score:
                scored.append((score, record.id, record))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [record for _, _, record in scored[:limit]]

    # -- internals ----------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> Record:
        dataset = self.get_dataset(row["dataset_id"])
        if dataset is None:  # pragma: no cover - foreign keys make this unreachable
            raise RegistryError(f"record {row['id']!r} references a missing dataset")
        return self._to_record(row["id"], dataset, json.loads(row["values_json"]))

    @staticmethod
    def _to_record(record_id: str, dataset: Dataset, values: dict[str, Any]) -> Record:
        return Record(
            id=record_id,
            dataset_id=dataset.id,
            provider_id=dataset.provider_id,
            values=values,
            disclosed=dataset.policy.redact(values),
        )


def provider_ids(records: Iterable[Record]) -> list[str]:
    """Provider ids behind a set of records, for the cohort check."""
    return [r.provider_id for r in records]


def render_sources(records: Sequence[Record]) -> str:
    """Render records for a prompt, in a stable order."""
    return "\n".join(r.render() for r in sorted(records, key=lambda r: r.id))


def _dump_policy(policy: DisclosurePolicy) -> str:
    return json.dumps(
        {
            "levels": {k: str(v) for k, v in policy.levels.items()},
            "buckets": dict(policy.buckets),
            "default": str(policy.default),
        },
        sort_keys=True,
    )


def _load_policy(blob: str) -> DisclosurePolicy:
    raw = json.loads(blob)
    return DisclosurePolicy(
        levels={k: Disclosure(v) for k, v in raw["levels"].items()},
        buckets=raw["buckets"],
        default=Disclosure(raw["default"]),
    )
