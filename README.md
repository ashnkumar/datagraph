# datagraph

Royalty splits for AI answers, measured from the answer itself.

[![ci](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![One query, four providers, 1000 credits. Leave-one-out pays borealis and cascade nothing and its weights sum to 0.2744; both Shapley engines split the credit between them and sum to 1.0000.](docs/demo.gif)

One query, four providers, 1000 credits, three ways of splitting it. `borealis` and `cascade`
hold the same fact, and the two red zeros are what the obvious method pays them for it. Its
weights come to `0.2744` of the payment, so the rest gets handed to whoever happened to be
unique. The seeded fixture makes this fire on every run — see [Limitations](#limitations).

## The problem

Independent providers publish datasets, and someone — a person, or an agent acting for one —
asks a question in plain language. Records come back from whichever providers hold something
relevant, a model writes one answer out of them, and the asker pays once for that answer. That
payment now has to be divided among the providers behind it, in the second the query takes,
with nobody negotiating a rate per question and nobody reviewing an invoice. Splitting it
across whoever got retrieved is the first thing anyone tries, and it pays for *showing up*: a
record that changed nothing earns what the record carrying the answer earns, and the way to
earn more becomes publishing more rows rather than better ones.

The better answer is to ask what each provider was worth by taking it away, and where providers
hold genuinely different data that is the right question. It breaks on the case a marketplace
produces by itself: once two providers hold the same fact, removing either changes nothing, so
both are worth nothing, and the shares that remain no longer add up to the payment being
divided. Closing that gap means scaling the survivors up to fit, which moves the money those
two earned to whoever happened to be the only source of something. **`datagraph` divides the
payment by how much each provider's data actually changed the answer, in shares that add up to
the whole by construction rather than by being scaled to fit.**

That costs model calls — the answer gets generated more than once per query, where the
alternatives generate it once per provider. The ceiling is bounded and
[spelled out below](#where-the-idea-comes-from). The rest of this page is how the measurement
works and how to run it.

## Quickstart

```bash
git clone https://github.com/ashnkumar/datagraph && cd datagraph
uv run datagraph compare
```

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). No API key, no services, no
containers — the default model is a deterministic offline stand-in, so `compare` prints the
same table on any machine. `uv run datagraph demo` runs one query end to end and shows the
working: what was retrieved, what redaction removed, the answer, and who got paid.

To use the real API instead, put your key in `.env` (`cp .env.example .env`) and add `--live`.

## What you get

A researcher escrows a payment and asks a question. Records are retrieved, redacted to what
each provider's policy permits, and answered by a model. The payment is then divided by **how
much each provider's data moved that answer**, and the escrow settles in one transaction.

| | Paying by retrieval | With `datagraph` |
|---|---|---|
| **Two providers hold the same fact** | Both look equally cited, or under leave-one-out both score zero and their share is silently reassigned | They split the credit — `195` and `203` of 1000 from the sampled engine, `197` each computed exactly |
| **Does the payment balance** | Shares don't add up to the whole, so the gap is closed by scaling them to fit | Shares add up to the whole exactly, so the escrow is exhausted with nothing left to redistribute |
| **Padding your dataset** | More rows returned means more payout | Payout is invariant to row count — 10 extra copies of every `delta` record leave it on `365` credits |
| **Hidden fields** | Redaction protects the prompt; the result object still carries the raw row | Raw values never cross the registry boundary, on the answered path or the refused one |
| **Narrow queries** | A query matching one provider is answered | Refused before the model is called if fewer than 3 providers are behind it |

Leave-one-out is the alternative worth taking seriously, and where providers hold disjoint data
it wins outright: one model call per provider instead of one per combination, and with no
overlap the two engines rank providers the same way. Redundancy is what it cannot survive, and
redundancy is what a marketplace accumulates as providers with similar data join. Two providers
supplying the same indispensable fact each measure as removable, so both score zero, and
because the surviving shares no longer add up to the payment, scaling them to fit hands that
credit to somebody else. It fails quietly, and quiet is the problem.

## How it works

![Three panels. One: a researcher escrows 1000 credits and asks a question; five records are retrieved and redacted by policy. Two: the answer is re-generated from every combination of providers to measure what each one contributed, and adding cascade to borealis changes the score by exactly zero. Three: the shares become whole credits and the escrow settles.](docs/how-it-works.png)

- **Redaction precedes the prompt.** Each dataset assigns every field `OPEN`, `DERIVED`
  (numbers banded, dates truncated to the month), or `HIDDEN`. A hidden field is absent from the
  object the prompt builder receives, so there is no prompt-side rule that could leak it.
  Policies fail closed: an unlisted field is hidden, and a `DERIVED` value with no safe coarse
  form is dropped.
- **Contribution is measured by regenerating the answer.** Every combination of providers is
  scored on how much of the full answer it can still produce on its own. Scores are cached per
  combination, so each one costs at most one generation per query.
- **Settlement is one transaction, and it is checked.** Fractional shares become whole credits
  under a rounding rule that cannot lose or invent one. The ledger refuses a settlement whose
  payouts don't exhaust the escrow, so an attribution bug fails loudly instead of quietly
  losing money.

### Architecture

![Four stacked layers: the command line; the marketplace loop holding escrow, retrieval, the cohort check and settlement; the attribution engines and the model client; and a SQLite registry beside the double-entry ledger.](docs/architecture.png)

| # | Component | Module | What it does |
|---|---|---|---|
| **1** | Command line | `cli.py` | The only interface — `demo`, `compare`, `providers` |
| **2** | Marketplace loop | `marketplace.py` | Escrow, retrieve, redact, check the cohort floor, attribute, settle |
| **3** | Attribution | `attribution.py` | Both engines and the scoring, in one file on purpose |
| **4** | Model | `models.py` | The `ModelClient` protocol, the Anthropic client, the offline stand-in |
| **5** | Policy | `policy.py` | Disclosure levels, redaction, the cohort floor |
| **6** | Registry | `registry.py` | Providers, datasets, records, retrieval — SQLite, stdlib only |
| **7** | Money | `ledger.py`, `money.py` | Double-entry accounts with escrow; integer credits and apportionment |

Start with `src/datagraph/attribution.py`. It holds both engines and it is 337 lines.

## Where the idea comes from

Machine learning has a well-worn way of asking *which training examples actually made this
model better*: hold out subsets of the data, retrain, and watch what changes.
[Data Shapley](https://arxiv.org/abs/1904.02868) (Ghorbani & Zou, 2019) is the version of that
idea aimed at compensation — value each example by how much it improves the model, and you have
a principled basis for paying whoever supplied it. The observation that leave-one-out is the
weaker measure is theirs as well.

It does not transfer directly. Retraining a model per subset is not something that can happen
inside a single query, and the thing being paid is not a training example but a **provider**,
who might have contributed one record or fifty. So the shape is borrowed and the internals are
different:

- Instead of retraining, `datagraph` **re-answers** — it regenerates the answer from a subset
  of providers and scores how much of the full answer survives.
- Instead of paying per training point, it **pays per provider**. That is what stops someone
  earning more by slicing one record into ten. With records as players it worked: cloning one
  of `delta`'s two records four times took it from `446` to `612` credits of 1000, a 37% raise
  for publishing nothing new. With providers as players, ten extra copies of every record it
  holds move the payout by nothing at all.
- Because the shares are worked out this way, they **always add up to exactly the payment being
  divided** — the property that lets an escrow settle with no leftover and no scaling step.

The cost is that measuring a provider's contribution properly means scoring every combination
of them, and that count doubles with each provider added. Two things bound it. Retrieval returns
at most six records, so no query ever has more than six providers in play and `2^6 = 64`
generations is the worst case; and every combination is scored once and cached, which is why
sampling 2000 orderings and enumerating all 16 combinations cost the same 16 model calls on the
demo query.

A cheaper route exists, and it is worth knowing why it isn't taken here.
[ContextCite](https://arxiv.org/abs/2409.00729) (Cohen-Wang et al., 2024) attributes a generated
answer to its sources by fitting a sparse linear model over a few dozen ablations rather than
scoring every combination, and for deciding *which source backs this sentence* it is the better
tool. It is not a way to divide a payment. It drives most coefficients to exactly zero, and
regression weights carry no constraint that they add up to the amount being split — both useful
for attribution, disqualifying for settlement.

`SPEC.md` has the formal version of all of this — the derivation, the sampling method and its
citation, the alternatives that were rejected and why, and every design revision with the test
that forced it.

## Commands

| Command | What it does |
|---|---|
| `datagraph demo` | One query end to end: retrieval, redaction, the answer, payouts, ledger check |
| `datagraph compare` | The same query under all three engines, side by side |
| `datagraph providers` | The seeded providers and which fields each one discloses |

`--question` and `--payment` work on `demo` and `compare`; `--engine` selects `shapley`,
`exact_shapley`, or `leave_one_out` on `demo`. `--live` calls the real API instead of the
offline model — on `compare` that is three runs, so it says what it is about to spend.
`.env.example` lists every environment variable.

## Tests

```bash
uv sync --extra dev
uv run pytest            # 138 tests, offline, no API key
uv run pytest -m live    # one end-to-end test against the real API
```

The engines are tested against small synthetic games whose correct answers are known in closed
form, so the assertions check the mathematics rather than a model's mood — read
`tests/test_attribution.py` first. Identical contributors being paid identically, a contributor
who changes nothing being paid zero, and the shares adding up to the whole are all asserted,
which is what a stubbed engine fails.

`tests/test_docs.py` pins this page to the code. Every number quoted above is re-derived there,
including the line count two sections up, so a claim cannot go stale without a test failing.

## Limitations

- **Replication-proof against rows, not against identities.** Ten extra copies of every record
  `delta` holds move its payout by 0 credits. Registering twice does: splitting `delta`'s two
  records across two provider accounts took its combined take from `365` to `446` credits of
  1000, and `446` is exactly what records-as-players paid `delta` in the first place. Paying per
  provider doesn't remove the record-level attack, it charges an identity for it — and nothing
  here verifies that two providers are different people, so a real deployment needs identity or
  stake underneath the measurement.
- **The demo's redundancy is staged.** `borealis` and `cascade` disclose identical records on
  purpose, and `sample_data.py` says so in its docstring. Real overlap is partial; total
  duplication is the case where the discount reaches 100% and the arithmetic is legible on one
  screen. The behavior doesn't depend on the staging — leave-one-out discounts *any*
  corroborated provider in proportion to how completely somebody else covers them.
- **Generation is not deterministic on the live path, and cannot be made so.** Any non-default
  `temperature`, `top_p` or `top_k` returns a 400, there is no `seed` parameter, and Anthropic's
  migration guidance notes that `temperature = 0` never guaranteed identical outputs anyway.
  Scores are cached per combination, so comparison *within* a query is self-consistent and
  across queries it is not. The offline model has no such problem, which is why it is the
  default.
- **Privacy is procedural, not cryptographic.** An operator with database access reads raw
  records — `HIDDEN` fields sit in cleartext SQLite — and a compromised process bypasses
  redaction. Real guarantees need the raw values never to reach this tier: trusted execution,
  secure aggregation, or local differential privacy on the provider's side. Running `--live`
  also sends the disclosed projection to a third party, which is the offline model's other
  reason for existing.

`SPEC.md` §7 and §8 have the rest: that the contribution score is a proxy for meaning rather
than a measure of it, that the cohort floor blocks the obvious narrowing attack and nothing
subtler across a *sequence* of queries, that retrieval is a deliberately dull lexical matcher,
and that the ledger is in-memory and single-process while only the registry persists. There is
also no chain, no wallet and no token here — the credits are integers in a double-entry ledger
in one process, and `ledger.py` is where a real payment rail would be substituted.

## License

MIT — see [LICENSE](LICENSE).
