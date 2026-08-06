# datagraph

The royalty split for an AI answer.

[![ci](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![One query, four providers, 1000 credits. Leave-one-out pays borealis and cascade nothing and its weights sum to 0.2744; both Shapley engines split the credit between them and sum to 1.0000.](docs/demo.gif)

**The setup.** Independent providers publish datasets. Someone — a person, or an agent acting for
one — asks a question in plain language. Records are pulled from whichever providers have something
relevant, a model writes the answer from them, and the asker pays once for that answer.

**The problem.** That one payment now has to be divided among the providers behind it, automatically,
in the second the query takes. Nobody negotiates a rate per question and nobody reviews an invoice.
The obvious rule is to split the payment across whoever got retrieved — but that pays for *showing
up*. A record that changed nothing earns what the record carrying the answer earns, and the way to
earn more becomes publishing more rows rather than better ones.

**What this does.** `datagraph` measures each provider's contribution by re-answering the question
from every combination of providers and seeing how much of the answer survives without them, then
pays each one in proportion to what they actually accounted for. The idea is borrowed from **Data
Shapley**, which values *training* data by how much each example improves a model and argues that is
a basis for paying the people who supply it. This applies the same shape to *retrieved* data and a
single generated answer — what had to change is [further down](#where-the-idea-comes-from).

Get that measurement right and a different kind of marketplace becomes possible: one where a provider
is paid per answer their data earned rather than per seat or per row, and where holding better data
beats holding more of it.

## Where this is meant to run

The natural home for this is a decentralised network. Providers there don't have to trust an
operator, payouts execute without one, and identities are pseudonymous so anyone can join without
asking — most of what an open data marketplace needs, and awkward to assemble any other way short of
a trusted middleman.
Such networks already have data providers and already pay them. What they tend not to have is a
defensible answer to *how much*. Moving the money is the solved part. Deciding the split is not.

**This repository implements none of that.** There is no chain, no wallet and no token; the "credits"
are integers in a double-entry ledger that lives in one process. What it implements is the part that
has to be right before settlement means anything — the measurement, the split, and the accounting
that proves the split is exact. `ledger.py` is the seam: wiring settlement to a real payment rail is
a substitution there, and nothing above it changes. Which rail is deliberately none of this project's
business.

That is the last you will read about it here, with one exception in [Limitations](#limitations),
where the choice of platform bites.

## Quickstart

```bash
git clone https://github.com/ashnkumar/datagraph && cd datagraph
uv run datagraph compare
```

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). No API key, no services, no containers.
`uv run datagraph demo` runs one query end to end and shows the working — what was retrieved, what
redaction removed, the answer, and who got paid.

To use the real API instead, put your key in `.env` (`cp .env.example .env`) and add `--live`.

## What you get

A researcher escrows a payment and asks a question. Records are retrieved, redacted to what each
provider's policy permits, and answered by a model. The payment is then divided by **how much each
provider's data moved that answer**, and the escrow settles in one transaction.

| | Paying by retrieval | With datagraph |
|---|---|---|
| **Two providers hold the same fact** | Both look equally cited, or under leave-one-out both score zero and their share is silently reassigned | They split the credit — `195` and `203` of 1000 from the sampled engine, `197` each when computed exactly |
| **Does the payment balance** | Shares don't add up to the whole, so the gap is papered over by normalising | Shares add up to the whole exactly, so the escrow is exhausted with nothing left to redistribute |
| **Padding your dataset** | More rows returned means more payout | Payout is invariant to row count — 10 extra copies of a record move `delta` by 0 credits |
| **Hidden fields** | Redaction protects the prompt; the result object still carries the raw row | Raw values never cross the registry boundary, on the answered path or the refused one |
| **Narrow queries** | A query matching one provider is answered | Refused before the model is called if fewer than 3 providers are behind it |

The obvious alternative is **leave-one-out** — "how much worse is the answer without you?" — and when
providers hold genuinely disjoint data it is the right choice: one model call per provider instead of
one per combination, and with no overlap the two engines rank providers the same way. What it cannot
survive is redundancy, which is what a marketplace accumulates as providers with similar data join.
Two providers supplying the same indispensable fact each measure as removable, so both score zero,
and because the surviving shares no longer add up to the payment, normalising hands their credit to
somebody else. It fails quietly, and quiet is the problem.

**The fixture behind those numbers is constructed.** `borealis` and `cascade` disclose identical
records on purpose, and `sample_data.py` says so in its docstring. Redundancy does not arrive on cue,
and staging it buys a failure that is visible on one screen and fires on every run — the offline model
is deterministic and the sampler is seeded, so `compare` prints the same table on any machine. What it
costs is that this is the minimal case rather than a natural one. Real overlap is partial, and
leave-one-out discounts *any* corroborated provider in proportion to how completely somebody else
covers them.

## How it works

![Three panels. One: a researcher escrows 1000 credits and asks a question; five records are retrieved and redacted by policy. Two: the answer is re-generated from every combination of providers to measure what each one contributed, and adding cascade to borealis changes the score by exactly zero. Three: the shares become whole credits and the escrow settles.](docs/how-it-works.png)

- **Redaction precedes the prompt.** Each dataset assigns every field `OPEN`, `DERIVED` (numbers
  banded, dates truncated to the month), or `HIDDEN`. A hidden field is absent from the object the
  prompt builder receives, so there is no prompt-side rule that could leak it. Policies fail closed:
  an unlisted field is hidden, and a `DERIVED` value with no safe coarse form is dropped.
- **Contribution is measured by regenerating the answer.** Every combination of providers is scored
  on how much of the full answer it can still produce on its own. Scores are cached per combination,
  so each one costs at most one generation per query.
- **Settlement is one transaction, and it is checked.** Fractional shares become whole credits under
  a rounding rule that cannot lose or invent one. The ledger refuses a settlement whose payouts don't
  exhaust the escrow, so an attribution bug fails loudly instead of quietly losing money.

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

Machine learning has a well-worn way of asking *which training examples actually made this model
better*: hold out subsets of the data, retrain, and watch what changes.
[Data Shapley](https://arxiv.org/abs/1904.02868) (Ghorbani & Zou, 2019) is the version of that idea
aimed at compensation — value each example by how much it improves the model, and you have a
principled basis for paying whoever supplied it. The observation that leave-one-out is the weaker
measure is theirs too, not mine.

It does not transfer directly. Retraining a model per subset is not something that can happen inside
a single query, and the thing being paid is not a training example but a **provider**, who might have
contributed one record or fifty. So the shape is borrowed and the internals are different:

- Instead of retraining, `datagraph` **re-answers** — it regenerates the answer from a subset of
  providers and scores how much of the full answer survives.
- Instead of paying per training point, it **pays per provider**. That is what stops someone earning
  more by slicing one record into ten. It used to work: with records as players, cloning a row four
  times moved one provider from 200 to 326 credits out of 600. With providers as players, ten extra
  copies move the payout by nothing at all.
- Because the shares are worked out this way, they **always add up to exactly the payment being
  divided** — which is the property that lets an escrow settle with no leftover and no fudge factor.

The cost is that measuring a provider's contribution properly means scoring every combination of
them, and that count doubles with each provider added. The default engine estimates it by sampling
instead of enumerating, and retrieval returns at most six records — so no query has more than six
providers in play, and the exact engine stays affordable too.

`SPEC.md` has the formal version of all of this — the derivation, the sampling method and its
citation, the alternatives that were rejected and why, and every design revision with the test that
forced it.

## Commands

| Command | What it does |
|---|---|
| `datagraph demo` | One query end to end: retrieval, redaction, the answer, payouts, ledger check |
| `datagraph compare` | The same query under all three engines, side by side |
| `datagraph providers` | The seeded providers and which fields each one discloses |

`--question` and `--payment` work on `demo` and `compare`; `--engine` selects `shapley`,
`exact_shapley`, or `leave_one_out` on `demo`. `--live` calls the real API instead of the offline
model — on `compare` that is three runs, so it says what it is about to spend. `.env.example` lists
every environment variable.

## Tests

```bash
uv sync --extra dev
uv run pytest            # 122 tests, offline, no API key
uv run pytest -m live    # one end-to-end test against the real API
```

The engines are tested against small synthetic games whose correct answers are known in closed form,
so the assertions check the mathematics rather than a model's mood — read `tests/test_attribution.py`
first. Identical contributors being paid identically, a contributor who changes nothing being paid
zero, and the shares adding up to the whole are all asserted, which is what a stubbed engine fails.

## Limitations

- **Replication-proof against rows, not against identities.** Cloning a record earns nothing — 10
  extra copies move `delta` by 0 credits. Registering twice does: splitting `delta`'s two records
  across two provider accounts raised its combined take from 365 to 446 credits of 1000. Nothing here
  verifies that two providers are different people. This is the exception promised at the top — the
  pseudonymous network that makes the rest of this natural is also the environment where a second
  identity is free, so a real deployment needs identity or stake sitting underneath the measurement.
- **`--live` sends the disclosed projection to a third party.** The privacy argument is about what
  leaves the provider's record, and `OPEN` and `DERIVED` values leave the machine entirely on a live
  run. The offline model exists partly so the whole system can be exercised without that happening.
- **The contribution score is a proxy.** The default measure is F1 over content words, so it captures
  whether the same *content* survived, not whether the same *meaning* did. `Similarity` is a protocol
  with one shipped implementation; nothing in the engines changes if you write another.
- **Generation is not deterministic on the live path.** The API rejects `temperature`, `top_p` and
  `top_k` on current models — `400 "temperature is deprecated for this model"` — so two identical
  calls can differ. Scores are cached per combination, so comparison *within* a query is
  self-consistent; across queries it is not. The offline model has no such problem, which is why it
  is the default.
- **Privacy is procedural, not cryptographic.** An operator with database access reads raw records —
  the store keeps `HIDDEN` fields in cleartext SQLite — and a compromised process bypasses redaction.
  Real guarantees need the raw values never to reach this tier: trusted execution, secure aggregation,
  or local differential privacy on the provider's side.
- **The cohort floor is blunt.** It refuses a query behind fewer than 3 providers, which blocks the
  obvious narrowing attack and nothing subtler. It does not track what an adversary learns across a
  *sequence* of permitted queries.

Also: retrieval is a deliberately dull lexical matcher; the ledger is in-memory and single-process
while only the registry persists; payouts are per query with no batching; and the escrow assumes one
researcher paying up front.

## License

MIT — see [LICENSE](LICENSE).
