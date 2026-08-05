# datagraph

Pay for the data that changed the answer, not the data that showed up.

[![ci](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![One query, four providers, 1000 credits. Leave-one-out pays borealis and cascade nothing and its weights sum to 0.2744; both Shapley engines split the credit between them and sum to 1.0000.](docs/demo.gif)

One question, four data providers, 1000 credits in escrow, three ways of dividing it. `borealis` and
`cascade` hold the same fact, so removing either changes nothing and leave-one-out scores both zero —
their credits go to whoever happened to be unique. That data is constructed to make the failure
visible; the run is not doctored, and it prints the same table on your machine.

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
| **Does the payment balance** | Weights don't sum to the whole, so the gap is papered over by normalising | Weights sum to `v(N)` exactly, so the escrow is exhausted with nothing left to redistribute |
| **Padding your dataset** | More rows returned means more payout | Payout is invariant to row count — 10 extra copies of a record move `delta` by 0 credits |
| **Hidden fields** | Redaction protects the prompt; the result object still carries the raw row | Raw values never cross the registry boundary, on the answered path or the refused one |
| **Narrow queries** | A query matching one provider is answered | Refused before the model is called if fewer than 3 providers are behind it |

The obvious alternative is **leave-one-out** — "how much worse is the answer without you?" — and when
providers hold genuinely disjoint data it is the right choice: `n` model calls instead of `2^n`, and
with no overlap the two engines rank providers the same way. What it cannot survive is redundancy,
which is what a marketplace accumulates as providers with similar data join. Two providers supplying
the same indispensable fact each measure as removable, so both score zero, and because the surviving
weights don't sum to the payment, normalisation hands their share to somebody else. It fails quietly,
and quiet is the problem.

## How it works

![Three panels. One: a researcher escrows 1000 credits and asks a question; five records are retrieved and redacted by policy. Two: the answer is regenerated from every subset of providers to measure what each one contributed. Three: the weights become whole credits and the escrow settles.](docs/how-it-works.png)

- **Redaction precedes the prompt.** Each dataset assigns every field `OPEN`, `DERIVED` (numbers
  banded, dates truncated to the month), or `HIDDEN`. A hidden field is absent from the object the
  prompt builder receives, so there is no prompt-side rule that could leak it. Policies fail closed:
  an unlisted field is hidden, and a `DERIVED` value with no safe coarse form is dropped.
- **Contribution is measured by regenerating the answer.** `v(S)` is how much of the full answer
  survives when only the providers in `S` are present, scored against the answer from all of them.
  Coalition values are memoised, so each subset costs at most one generation per query.
- **Settlement is one transaction, and it is checked.** Weights become whole credits by
  largest-remainder apportionment over exact rationals. The ledger refuses a settlement whose payouts
  don't exhaust the escrow, so an attribution bug fails loudly instead of quietly losing money.

### Architecture

![Four stacked layers: the command line; the marketplace loop holding escrow, retrieval, the cohort check and settlement; the attribution engines and the model client; and a SQLite registry beside the double-entry ledger.](docs/architecture.png)

| # | Component | Module | What it does |
|---|---|---|---|
| **1** | Command line | `cli.py` | The only interface — `demo`, `compare`, `providers` |
| **2** | Marketplace loop | `marketplace.py` | Escrow, retrieve, redact, check the cohort floor, attribute, settle |
| **3** | Attribution | `attribution.py` | Both engines and `v(S)`, in one file on purpose |
| **4** | Model | `models.py` | The `ModelClient` protocol, the Anthropic client, the offline stand-in |
| **5** | Policy | `policy.py` | Disclosure levels, redaction, the cohort floor |
| **6** | Registry | `registry.py` | Providers, datasets, records, retrieval — SQLite, stdlib only |
| **7** | Money | `ledger.py`, `money.py` | Double-entry accounts with escrow; integer credits and apportionment |

Start with `src/datagraph/attribution.py`. It holds both engines and it is 337 lines.

## Paying for contribution, not for presence

**Dividing one payment among the data behind one answer is a cost-allocation problem, not a ranking
problem.** Framed as a cooperative game, the players are the providers whose records reached the
model, `v(S)` is how much of the answer is recoverable from a subset, and each player's payout is
their share of `v(N)`.

That frame is [Data Shapley](https://arxiv.org/abs/1904.02868) (Ghorbani & Zou, ICML 2019), which
values *training* data by its marginal contribution to a trained model's performance, motivated by
the same premise — that people should be compensated for the data they generate. This project moves
the skeleton to *retrieved* data and a single generated answer: `v(S)` is answer similarity rather
than validation accuracy, and the players are providers rather than training points. The finding that
leave-one-out is the weaker measure is theirs, not mine.

The **Shapley value** is the unique allocation satisfying efficiency, symmetry, null player, and
additivity. Efficiency is the one that makes settlement sound: the weights sum to `v(N)`, so the
escrow is exhausted with no fudge factor, and providers holding the same fact split the credit
instead of both being zeroed.

Four other designs were tried or considered first:

| Rejected | Why |
|---|---|
| **Count retrievals** — split the payment across every record returned | Presence in a result set is not contribution. A record that changed nothing earns what the record carrying the answer earns, and adding rows adds income |
| **Leave-one-out** | Not an efficient allocation — weights sum to `0.2744` of the payment on the demo query. Normalising that gap is what moves redundant providers' credit to unique ones |
| **A sparse linear surrogate**, as in [ContextCite](https://arxiv.org/abs/2409.00729) (Cohen-Wang et al., NeurIPS 2024) | The right tool for its job and far cheaper — it fits a LASSO surrogate over random source ablations and reports needing about 32 of them where exact Shapley needs `2^n`. But it is built to *cite*, not to *pay*: sparsity drives most sources to exactly zero by design, and regression coefficients carry no constraint that they sum to the whole. Zero means unpaid, and no sum means the escrow does not balance |
| **Per-record Shapley players** | The Shapley value is not replication-proof. With records as players, one provider split a row into four copies and moved from 200 to 326 credits of 600 for no new information |

Players are therefore **providers**, not records. A coalition names providers and is answered from
all of that provider's records at once, so duplicating a row changes neither the player set nor any
coalition's content. It is also cheaper: the coalition space is `2^providers`, not `2^records` — the
demo went from 32 model calls to 16.

**The cost.** Exact Shapley is `2^n` evaluations. The default engine estimates it by Monte-Carlo
permutation sampling ([Castro, Gómez & Tejada 2009](https://doi.org/10.1016/j.cor.2008.04.004)) —
sample random arrival orders, average each player's marginal contribution. Because every
permutation's marginals telescope to `v(N) − v(∅)`, **the estimator is efficient exactly, not just in
expectation**: sampling moves credit between players but never creates or destroys any. With
memoisation and a six-provider cap, sampling ends up visiting most of the coalition space anyway, so
`--engine exact_shapley` costs about the same here and has no variance.

**The demo data is built to make this visible, and the construction is the argument.** `borealis` and
`cascade` disclose identical records because `sample_data.py` was written that way, and its docstring
says so. Redundancy does not arrive on cue, so a demo that waits for it is not a demo. What the
staging buys is a failure you can see in one screen that fires on every run — the offline model is
deterministic and the permutation sampler is seeded, so `compare` prints the table above on any
machine. What it costs is that this is the minimal case rather than a natural one. Real overlap is
partial rather than total, and leave-one-out's discount applies to *any* corroborated provider, not
only perfectly duplicated ones: the more of your contribution somebody else also covers, the less
removing you changes, and the less you are paid.

`SPEC.md` has the derivation, the data model, and every design revision with the test that forced it.

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

The engines are tested against synthetic cooperative games with known closed-form Shapley values, so
the assertions check the mathematics rather than a model's mood — read `tests/test_attribution.py`
first. Efficiency, symmetry and null-player are asserted, which is what a stubbed engine fails.

## Limitations

- **Replication-proof against rows, not against identities.** Cloning a record earns nothing — 10
  extra copies move `delta` by 0 credits. Registering twice does: splitting `delta`'s two records
  across two provider accounts raised its combined take from 365 to 446 credits of 1000. Nothing here
  verifies that two providers are different people, and the Shapley value is not false-name-proof.
- **`--live` sends the disclosed projection to a third party.** The privacy argument is about what
  leaves the provider's record, and `OPEN` and `DERIVED` values leave the machine entirely on a live
  run. The offline model exists partly so the whole system can be exercised without that happening.
- **`v(S)` is a proxy.** The default similarity is F1 over content words, so it measures whether the
  same *content* survived, not whether the same *meaning* did. `Similarity` is a protocol with one
  shipped implementation; nothing in the engines changes if you write another.
- **Generation is not deterministic on the live path.** The API rejects `temperature`, `top_p` and
  `top_k` on current models — `400 "temperature is deprecated for this model"` — so two identical
  calls can differ. Coalition values are memoised, so comparison *within* a query is self-consistent;
  across queries it is not. The offline model has no such problem, which is why it is the default.
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
