# datagraph

A reference implementation for buying access to third-party data per RAG query and splitting the
payment by contribution.

[![ci](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ashnkumar/datagraph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![One query, four providers, 1000 credits. Leave-one-out pays borealis and cascade nothing and its weights sum to 0.2744; both Shapley engines split the credit between them and sum to 1.0000.](docs/demo.gif)

The demo runs one synthetic query against 4 data providers and compares 3 payout methods.
`borealis` and `cascade` intentionally supply the same fact. Leave-one-out pays both 0 because
removing either provider leaves the other one in the answer. Its measured shares total only
`0.2744`. Both Shapley methods split the credit between them and total `1.0000`.

*See the **[technical post](https://example.com/datagraph-technical-post)** for more details.*

## Quickstart

```bash
git clone https://github.com/ashnkumar/datagraph && cd datagraph
uv run datagraph compare
```

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). No API key, service, or container is needed.
The default model is a deterministic offline stand-in, so `compare` prints the same result on every
machine. It runs the same query through all 3 payout methods.

`uv run datagraph demo` runs one query from retrieval through settlement. It shows the records that
were retrieved, the fields removed by each provider's disclosure policy, the generated answer, and
the final payouts.

**To use the Anthropic API:** `cp .env.example .env`, put your key in it, and add `--live`.

## What this implements

In a conventional retrieval-augmented generation (RAG) application, the developer first assembles
or licenses a corpus and retrieval searches that corpus. `datagraph` is a reference implementation
of a different setup: an AI application can buy access to relevant records from _multiple_
independent data providers for a query, then pay those providers from the query fee.

This gives AI developers a way to search beyond their own corpus without first importing or
licensing every provider's full dataset. It would let a provider make useful, paywalled records
available for specific questions instead of selling unrestricted access to the whole collection.

This is a _proposed_ transaction model that we've implemented locally; this market doesn't exist
yet. The providers and records are synthetic, the registry is local, and payments use internal
credits rather than an external payment rail. `datagraph` implements the complete query
transaction:

1. A user asks a question and puts a payment in escrow.
2. Records are retrieved from several providers and redacted under each provider's policy.
3. A model uses the returned records to write the answer.
4. **The system measures how much each provider's records informed that answer.**
5. The escrowed payment is divided in those proportions and released to the providers.

The main design problem is step 4. A provider should be paid for improving the answer, not just for
appearing in the retrieval results.

## How the payment is split

Paying _per retrieved chunk_ would reward providers with larger datasets, even if their records
weren't useful to the final answer. A provider could earn more by publishing more matching rows
without adding new information.

_Leave-one-out_ seems like a better option. It removes one provider, generates the answer again, and
measures what changed. This works when every provider supplies _different_ information. It fails when
2 providers supply the same fact: removing either one changes nothing because the other still
supplies it, so both receive a contribution score of 0.

The demo makes that case explicit:

| Provider | Sampled Shapley | Exact Shapley | Leave-one-out |
|---|---:|---:|---:|
| `aurora` | 243 | 241 | 287 |
| `borealis` | 195 | 197 | 0 |
| `cascade` | 203 | 197 | 0 |
| `delta` | 359 | 365 | 713 |
| **Weights sum to** | **1.0000** | **1.0000** | **0.2744** |

The sampled engine pays the redundant pair `195` and `203` of 1000. The exact engine pays them
`197` each computed exactly. Leave-one-out has only `0.2744` of a payment to allocate. Scaling its
remaining weights to fill the escrow moves the missing share to `aurora` and `delta`, even though
that share came from the fact supplied by `borealis` and `cascade`.

We included leave-one-out to make this failure reproducible, not because it's safe for settlement.
When selected, it produces a complete payout by normalizing the weights, and the CLI labels the
result as inefficient. The default engine is sampled Shapley attribution.

Shapley measures each provider in different provider orderings. When 2 providers hold the same fact,
each gets credit when it appears first. Exact Shapley averages every ordering; the default engine
estimates the same split by sampling them. The contributions in each ordering add up to the full
payment, so the final shares do too.

The tradeoff is cost. Leave-one-out needs one generation per provider, plus the full answer and the
no-records baseline. On this demo that's 6 calls against 16. Sampled and exact Shapley both reach
the same 16 model calls because results are cached by provider combination. For larger provider sets,
exact enumeration grows exponentially, while the sampled engine can stop before visiting every
combination.

## How one query works

![Three panels. One: a researcher escrows 1000 credits and asks a question; five records are retrieved and redacted by policy. Two: the answer is re-generated from every combination of providers to measure what each one contributed, and adding cascade to borealis changes the score by exactly zero. Three: the shares become whole credits and the escrow settles.](docs/how-it-works.png)

**Retrieve and redact.** The registry searches the fields that each provider allows the application
to see. Fields marked `OPEN` pass through unchanged, `DERIVED` fields are coarsened, and `HIDDEN`
fields are removed before the prompt gets built. Unlisted fields are hidden by default. The query is
refused before generation unless its records come from at least 3 provider accounts. These are
application rules, not cryptographic privacy; the raw records remain in SQLite.

**Generate and measure.** The model first answers from all retrieved providers. The attribution
engine then answers from subsets of those providers and compares each result with the full answer.
The shipped scorer is token-set F1 with the similarity of the no-records answer subtracted as a
baseline. Providers, not individual records, are the units being measured. This is why adding 10
extra copies of every `delta` record would still leave it on `365` credits under exact Shapley.

**Settle or refund.** The measured shares are converted to whole credits with largest-remainder
rounding. The double-entry ledger accepts the settlement only if its payouts exhaust the escrow. A
provider can sometimes score below 0 when its records make the answer less similar to the full
answer. The system doesn't turn that score into a debt or scale the positive shares down. It refunds
the query instead.

### Architecture

![Four stacked layers: the command line; the marketplace loop holding escrow, retrieval, the cohort check and settlement; the attribution engines and the model client; and a SQLite registry beside the double-entry ledger.](docs/architecture.png)

| # | Component | Module | What it does |
|---|---|---|---|
| **1** | Command line | `cli.py` | Runs the seeded demo and exposes the 3 attribution engines |
| **2** | Query transaction | `marketplace.py` | Escrows the payment, retrieves and redacts records, measures contribution, and settles or refunds |
| **3** | Attribution | `attribution.py` | Implements sampled Shapley, exact Shapley, leave-one-out, and answer scoring |
| **4** | Model | `models.py` | Provides the deterministic offline model and the Anthropic API client behind one interface |
| **5** | Disclosure policy | `policy.py` | Applies `OPEN`, `DERIVED`, and `HIDDEN` rules and enforces the cohort floor |
| **6** | Data registry | `registry.py` | Stores providers, datasets, and records in SQLite and runs lexical retrieval |
| **7** | Credits | `ledger.py`, `money.py` | Holds payments in escrow, converts fractional shares to integer credits, and records balanced transfers |

Start with `src/datagraph/attribution.py`. It holds all 3 engines and it's 365 lines. `SPEC.md`
documents the formulas, rejected alternatives, failure cases, and the tests that led to the current
design.

## Commands

| Command | What it does |
|---|---|
| `datagraph compare` | Runs the same question under sampled Shapley, exact Shapley, and leave-one-out |
| `datagraph demo` | Runs one query end to end and prints retrieval, redaction, answer, attribution, payout, and ledger status |
| `datagraph providers` | Lists the seeded providers, their record counts, and the fields they disclose |

`--question` and `--payment` work with `demo` and `compare`. `--engine` selects `shapley`,
`exact_shapley`, or `leave_one_out` for `demo`. `--live` uses the Anthropic API instead of the
offline model. With `compare`, that means 3 complete query runs. `.env.example` lists every
environment variable.

## Tests

```bash
uv sync --extra dev
uv run pytest            # 150 tests, offline, no API key
uv run pytest -m live    # one end-to-end test against the Anthropic API
```

The attribution tests use synthetic games with known answers. They check that the payment is fully
allocated, identical contributors are treated identically, and a provider that changes nothing is
paid nothing. The marketplace tests cover escrow, refunds, disclosure policies, the cohort floor,
and negative contribution scores. Property-based tests check that credits are conserved, accounts
don't go negative, and every escrow is settled or refunded.

`tests/test_docs.py` derives the figures in this README from the implementation. A payout, weight
sum, model-call count, replication example, or source line count that changes without a matching
documentation update fails the test suite.

## Limitations

- **The cost grows with the provider count.** Exact Shapley needs every provider combination. The
  default cap is 6 providers, or 64 combinations. Sampling can reduce the work for larger sets.
- **Live payouts are estimates.** The scorer measures token overlap, and live model responses vary.
  The shares still total the payment, but the split can move between runs.
- **Provider accounts aren't verified identities.** Splitting `delta`'s 2 records across 2 accounts
  changes their combined exact-Shapley payout from `365` to `446` of 1000. Preventing this requires
  identity controls outside the attribution layer.

## License

MIT — see [LICENSE](LICENSE).
