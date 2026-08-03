# datagraph

A data marketplace where providers get paid for *measurably* changing the answer, not for
happening to show up in a search result.

<!-- TODO: CI badge once the repository is public — must point at the real workflow. -->

Providers publish datasets under a disclosure policy. A researcher escrows a payment and asks
a question. The system retrieves records, redacts each one to what its policy permits, has
Claude answer from what survives, then measures how much each provider's records actually
changed that answer — and settles the escrow in proportion.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. No API key, no services, no
containers — the default model is a deterministic local stand-in.

```bash
git clone TODO-REPO-URL && cd datagraph
uv run datagraph demo
```

To see the point of the project in one command:

```bash
uv run datagraph compare
```

```
                     Payout by attribution engine (credits)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ provider             ┃     shapley ┃      exact_shapley ┃      leave_one_out ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ aurora               │         243 │                241 │                287 │
│ borealis             │         195 │                197 │                  0 │
│ cascade              │         203 │                197 │                  0 │
│ delta                │         359 │                365 │                713 │
├──────────────────────┼─────────────┼────────────────────┼────────────────────┤
│ weights sum to       │      1.0000 │             1.0000 │             0.2744 │
└──────────────────────┴─────────────┴────────────────────┴────────────────────┘
```

`borealis` and `cascade` hold the same fact. Remove either and the answer is unchanged, so
leave-one-out scores both zero — and their credits are silently reassigned to the providers
whose records happened to be unique. That is the failure this project is built around.

To run against the real API instead:

```bash
cp .env.example .env      # add your ANTHROPIC_API_KEY
uv run datagraph demo --live
```

`.env` is read at startup and is gitignored. Anything already exported in your shell wins, so
`ANTHROPIC_API_KEY=... uv run datagraph demo --live` works too.

## The interesting decision

**Paying providers for one answer is a cost-allocation problem, not a ranking problem.**

Frame it as a cooperative game: the *players* are the **providers** whose records reached the
model, the *characteristic function* `v(S)` is how much of the full answer is recoverable from
a subset `S`, and the payout is each player's share of `v(N)`.

Providers rather than records, and that distinction is load-bearing. **The Shapley value is not
replication-proof.** If each record were its own player and a provider's payout were the sum of
its records' shares, a provider could split one record into four identical copies and take a
bigger cut for contributing nothing new — on this demo data that moved one provider from 200 to
326 credits out of 600. Making the provider the player means duplicating a row changes neither
the player set nor any coalition's content, so the payout depends on what you contribute rather
than how you choose to slice it. It is also cheaper: the coalition space is `2^providers`, not
`2^records`.

The obvious engine is **leave-one-out** — "how much worse is the answer without you?" — and
it is the one most implementations reach for. It is not an efficient allocation: the weights
don't sum to `v(N)`, so the payouts don't exhaust the payment and the difference has to be
papered over by normalising. Its failure mode is redundancy, which is exactly what a
marketplace accumulates as providers with overlapping data join. Two providers supplying the
same indispensable fact each score zero, and normalisation hands their share to somebody else.
It fails quietly, which is worse than failing loudly.

The **Shapley value** is the unique allocation satisfying efficiency, symmetry, null player,
and additivity. Efficiency is not a nicety here — it is what makes settlement sound. The
weights sum to `v(N)` exactly, so the escrow is exhausted with no fudge factor, and redundant
providers split the credit rather than both being zeroed.

Exact Shapley is `2^n` evaluations. The default engine estimates it by Monte-Carlo permutation
sampling ([Castro, Gómez & Tejada 2009](https://doi.org/10.1016/j.cor.2008.04.004)): sample
random arrival orders, average each player's marginal contribution. Because every
permutation's marginals telescope to `v(N) − v(∅)`, **the estimator is efficient exactly, not
just in expectation** — sampling moves credit between players but never creates or destroys
any.

Both engines live in [`src/datagraph/attribution.py`](src/datagraph/attribution.py), on
purpose. The contrast is the thing worth reading.

## How it works

```
escrow payment → retrieve records → redact by policy → answer → attribute → settle
```

Every step is ordered deliberately. Payment is escrowed before any work happens. The cohort
floor is checked before the model is ever called. Settlement happens only once attribution has
produced weights that exhaust the payment. Every exit path either settles the escrow or
refunds it in full — a query cannot end with money stranded.

| Module | What it does |
| --- | --- |
| [`attribution.py`](src/datagraph/attribution.py) | The two engines, and `v(S)` |
| [`ledger.py`](src/datagraph/ledger.py) | Double-entry accounts and escrow |
| [`money.py`](src/datagraph/money.py) | Integer credits, largest-remainder apportionment |
| [`policy.py`](src/datagraph/policy.py) | Disclosure levels, redaction, cohort floor |
| [`registry.py`](src/datagraph/registry.py) | Providers, datasets, records, retrieval |
| [`marketplace.py`](src/datagraph/marketplace.py) | The loop above |

### Privacy: what is actually enforced

Each dataset carries a policy assigning every field one of `OPEN` (verbatim), `DERIVED`
(coarsened — numbers banded, dates truncated to the month), or `HIDDEN` (never exposed).
Redaction runs *before* records are assembled into a prompt, so a hidden field is absent from
the object the prompt builder receives. Policies fail closed: an unlisted field is hidden, and
a `DERIVED` value with no safe coarse form is dropped rather than passed through. A query
whose results span fewer than `k` distinct providers is refused before generation, which
blocks the obvious narrowing attack. Retrieval suppresses a provider's duplicate records and
caps how many result slots any one provider can occupy, so padding a dataset cannot crowd
others out and force a refusal.

Raw values never cross the registry boundary. A query returns `SourceView` objects carrying the
disclosed projection and the *names* of suppressed fields — never their values — so a caller
holding a `QueryResult` has no route back to the data the policy removed, on the answered path
or the refused one.

**All of this is enforced by the application, in process, by a trusted operator. None of it is
cryptographic.** An operator with database access reads raw records; a compromised process
bypasses redaction entirely. Real guarantees would require the raw values never to reach this
tier — trusted execution, secure aggregation, or local differential privacy on the provider's
side. That is out of scope here.

### Money

All amounts are integer credits. There are no floats in the money path. Attribution produces
real-valued weights; converting them to payouts uses largest-remainder apportionment, which
guarantees the shares sum to the payment exactly. The ledger refuses a settlement whose
payouts don't exhaust the escrow, so an attribution bug fails loudly instead of quietly losing
money.

## Limitations

These are real, and stated because the project is about measuring honestly.

- **Cost.** Measuring contribution properly means regenerating the answer from many subsets.
  With memoisation the bound is `2^n` model calls per query, where `n` is the number of distinct
  *providers* among the retrieved records — at most 6 by default, so up to 64 generations for
  one answer. There is no cheap version of this that is also correct.
- **Sampling vs. exact.** At small `n`, memoised sampling ends up visiting most of the
  coalition space anyway, so `--engine exact_shapley` costs about the same and has no
  variance. Sampling is the right tool when you deliberately keep the permutation count below
  saturation, or when `n` is too large for `2^n`.
- **`v(S)` is a proxy.** The default similarity is F1 over content words. It measures whether
  the same *content* survived, not whether the same *meaning* did. It is adequate for
  extractive question answering over retrieved records and it is deterministic, which is what
  lets the whole test suite exercise the real attribution path offline. `Similarity` is a
  plug-in point; nothing in the engines changes if you swap in embeddings.
- **Generation is not deterministic.** `temperature` is not accepted on current Claude models,
  so two identical calls can return different text — real noise in a measurement built on
  comparing regenerated answers. Coalition values are memoised, so each subset is generated
  once per query and the comparison *within* a query is self-consistent; across queries it is
  not.
- **Retrieval is deliberately dull.** A lexical matcher, kept simple so nobody mistakes it for
  the interesting part. The engines don't care how sources were selected.
- **In-memory ledger, single process.** Persistence covers the registry, not the accounts.

## Development

```bash
uv sync --extra dev
uv run pytest            # offline, no API key
uv run pytest -m live    # one end-to-end test against the real API
uv run ruff check .
```

The tests worth reading are in [`tests/test_attribution.py`](tests/test_attribution.py): the
engine properties are asserted against synthetic games with known closed-form Shapley values,
so they check the mathematics rather than a model's mood.

## License

TODO — MIT intended; see `LICENSE`.
