# datagraph — specification

`datagraph` is a local reference implementation of a transaction that a pay-per-query RAG data
marketplace would need.

A researcher asks one question and escrows one payment. The system retrieves records from multiple
data providers, applies each provider's disclosure policy, and generates an answer from the disclosed
data. It then measures each provider's contribution to that answer and divides the payment in those
proportions.

This is a proposed transaction model. The repository implements the query, attribution, and
settlement mechanics with synthetic providers and internal credits. It doesn't implement provider
discovery, commercial pricing, access control, verified identity, or an external payment rail.

---

## 1. Query contract

`Marketplace.query(researcher_id, question, payment)` runs one transaction in this order:

1. Move the researcher's integer credits into a query-specific escrow account.
2. Retrieve at most 6 matching records, subject to duplicate and per-provider caps.
3. Convert the records to disclosure-only views.
4. Require at least 3 distinct provider accounts in the result set.
5. Generate the reference answer from all retrieved providers.
6. Regenerate answers from subsets of those providers and measure their contributions.
7. Convert contribution weights to integer payouts and settle the escrow.

The payment is escrowed before retrieval or model work begins. Every path after that point either
settles the full escrow or refunds it. An unexpected exception is re-raised after the refund so the
caller sees the failure without leaving credits stranded.

The returned `QueryResult` contains the query ID, question, answer, disclosed source views,
attribution result, provider weights, payouts, model-call count, and refund status. It never contains
the raw values removed by a disclosure policy.

### Refund conditions

A query is refunded when:

- retrieval returns records from fewer than 3 provider accounts;
- no provider has a positive measured contribution;
- any provider has a negative marginal contribution;
- the model refuses to answer;
- a server-side fallback returns an answer from a different model; or
- any other failure occurs before settlement completes.

---

## 2. Attribution

The attribution problem has 3 parts:

- **Players:** the providers whose records reached the model.
- **Value:** how much of the full answer can be produced from a subset of those providers.
- **Allocation:** the rule that turns those subset values into one weight per provider.

### 2.1 Providers are the players

A provider enters or leaves a subset with all of its retrieved records. Individual records aren't
players.

This keeps a provider's payout from depending on how it divides the same disclosed content into
rows. Retrieval also suppresses exact duplicates from the same provider. Adding copies therefore
doesn't create more players or more visible information.

This doesn't establish provider identity. Two provider accounts are 2 players even if one operator
controls both. Identity verification or a cost to registration would have to sit below this layer.

### 2.2 Valuing a provider subset

For provider set `N` and subset `S`, `v(S)` compares the answer generated from `S` with the reference
answer generated from all of `N`:

```text
v(S) = (similarity(answer(S), answer(N)) - floor) / (1 - floor)
floor = similarity(answer(empty), answer(N))
v(empty) = 0
```

The shipped similarity function is `TokenF1`, F1 over case-folded content-word sets. It measures
lexical overlap, not semantic correctness.

The `floor` removes overlap that exists even when no records are available. Without it, boilerplate
shared by the no-records answer and the reference answer would be treated as provider contribution.
Scores are clamped to `[0, 1]`. For a nondegenerate query, the grand coalition reuses the reference
answer, so `v(N) = 1` exactly. If the reference answer is indistinguishable from the no-records
answer, the whole value function is 0 and the query is refunded.

Each distinct provider subset is generated at most once per query. `CoalitionValue` caches its score,
and every comparison in that query uses the same reference answer.

The cache stabilizes the arithmetic, not the model. With a live model, two subsets are still answered
by separate nondeterministic generations. Their difference can include both a data contribution and
normal generation variance. The offline model is deterministic, so this problem doesn't affect the
demo or offline tests.

### 2.3 Leave-one-out

Leave-one-out measures what disappears when one provider is removed from the full set:

```text
weight(i) = v(N) - v(N without i)
```

It needs one generation per provider, plus the full answer and no-records baseline. The 4-provider
demo therefore uses 6 model calls.

The method works when contributions are independent. Its weights don't generally sum to `v(N)`:

- **Redundant providers make it fall short.** If 2 providers supply the same fact, removing either
  one leaves the fact in the answer, so both can receive a weight of 0. Normalization raises the
  remaining positive shares to fill the escrow.
- **Complementary providers make it overshoot.** If an answer requires every provider, removing any
  one can collapse the answer. Each provider can then receive the full weight, and normalization
  reduces every share to fit the escrow.

Leave-one-out weights can't be negative because `v(N) = 1` and every subset score is at most 1. The
allocator normalizes any positive leave-one-out vector to exhaust the escrow, whether its sum is below
or above `v(N)`. The CLI reports that the attribution was inefficient. Leave-one-out isn't the
default and isn't the recommended settlement method.

### 2.4 Shapley attribution

Shapley attribution measures a provider in different provider orderings. In each ordering, it records
how much the answer value changes when that provider is added.

If 2 providers supply the same fact, each receives credit in orderings where it appears first and no
credit where it appears second. Across every possible ordering, identical providers split the credit.

The contributions in one ordering run from `v(empty)` to `v(N)`, so they add up to `v(N)`. Averaging
multiple orderings preserves that total. Sampling can move credit between providers, but it doesn't
create a shortfall or excess in the combined weight.

The code exposes 2 Shapley engines:

| CLI engine | Calculation | Use |
|---|---|---|
| `shapley` | Samples provider orderings; 2000 by default, with a configurable seed | Default |
| `exact_shapley` | Enumerates every subset and computes the exact Shapley value | Small provider sets and verification |

`exact_shapley` refuses more than 12 providers. The marketplace's default retrieval cap is lower: at
most 6 records can produce at most 6 provider players.

The sampler's seed makes the ordering sequence reproducible. The final payout is reproducible only
when the model and retrieved source set are also deterministic.

### 2.5 Model-call cost

Without caching, `m` sampled orderings over `n` providers would require `m * n` value evaluations.
Caching bounds the number of distinct generated subsets at `2^n`.

With the default cap of 6 retrieved records, one query can involve at most 6 providers and 64
generations. On the 4-provider demo, sampled and exact Shapley both reach all 16 combinations. More
sampled orderings then cost dictionary lookups rather than model calls.

Exact enumeration grows exponentially. Sampling is useful when the provider cap is raised and the
sample count is kept below the full subset space. Fewer samples reduce model calls but add variance to
the split.

### 2.6 Negative contributions

`v(S)` isn't necessarily monotone. Adding a provider can make an answer less similar to the reference
produced by the full set, which gives that provider a negative average marginal contribution.

A negative weight can't become a negative payout. In an efficient Shapley vector, flooring it to 0
increases the sum of the remaining weights above `v(N)`. Normalizing those positive weights would
reduce every other provider's measured share without reporting the change.

The marketplace doesn't do that. If flooring a negative weight changes the total beyond the floating-
point tolerance, it refunds the query instead of settling.

---

## 3. Retrieval and disclosure

The registry stores providers, datasets, disclosure policies, and raw records in SQLite through the
Python standard library.

Retrieval is deterministic lexical matching over disclosed field names and values. Records with no
token overlap with the question are excluded. Ties are resolved by record ID.

Before applying the 6-record result cap, retrieval:

- removes records with disclosed content identical to another record from the same provider; and
- limits each provider to `max(1, max_sources // cohort_floor)` records.

With the defaults, one provider can occupy at most 2 of the 6 result slots. This prevents one padded
dataset from crowding every other provider out of the result set.

### 3.1 Disclosure policies

Every dataset assigns each field one of 3 levels:

| Level | Returned value |
|---|---|
| `OPEN` | Original value |
| `DERIVED` | Coarsened value: numeric band or year-month date |
| `HIDDEN` | Field omitted from the disclosed view |

An unlisted field defaults to `HIDDEN`. Numeric fields use a band width of 10 unless the dataset
specifies another width. A `DERIVED` value with no supported coarse representation is omitted. Policy
projection happens before a prompt or `QueryResult` is built.

The raw record remains in SQLite, and registry methods used by the trusted application can read it.
These policies provide an application boundary, not cryptographic privacy.

### 3.2 Cohort floor

The query must retrieve records from at least 3 distinct provider accounts before any model call is
made. A smaller result set is refunded.

This is a source-diversity rule, not k-anonymity. It counts provider IDs, not people or data subjects.
Multiple accounts controlled by one operator satisfy it, as do records about the same person from
multiple providers.

---

## 4. Money and settlement

Money is represented as integer credits. Attribution weights are floating-point measurements, but no
account balance or transfer amount is a float.

### 4.1 Ledger

The ledger is append-only, double-entry, in memory, and single-process. Every entry's signed postings
must sum to 0. Every account except the external funding boundary must remain non-negative.

The ledger enforces these conditions:

- a researcher can't escrow more credits than their balance;
- an escrow ID can be opened only once;
- every payout must be non-negative;
- settlement payouts must sum exactly to the escrowed amount; and
- an open escrow's account balance must match the amount recorded as held.

### 4.2 Integer allocation

`allocate()` converts non-negative weights to integer credits with largest-remainder apportionment:

1. Convert each proportional quota to an exact rational number.
2. Take the integer floor of each quota.
3. Assign the remaining credits to the largest fractional remainders.
4. Break ties by recipient position; the marketplace supplies providers in sorted ID order.

The returned amounts always sum to the input total. Negative, non-finite, or empty weight vectors are
rejected.

An all-zero vector is accepted by `allocate()` and split equally. The marketplace doesn't use that
fallback: it refunds a query with no positive contribution before calling the allocator.

The allocator treats its inputs as relative weights and normalizes them by their sum. That behavior
doesn't prove an attribution is valid. The default Shapley path supplies weights that already sum to
the value being allocated, and the marketplace separately rejects the negative-contribution case
that would break that condition. Leave-one-out is the documented comparison mode that doesn't have
this property.

---

## 5. Model clients

`ModelClient` is the interface between attribution and answer generation. Two implementations ship.

### 5.1 Offline model

`FakeModel` is the default for the CLI and tests. It returns the sorted union of the disclosed facts
in its source records. It is deterministic and makes redundant and unique contributions directly
observable. It isn't intended to model language quality.

### 5.2 Anthropic API client

`AnthropicModel` is enabled with `--live`. It asks for an answer grounded only in the supplied
records and no longer than 3 sentences. The request uses adaptive thinking at low effort and omits
thinking text from the response used for scoring.

The live client checks 2 response conditions before returning text:

- a model refusal causes a full query refund; and
- a response served by a model other than the requested model causes a refund because generations
  from different models aren't comparable within one attribution run.

The client requests server-side refusal fallbacks by default. If an API deployment rejects that
option, the client retries without it. Internal XML-like tags are removed from returned text before
scoring as a final defensive measure.

Live generations aren't deterministic. Memoization ensures one generation per provider subset, but
it doesn't remove variation between different subsets or between separate runs.

---

## 6. Components

| Component | Module | Responsibility |
|---|---|---|
| Command line | `cli.py` | `demo`, `compare`, and `providers` commands |
| Query transaction | `marketplace.py` | `Marketplace`; escrow, retrieval, disclosure boundary, attribution, settlement, and refunds |
| Attribution | `attribution.py` | `TokenF1`, coalition values, leave-one-out, sampled Shapley, and exact Shapley |
| Models | `models.py` | Deterministic offline model and Anthropic API client |
| Environment | `env.py` | Minimal `.env` loader; exported values take precedence |
| Policy | `policy.py` | Field disclosure and cohort-floor enforcement |
| Registry | `registry.py` | SQLite schema, records, disclosed views, and lexical retrieval |
| Ledger | `ledger.py` | Double-entry accounts and escrow lifecycle |
| Allocation | `money.py` | Exact-rational largest-remainder apportionment |
| Fixture | `sample_data.py` | Synthetic providers and records used by the demo and tests |
| Tokenization | `text.py` | Shared tokens for retrieval and similarity |

Retrieval and attribution are deliberately separate. Replacing lexical retrieval with another
retriever doesn't change the attribution interface. Replacing `TokenF1` with another similarity
function doesn't change either Shapley implementation.

---

## 7. Defaults and interfaces

| Setting | Default |
|---|---:|
| Maximum retrieved records | 6 |
| Minimum provider accounts | 3 |
| Sampled Shapley orderings | 2000 |
| Sampler seed | 0 |
| Exact Shapley guard | 12 providers |
| Demo payment | 1000 credits |
| Live model | `claude-opus-5` |
| Live-model effort | `low` |
| Live-model maximum output | 2048 tokens |

CLI commands:

| Command | Behavior |
|---|---|
| `datagraph demo` | Runs one query with a selected attribution engine |
| `datagraph compare` | Runs the same question under all 3 CLI engines |
| `datagraph providers` | Lists the seeded providers and disclosed fields |

`demo` and `compare` accept `--question`, `--payment`, and `--live`. `demo` also accepts `--engine`
with `shapley`, `exact_shapley`, or `leave_one_out`.

The CLI reads `.env` before it runs a command. `ANTHROPIC_API_KEY` enables live generation, and
`DATAGRAPH_MODEL` overrides the default model. Values already exported in the shell take precedence
over the file. No configuration is needed for the offline commands or tests.

---

## 8. Verified properties and boundaries

The offline test suite verifies:

- sampled and exact Shapley weights exhaust `v(N)`;
- identical contributors receive identical exact weights;
- a provider that never changes the value receives a weight of 0;
- redundant providers receive credit under Shapley and 0 under leave-one-out;
- duplicate rows don't increase a provider's payout;
- model refusal, narrow retrieval, zero contribution, negative contribution, and unexpected failures
  refund the escrow;
- raw suppressed values don't cross the `QueryResult` boundary;
- every ledger entry balances and no internal account becomes negative; and
- integer allocation conserves the full payment across its accepted input range.

The implementation doesn't establish:

- that a provider account represents a unique person or organization;
- that a disclosed answer is factually correct or semantically complete;
- that application-level redaction protects raw data from the registry operator;
- that live payouts are stable across repeated model generations;
- that the local ledger can serve as a durable or distributed payment system; or
- that the exponential attribution cost is practical for an unbounded provider set.

Run the offline verification with:

```bash
uv run --extra dev pytest
```
