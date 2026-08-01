# datagraph — specification

A data marketplace where per-use compensation is **measured**, not assumed.

A provider publishes a dataset under a disclosure policy. A researcher escrows a payment and asks a
question in natural language. The system retrieves candidate records, redacts each one to the
projection its policy permits, and has a model answer from what survives redaction. It then measures
how much each provider's records actually **changed that answer**, and settles the escrow in
proportion to measured contribution.

The interesting part is the measurement, and the fact that the payout is an *allocation* with
provable properties rather than a heuristic ranking.

---

## 1. What this is not

It is worth being blunt about the failure mode this design exists to avoid.

The easy version of this project is a CRUD app: providers table, datasets table, a query endpoint
that returns matching rows, and a payout that splits the fee evenly across whoever got returned.
That version has no reason to exist. Splitting a fee across retrieved rows is not attribution — it
rewards being *retrieved*, which is a property of the search index, not of whether the data was
useful.

Everything below follows from taking the measurement seriously.

---

## 2. The core problem: payout is a cooperative game

A researcher pays `P` credits for one answer. That answer was synthesised from `n` records
belonging to some set of providers. How much does each provider get?

This is not a ranking problem. It is a **cost-allocation problem**, and it has a well-developed
theory. Frame it as a cooperative game:

- **Players** — the records that reached the model.
- **Characteristic function** `v(S)` — the value of the answer obtainable from the subset `S ⊆ N`,
  scored against the answer obtainable from all of `N`. By construction `v(∅) = 0` and `v(N) = 1`.
- **Payout** — each player's share of `v(N)`, scaled to `P` credits.

Two engines are implemented against this frame.

### 2.1 Leave-one-out (`loo`)

The intuitive one, and the one most implementations reach for:

```
φᵢ = v(N) − v(N \ {i})
```

"How much worse is the answer without you?" One regeneration per record, so `n + 1` model calls.

**It is not an efficient allocation.** `Σφᵢ ≠ v(N)` in general, which means the payouts do not
exhaust the payment, and the shortfall or excess has to be papered over by normalising — dividing by
`Σφᵢ`.

The failure is not academic. It is the **redundancy case**, and it is the common case in a data
marketplace, because marketplaces accumulate providers with overlapping data:

> Two providers each supply the same fact. Remove either one and the answer is unchanged, because the
> other still supplies it. So `φ₁ = φ₂ = 0`. Both providers earn nothing, despite the answer
> depending entirely on a fact only they supplied. If *every* record is redundant, `Σφᵢ = 0` and the
> normalisation divides by zero.

A marketplace that pays by leave-one-out systematically underpays exactly the providers whose data
is well-corroborated, and it loses money into a rounding gap it cannot account for.

### 2.2 Shapley value (`shapley`, the default)

The Shapley value is the unique allocation satisfying **efficiency** (payouts sum exactly to the
value being divided), **symmetry** (identical contributors are paid identically), **null player**
(a record that never changes any answer earns exactly zero), and **additivity**:

```
φᵢ = Σ_{S ⊆ N\{i}}  [ |S|! (n−|S|−1)! / n! ] · ( v(S ∪ {i}) − v(S) )
```

Equivalently, and this is the form implemented: the expected marginal contribution of `i` over a
uniformly random arrival order.

Efficiency is not a nicety here — it is the property that makes settlement sound. Because
`Σφᵢ = v(N)` exactly, the escrow is exhausted with no normalisation fudge, and the redundancy case
resolves correctly: two providers supplying the same indispensable fact split its credit, rather
than both being zeroed.

Exact computation is `2ⁿ` coalitions. The implementation uses **Monte-Carlo permutation sampling**
(Castro, Gómez & Tejada 2009, *Computers & Operations Research* 36(5), 1726–1730): sample random
permutations, walk each one accumulating marginal contributions, average. This is an unbiased
estimator of the Shapley value. It is seeded, so a given query is reproducible.

### 2.3 Making it affordable

Naively, sampling `m` permutations over `n` records costs `m·n` model calls. Two things cut it:

- **Coalition memoisation.** `v(S)` depends only on the *set* `S`, not on the permutation that
  produced it. Results are cached on `frozenset(record_ids)`. The number of *distinct* coalitions
  reachable is at most `2ⁿ`, so the cache saturates and sampling more permutations becomes nearly
  free.
- **A bounded player set.** Retrieval returns at most `max_sources` records — **6** by default,
  which caps a live query at 64 generations. `n` is the only real lever on cost; the estimator is
  not where savings come from.

That second point cuts the other way too, and the implementation says so rather than pretending
sampling is a free win. **Once sampling has visited most of the coalition space, it has paid for the
whole space anyway** — at small `n`, `exact_shapley` costs about the same and has no variance.
Sampling earns its place when the permutation count is deliberately held below saturation, or when
`n` is too large for `2ⁿ`. The permutation default is correspondingly high (2000), because below
saturation the marginal permutation costs a dictionary lookup rather than a model call, and
sampling sparsely bought nothing but noise.

The cost is real and the README states it plainly rather than hiding it. `loo` remains available as
the cheap engine, documented with the defect above, because showing the two side by side is more
useful than shipping only the right one.

---

## 3. Scoring an answer: `v(S)`

`v(S)` regenerates an answer from the subset `S` and scores it against the reference answer built
from all of `N`:

```
v(S) = ( similarity(answer(S), answer(N)) − floor ) / ( 1 − floor )     with  v(∅) := 0
where   floor = similarity( answer(∅), answer(N) )
```

**The `floor` term is load-bearing, and it was not in the first draft of this design.** Every answer
shares boilerplate with every other — "the records show…" against "the records do not support an
answer" — and that shared vocabulary puts a constant, non-zero floor under the raw similarity. It is
not evidence that anything contributed. Without subtracting it, a source that changes nothing scores
well above zero and inherits the boilerplate as earnings, which breaks the null-player property that
makes the whole allocation meaningful. Rescaling against the no-source answer costs one extra
generation per query and restores `v(∅) = 0`, `v(N) = 1` exactly. The null-player test is what caught
this.

`similarity` is a pluggable protocol so the choice is not baked in:

- **`TokenF1`** (default) — F1 over content-word token sets, stopworded and case-folded.
  Deterministic, dependency-free, and computable offline, which is what lets the entire test suite
  exercise the *real* attribution path with no API key.
- **Embedding-backed cosine** — available when an API key is present, for semantic rather than
  lexical agreement.

This is a deliberate, stated limitation. Lexical F1 measures whether the same content survived, not
whether the *meaning* did. It is adequate for the extractive question-answering this system does —
answers are grounded in retrieved records, so contribution shows up as content appearing or
disappearing — and it is honest about being a proxy. The interface exists so a reader can swap in
something better without touching the attribution engines.

---

## 4. Privacy gating

The reference implementation for this idea hashed sensitive rows, stored the hash on-chain, and
wrote the plaintext to an ordinary database. That is a commitment scheme, not a privacy mechanism —
anyone with database access reads everything. This design does not repeat that, and does not claim
more than it does.

Each dataset carries a **`DisclosurePolicy`** assigning every field one of:

| Level | Meaning |
| --- | --- |
| `OPEN` | may be exposed verbatim |
| `DERIVED` | may be exposed only coarsened — numeric values bucketed into bands, dates truncated to month |
| `HIDDEN` | never leaves the provider's record under any query |

Two rules enforce it:

1. **Redaction precedes generation.** Records are projected through their policy *before* they are
   assembled into a prompt. A `HIDDEN` field is absent from the object the prompt builder receives,
   so no prompt can contain it and no model output can leak it. This is a structural guarantee about
   the data path, not a request to the model to behave.
2. **Cohort floor.** A query whose redacted result set spans fewer than `k` distinct providers
   (default `k = 3`) is refused before any generation happens. Without this, a researcher narrows a
   query until it resolves to one person and reads that person's record out of the answer.

**What this is and is not.** These are enforced *by the application*, in process, by a trusted
operator. Nothing here is cryptographic. An operator with database access reads raw records; a
compromised process bypasses redaction. Real guarantees would need the raw data never to reach this
tier at all — trusted execution, secure aggregation, or local differential privacy on the provider
side. That is out of scope, and the README says so in those words rather than implying otherwise.

---

## 5. Money

All amounts are integer **credits** (minor units). There are no floating-point values anywhere in
the money path. Attribution weights are real-valued; the conversion to integer payouts is where
exactness is enforced.

- **Escrow.** A query deposits `P` credits into escrow before retrieval. Settlement releases them.
  A refused query (cohort floor, no results) refunds in full. Escrow is never partially stranded.
- **Double-entry.** Every movement is a balanced set of postings; the ledger asserts that debits
  equal credits on every write.
- **Largest-remainder allocation.** Real-valued weights become integer payouts by Hamilton's
  method: floor each share, then distribute the remaining credits to the largest fractional
  remainders, ties broken deterministically by record id. This guarantees `Σ payouts == P` exactly —
  no credits created, none lost to rounding.

Invariants, asserted as property-based tests:

- Total credits in the system are conserved across any sequence of operations.
- No account goes negative.
- Every escrow reaches a terminal state — fully settled or fully refunded.
- `Σ payouts == escrowed amount`, for any weight vector including all-zero.

---

## 6. Components

```
src/datagraph/
  money.py          Credits, largest-remainder allocation
  ledger.py         double-entry accounts, escrow, invariant assertions
  policy.py         DisclosurePolicy, field redaction, cohort floor
  registry.py       providers, datasets, records, retrieval, SQLite persistence
  models.py         ModelClient protocol; AnthropicModel; deterministic FakeModel
  attribution.py    Similarity, v(S), and both engines
  marketplace.py    orchestration: escrow -> retrieve -> redact -> answer -> attribute -> settle
  sample_data.py    synthetic demo data
  text.py           shared tokenisation
  cli.py            demo, compare, providers
```

Two things collapsed relative to the first draft of this plan, both in the direction of less
structure. Retrieval lives in `registry.py`, because it is a query against the store and pretending
otherwise added a module without adding a concept. **Both attribution engines live in one file**,
because the contrast between them is the lesson — splitting them across `loo.py` and `shapley.py`
would hide the one thing a reader should see side by side.

**Storage** is SQLite through the standard library — no service to run, so the quickstart is one
command. There is deliberately **no Docker**: with a local database and a hosted API there is
nothing to orchestrate, and a compose file would be ceremony that makes the project look harder to
run than it is.

**Generation** is the Anthropic API (`claude-opus-5`); `FakeModel` is a deterministic stand-in that
composes answers from the facts present in its sources, which makes leave-one-out and Shapley
produce *meaningful, assertable* differences offline. The offline suite therefore tests the real
attribution code rather than mocking past it.

Answers are generated with thinking disabled at low effort. The task is short extractive work
repeated up to `2^n` times per query, so depth buys nothing and costs a great deal. Requests opt
into server-side refusal fallbacks by default, and fall back to a plain request once if that beta
is not available to the caller's organisation.

---

## 7. Failure modes this design is built against

Data marketplaces that promise per-use compensation tend to share the same three gaps, and each one
here is a direct response.

**Attribution declared but not computed.** It is easy to design a schema with a
`contribution_by_provider` table and never write the code that populates it — the API looks
complete, the payout path reads plausibly, and the numbers are always zero or uniform. The guard is
that attribution here has properties that are *asserted*, not just documented: efficiency, symmetry,
and null-player are tested against closed-form games, so a stubbed engine fails the suite.

**Payouts that don't reconcile.** Splitting a payment with floating-point division and rounding for
display loses money on every query, invisibly. Integer credits plus largest-remainder apportionment
make the shares sum exactly, and the ledger *refuses* a settlement that doesn't exhaust its escrow,
so an attribution bug surfaces as a loud failure rather than a slow leak.

**Privacy claimed by hashing.** Storing a hash alongside the plaintext is a commitment scheme, not a
privacy mechanism — anyone with database access still reads everything. Redaction here is structural
(the suppressed value is never placed in the object a prompt is built from), and the README states
plainly that this is application-enforced and not cryptographic, rather than implying more.

The **domain shape** — providers with periodic records, researchers paying per query, payout
proportional to use — is kept because it makes the attribution problem concrete rather than
abstract.

---

## 8. Decisions recorded

| Decision | Choice | Why |
| --- | --- | --- |
| Attribution default | Shapley (sampled) | Efficiency and null-player are what make settlement sound; LOO has neither |
| LOO retained | Yes, as an option | The contrast is the most instructive thing in the repo |
| `v(S)` similarity | Pluggable; lexical F1 default | Lets the real attribution path run offline and be tested |
| Money type | Integer credits | Float division is how the reference lost money |
| Weight → payout | Largest remainder | Only method here that guarantees exact exhaustion |
| Storage | SQLite, stdlib | One-command quickstart beats a more "correct" service dependency |
| Containers | None | Nothing to orchestrate; a compose file would only make it look harder to run |
| Module layout | Both engines in one file | The contrast is the lesson and belongs on one screen |
| `v(S)` calibration | Rescale against the no-source answer | Otherwise shared boilerplate is paid out as contribution |
| Model settings | Thinking off, effort low | Short extractive work repeated up to 2ⁿ times; depth buys nothing |
| Language | Python | The audience reads Python; the attribution logic is the point and must be legible |
| Privacy claim | Application-enforced, stated as such | Overclaiming here is the specific dishonesty this design is reacting to |

### Where the build changed the plan

Three things in this document were revised *after* being tested rather than before, and are called
out because the reasoning is more useful than the conclusion:

1. **The `v(S)` floor** (§3) — added because the null-player test failed. Shared boilerplate was
   being paid out as contribution.
2. **Permutation count and source cap** (§2.3) — the first draft sampled sparsely to save money.
   Memoisation means it was saving nothing and paying for it in variance.
3. **What leave-one-out actually does on redundant data** (§2.1) — the expectation was that it would
   collapse loudly, all weights zero, and force a refund. On realistic data it does something worse:
   it zeroes only the corroborated providers and normalisation silently reassigns their credits to
   whoever happened to be unique. The integration test asserts the quiet failure, because that is
   the one a marketplace would actually ship.
