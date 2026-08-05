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

- **Players** — the **providers** whose records reached the model. Providers, not records:
  the Shapley value is not replication-proof, so per-record players let a provider inflate its
  cut by splitting one record into several identical rows (measured: 200 → 326 credits out of
  600 by cloning a row four times). A coalition names providers and is answered from all of
  that provider's retrieved records at once, which makes the payout invariant to row count —
  and shrinks the coalition space from `2^records` to `2^providers`.
- **Characteristic function** `v(S)` — the value of the answer obtainable from the subset `S ⊆ N`,
  scored against the answer obtainable from all of `N`. By construction `v(∅) = 0` and `v(N) = 1`.
- **Payout** — each player's share of `v(N)`, scaled to `P` credits.

Two engines are implemented against this frame.

### 2.1 Leave-one-out (`loo`)

The intuitive one, and the one most implementations reach for:

```
φᵢ = v(N) − v(N \ {i})
```

"How much worse is the answer without you?" One regeneration per player, so `n + 1` model calls.

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
  produced it. Results are cached on `frozenset(provider_ids)`. The number of *distinct* coalitions
  reachable is at most `2ⁿ`, so the cache saturates and sampling more permutations becomes nearly
  free.
- **A bounded player set.** Retrieval returns at most `max_sources` records — **6** by default — and
  `n` is the number of *distinct providers* among them, so a live query is capped at 64 generations.
  `n` is the only real lever on cost; the estimator is not where savings come from.

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

### 2.4 Where this sits relative to published work

None of the framing is new, and pretending otherwise would be the fastest way to lose a reader who
knows the area.

**Data Shapley** (Ghorbani & Zou, ICML 2019, [arXiv:1904.02868](https://arxiv.org/abs/1904.02868))
applies the Shapley value to *training* data, with `v(S)` the performance of a model trained on `S`,
and motivates it with the same premise this project starts from: that people should be compensated
for the data they generate. It also establishes, with experiments, that leave-one-out is the weaker
measure. What changes here is the characteristic function and the player set — `v(S)` is the
recoverable content of one generated answer rather than validation accuracy, and players are
providers rather than individual training points, which is what makes a payout invariant to how a
provider slices its rows.

**ContextCite** (Cohen-Wang et al., NeurIPS 2024,
[arXiv:2409.00729](https://arxiv.org/abs/2409.00729)) is the closer neighbour in method: it ablates
context sources and fits a sparse linear surrogate over the ablations, reporting roughly 32 ablations
even for contexts with hundreds of sources. For deciding *which source supports this sentence* it is
strictly the better tool, and much cheaper than `2ⁿ`. It is not an allocation. LASSO drives most
coefficients to exactly zero, and regression weights carry no constraint that they sum to the value
being divided — both are features for attribution and disqualifying for settlement, where zero means
a contributor is unpaid and an unconstrained sum means the escrow does not balance.

The cheap-and-approximate direction is real, though, and a production system with large `n` would
have to take it. The honest statement of scope is that this project buys an exact allocation at
exponential cost and caps `n` at 6 to afford it.

### 2.5 The demo data is constructed

`sample_data.py` gives `borealis` and `cascade` identical disclosed records so that the redundancy
case appears in a five-record fixture. That is staging, and it is stated here, in the module
docstring, and in the README.

It buys a failure that is visible on one screen and reproducible — the offline model is deterministic
and the sampler is seeded, so `compare` prints identical figures on any machine, which is also what
lets the integration tests assert exact payouts. It costs generality: this is the minimal instance of
redundancy rather than a naturally occurring one. The behaviour it demonstrates does not depend on
the staging. Leave-one-out discounts *any* provider whose contribution is corroborated, in proportion
to how completely somebody else covers it; total duplication is simply the case where the discount
reaches 100% and the arithmetic is legible.

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

`similarity` is a pluggable protocol so the choice is not baked in. **One implementation ships**:
`TokenF1`, F1 over content-word token sets, stopworded and case-folded. It is deterministic,
dependency-free and computable offline, which is what lets the entire test suite exercise the *real*
attribution path with no API key. An embedding-backed cosine would slot in behind the same protocol
and is the obvious next one to write; it is not written, and the engines are indifferent either way.

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
  remainders, ties broken by position in the recipient list, which the marketplace builds in sorted
  provider order. This guarantees `Σ payouts == P` exactly — no credits created, none lost to
  rounding.

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
| Attribution players | Providers, not records | Shapley is not replication-proof; per-record players are gameable by cloning |
| Query boundary | Disclosure-only `SourceView` | "Raw values stay in the store" is only true if no object carrying them crosses the boundary |
| Failure handling | Refund guard around the whole query | Any exception between escrow and settlement would otherwise debit the payer and strand the credits |
| Apportionment arithmetic | Exact rationals | Floats silently lost credits at the edges of the accepted input range |
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

An adversarial review then found five more, all reproduced before being fixed, and the first two
were design errors rather than slips:

4. **Per-record players were gameable** (§2) — cloning a row raised a provider's take by 63% for no
   new information. Players are providers now. The same attack then reappeared as a *crowd-out*:
   cloned rows filled the retrieval cap and tripped the cohort floor, turning payout theft into a
   denial of service, so retrieval also suppresses a provider's duplicates and caps its share of
   the result slots.
5. **`QueryResult` handed back raw records** — including on the cohort-floor refusal, where the
   whole point was that the data was too narrow to expose. Redaction was careful about the *prompt*
   and careless about the *return value*. One of my own tests asserted the leak was correct.
6. **Only `ModelRefusal` was caught** — a timeout debited the researcher and left the credits in an
   escrow nothing would settle. The whole query now runs under a refund guard, `BaseException`
   included, because ctrl-C is exactly when you least want money stranded.
7. **A rejected insert poisoned the store** — the row was committed before the projection that
   validated it, so one non-finite value made every later read raise. Validation moved ahead of the
   commit.
8. **Float apportionment lost credits** — exact within the range the property test covered, wrong
   outside it (`allocate(10**18, [1,2,3])` was 29 credits short). Rational arithmetic now, and the
   property test runs to the contract rather than to the comfortable range.

Rehearsing a cold start — fresh clone, empty dependency cache, no key, following the README
literally rather than from memory — found one more:

9. **The documented `.env` file was never read** (§6) — the README told the reader to copy
   `.env.example` and put their key in it, and nothing in the project ever loaded that file.
   Obeying the instructions produced an SDK-internal `TypeError` about resolving an authentication
   method. There is a loader now; a missing key is reported as an instruction rather than a stack
   trace; and `.env.example` no longer documents a database path that nothing reads. The lesson is
   narrow but general: the offline path was tested exhaustively because it was cheap to test, and
   the one step that needed configuration was the one step never run from a clean machine.
