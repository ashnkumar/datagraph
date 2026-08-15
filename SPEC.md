# datagraph — design notes

A data marketplace where per-use compensation is **measured**, not assumed.

A provider publishes a dataset under a disclosure policy. A researcher escrows a payment and asks a
question in natural language. The system retrieves candidate records, redacts each one to the
projection its policy permits, and has a model answer from what survives redaction. It then measures
how much each provider's records actually **changed that answer**, and settles the escrow in
proportion to measured contribution.

The interesting part is the measurement, and the fact that the shares are guaranteed to add up to
the payment rather than being scaled to fit.

---

## 1. What this is not

It is worth being blunt about the failure mode this design exists to avoid.

The easy version of this project is a CRUD app: providers table, datasets table, a query endpoint
that returns matching rows, and a payout that splits the fee evenly across whoever got returned.
That version has no reason to exist. Splitting a fee across retrieved rows is not attribution — it
rewards being *retrieved*, which is a property of the search index, not of whether the data was
useful.

Everything below follows from taking the measurement seriously.

It is also not a payment system. Settlement here is a double-entry ledger inside one process and
credits are integers; there is no chain, no wallet and no external rail. The design assumes such a
rail exists — the shape it assumes is a network where providers need not trust an operator, payouts
execute without one, and identities are pseudonymous — and deliberately declines to pick one.
`ledger.py` is where a real one would attach, and nothing above it would change. What the assumption
costs is recorded in §7, because pseudonymous identity is exactly what makes the false-name attack
cheap.

---

## 2. Splitting the payment

A researcher pays `P` credits for one answer, synthesized from `n` records belonging to some set of
providers. How much does each provider get?

This is not a ranking problem. The output has to be a *division* — shares that add up to `P`,
because there is an escrow that has to come out empty. Three things have to be settled before any
code gets written:

- **Who the players are.** The **providers** whose records reached the model, not the records
  themselves. Per-record players let a provider inflate its cut by splitting one record into several
  identical rows: reconstructing that design on the demo fixture, cloning one of `delta`'s two
  records four times took it from `446` to `612` credits of 1000, a **37%** raise for publishing
  nothing new. A combination names providers and is answered from all of that provider's retrieved
  records at once, which makes the payout invariant to row count — and shrinks the search space from
  `2^records` to `2^providers`. (In the code a combination is a `frozenset` called a *coalition*.)
- **What a subset is worth.** `v(S)`, the value of the answer obtainable from the subset `S ⊆ N`,
  scored against the answer obtainable from all of `N`. By construction `v(∅) = 0` and `v(N) = 1`.
  §3 has the definition.
- **How a subset's value becomes a share.** Two engines are implemented, below.

### 2.1 Leave-one-out (`loo`), and why it is not the default

The intuitive one, and the one most implementations reach for:

```
φᵢ = v(N) − v(N \ {i})
```

"How much worse is the answer without you?" One regeneration per player, so `n + 1` model calls.

**The shares do not add up.** `Σφᵢ ≠ v(N)` in general, which means the payouts do not exhaust the
payment, and the shortfall or excess has to be papered over by normalizing — dividing by `Σφᵢ`.

The failure is not academic. It is the **redundancy case**, and it is the common case in a data
marketplace, because marketplaces accumulate providers with overlapping data:

> Two providers each supply the same fact. Remove either one and the answer is unchanged, because the
> other still supplies it. So `φ₁ = φ₂ = 0`. Both providers earn nothing, despite the answer
> depending entirely on a fact only they supplied. If *every* record is redundant, `Σφᵢ = 0` and the
> normalization divides by zero.

A marketplace that pays by leave-one-out systematically underpays exactly the providers whose data
is well-corroborated, and it loses money into a rounding gap it cannot account for.

### 2.2 The engine that ships (`shapley`, the default)

Instead of asking what is lost without a provider, walk every order in which the providers could
have arrived and ask what each one added when it arrived. Average those contributions.

That one change fixes both problems above:

- **The shares add up exactly.** Each order is a chain of differences running from nothing to
  everything, so every individual order sums to `v(N)` on its own — and therefore so does any
  average of orders. The escrow is exhausted with no normalization step.
- **Redundancy resolves correctly.** A provider that arrives before its duplicate adds a lot; one
  that arrives after it adds nothing. Both happen equally often across all orders, so two providers
  supplying the same indispensable fact split its credit rather than both being zeroed.

Three properties follow, and all three are asserted against closed-form games in
`tests/test_attribution.py` rather than merely documented: the shares exhaust the payment,
identical contributors are paid identically, and a contributor that never changes any answer is
paid exactly zero. A stubbed engine fails those tests.

Enumerating every combination is `2ⁿ` generations. The default engine instead samples random
arrival orders and averages — the standard permutation-sampling approach, cited in
`attribution.py`'s module docstring. It is seeded, so a given query is reproducible. Because every
individual order is exact, sampling moves credit *between* providers without creating or destroying
any: noisy in the split, exact in the total.

### 2.3 What it costs

Naively, sampling `m` orders over `n` providers costs `m·n` model calls. Two things cut it:

- **Memoization.** `v(S)` depends only on the *set* `S`, not on the order that produced it. Results
  are cached on `frozenset(provider_ids)`. The number of *distinct* subsets reachable is at most
  `2ⁿ`, so the cache saturates and sampling more orders becomes nearly free.
- **A bounded player set.** Retrieval returns at most `max_sources` records — **6** by default — and
  `n` is the number of *distinct providers* among them, so a live query is capped at 64 generations.
  `n` is the only real lever on cost; the estimator is not where savings come from.

That second point cuts the other way too, and the implementation says so rather than pretending
sampling is a free win. **Once sampling has visited most of the space, it has paid for the whole
space anyway** — at small `n`, `exact_shapley` costs about the same and has no variance. Sampling
earns its place when the order count is deliberately held below saturation, or when `n` is too large
for `2ⁿ`. The default order count is correspondingly high (2000), because below saturation the
marginal order costs a dictionary lookup rather than a model call, and sampling sparsely bought
nothing but noise.

The cost is real and the README states it plainly rather than hiding it. `loo` remains available as
the cheap engine, documented with the defect above, because showing the two side by side is more
useful than shipping only the right one.

### 2.4 The demo data is constructed

`sample_data.py` gives `borealis` and `cascade` identical disclosed records so that the redundancy
case appears in a five-record fixture. That is staging, and it is stated here, in the module
docstring, and in the README.

It buys a failure that is visible on one screen and reproducible — the offline model is deterministic
and the sampler is seeded, so `compare` prints identical figures on any machine, which is also what
lets the integration tests assert exact payouts. It costs generality: this is the minimal instance of
redundancy rather than a naturally occurring one. The behavior it demonstrates does not depend on
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
well above zero and inherits the boilerplate as earnings, which breaks the guarantee that a
contributor of nothing is paid nothing. Rescaling against the no-source answer costs one extra
generation per query and restores `v(∅) = 0`, `v(N) = 1` exactly. That test is what caught this.

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
  text.py           shared tokenization
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

**Thinking stays enabled, at `low` effort.** An earlier version of this document specified thinking
*disabled*, on the reasoning that the task is short extractive work repeated up to `2ⁿ` times per
query and depth buys nothing. That was wrong, for a reason specific to this project. Anthropic's
documentation recommends the opposite trade — "for most tasks, thinking enabled at `low` effort
performs better than thinking disabled at similar cost" — and, more decisively, disabling thinking
is what causes the model to "emit `<thinking>` tags or other internal XML tags into its visible
response." Here the visible response *is* the measuring instrument, so a leaked tag is spurious
tokens inside a similarity score and a wrong payout. Effort is the cost lever instead, and
`display: "omitted"` is set explicitly — already the default on this model — so thinking text can
never reach the scored string.

Requests opt into server-side refusal fallbacks by default. The feature is in beta and unavailable
on the Batches API and the cloud-provider platforms, so if the request is rejected for it, the first
call drops the parameter and carries on unprotected rather than failing the whole query.

---

## 7. Failure modes this design is built against

Data marketplaces that promise per-use compensation tend to share the same three gaps, and each one
here is a direct response.

**Attribution declared but not computed.** It is easy to design a schema with a
`contribution_by_provider` table and never write the code that populates it — the API looks
complete, the payout path reads plausibly, and the numbers are always zero or uniform. The guard is
that the properties in §2.2 are *asserted* against closed-form games, not just documented, so a
stubbed engine fails the suite.

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

### And one it is not built against

Moving the players from records to providers (§2) makes a payout invariant to how a provider slices
its rows. It does nothing about a provider registering **twice**. Splitting `delta`'s two records
across two provider accounts raises its combined take from 365 to 446 credits out of 1000 — measured
on the demo fixture, +22% for no new information.

This is not a bug in the implementation. No split computed over self-declared identities can resist
this while a second identity is free. Closing it requires something underneath the measurement —
identity attestation, staking, or a cost to registering. The honest scope statement is that this
project solves the split given a trustworthy player set, and does not establish one. The README says
so in its limitations rather than leaving a reader to find it.

---

## 8. Decisions recorded

| Decision | Choice | Why |
| --- | --- | --- |
| Attribution default | Shapley (sampled) | Shares exhaust the payment and a null contributor is paid zero; LOO has neither |
| LOO retained | Yes, as an option | The contrast is the most instructive thing in the repo |
| `v(S)` similarity | Pluggable; lexical F1 default | Lets the real attribution path run offline and be tested |
| Money type | Integer credits | Float division is how the reference lost money |
| Weight → payout | Largest remainder | Only method here that guarantees exact exhaustion |
| Storage | SQLite, stdlib | One-command quickstart beats a more "correct" service dependency |
| Containers | None | Nothing to orchestrate; a compose file would only make it look harder to run |
| Module layout | Both engines in one file | The contrast is the lesson and belongs on one screen |
| `v(S)` calibration | Rescale against the no-source answer | Otherwise shared boilerplate is paid out as contribution |
| Model settings | Thinking adaptive, effort low | Disabling it leaks internal tags into the string this project scores (§6) |
| Attribution players | Providers, not records | Per-record players are gameable by cloning a row |
| Query boundary | Disclosure-only `SourceView` | "Raw values stay in the store" is only true if no object carrying them crosses the boundary |
| Failure handling | Refund guard around the whole query | Any exception between escrow and settlement would otherwise debit the payer and strand the credits |
| Apportionment arithmetic | Exact rationals | Floats silently lost credits at the edges of the accepted input range |
| Language | Python | The audience reads Python; the attribution logic is the point and must be legible |
| Privacy claim | Application-enforced, stated as such | Overclaiming here is the specific dishonesty this design is reacting to |

### Where the build changed the plan

Three things in this document were revised *after* being tested rather than before, and are called
out because the reasoning is more useful than the conclusion:

1. **The `v(S)` floor** (§3) — added because the null-contributor test failed. Shared boilerplate was
   being paid out as contribution.
2. **Order count and source cap** (§2.3) — the first draft sampled sparsely to save money.
   Memoization means it was saving nothing and paying for it in variance.
3. **What leave-one-out actually does on redundant data** (§2.1) — the expectation was that it would
   collapse loudly, all weights zero, and force a refund. On realistic data it does something worse:
   it zeroes only the corroborated providers and normalization silently reassigns their credits to
   whoever happened to be unique. The integration test asserts the quiet failure, because that is
   the one a marketplace would actually ship.

An adversarial review then found five more, all reproduced before being fixed, and the first two
were design errors rather than slips:

4. **Per-record players were gameable** (§2) — cloning a row raised a provider's take by 37% for no
   new information. Players are providers now. The same attack then reappeared as a *crowd-out*:
   cloned rows filled the retrieval cap and tripped the cohort floor, turning payout theft into a
   denial of service, so retrieval also suppresses a provider's duplicates and caps its share of
   the result slots.
5. **`QueryResult` handed back raw records** — including on the cohort-floor refusal, where the
   whole point was that the data was too narrow to expose. Redaction was careful about the *prompt*
   and careless about the *return value*. One of the project's own tests asserted the leak was
   correct.
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
