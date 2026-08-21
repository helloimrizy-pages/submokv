# Sub-MoKV Milestone 3b decision rules

Written 2026-08-21, **before** Tasks B, C, and D were run, against commit
`d043b85`. Nothing below may be changed after seeing a classification. If a rule
turns out to be unmeasurable as stated, the amendment is recorded at the bottom
with its date and its reason, and the original text stays.

## What is on trial

Milestone 3b decides whether Sub-MoKV is a **method paper**, a **null paper**,
or **dead**. The solver is on hold. Milestone 4 does not start until this file
says which branch the evidence landed in.

## What is already settled

Taken as given from the runs behind
`results/noise_floor__20260820T154918Z__b54b8f0e.json`,
`results/diagnostic_0__merged__20260820T210527Z__75d4665a.json`, and
`results/diagnostic_2__20260820T212022Z__75d4665a.json`. These are not
re-derived and not re-litigated.

| Quantity | Value |
| --- | --- |
| Paired weight-probe ΔPPL stdev over calibration subsamples | 0.0326 PPL |
| Paired KV-probe ΔPPL stdev over calibration subsamples | 0.0049 PPL |
| Reference PPL spread across four subsamples | 6.20 – 8.13 (pairing carries the result) |
| Weight ladder, mean ΔPPL vs 16-bit | 2 bit 0.936, 3 bit 0.118, 4 bit 0.023, 8 bit 0.0003 |
| KV ladder, mean ΔPPL vs full cache | r=0.25 0.054, r=0.50 0.0116, r=0.75 0.0005 |
| Real-ladder steps | 3→4 buys ≈0.095 (≈3 noise units); 4→8 buys 0.023 (below noise); 8→16 buys nothing measurable |
| Per-expert probes, 16 single experts at 2 bit | −0.0032 to +0.0287, every one below the 0.0326 weight floor |
| Diagnostic 2 interaction, perplexity | −0.0252 on a joint utility of 2.151 = 1.2%, below the weight floor |

Consequences that bind this milestone:

* Both utility ladders carry signal only on their bottom step.
* **Expert-level allocation carries no signal.** No time is spent in Milestone
  3b on per-expert granularity.
* The second Diagnostic 2 pairing returned exactly 0.0 because `weight_only`
  had zero utility and therefore tested nothing. That pairing is void, not a
  measurement of separability.

## Definitions the rules below are stated in

**Second-order difference.** For a target upgrade `j` measured with one
conditioning component low (`S_A`) and high (`S_B`):

```
D = [F(S_A + j) − F(S_A)] − [F(S_B + j) − F(S_B)]
```

**Second-order noise floor `σ₂`.** The stdev of `D` itself across calibration
subsamples, measured per modality (Task B). It is *not* the stdev of a single
ΔPPL. Weight-side and KV-side scales differ by roughly a factor of seven, so
`σ₂` is a per-modality number, never one global constant.

**Epsilon.** `ε(modality) = σ₂(modality)`, set from measurement in Task B and
frozen before Task C runs. Every classification in every result record carries
the `ε` it was compared against, in the same record.
*Superseded by Amendment 1 (2026-08-21), below, which was recorded before any
Task C classification existed.*

**Resolved cell.** A cell with `|D| > ε`. A cell with `|D| ≤ ε` is *unresolved*:
it is evidence of nothing, and in particular is **not** evidence of
submodularity. The count of resolved cells is reported separately from the
count of submodular classifications, always.
*Extended by Amendment 2 (2026-08-21), below: resolution now requires a
statistical gate **and** an effect gate, and the resolved rate is reported as a
band across the `σ₂` interval.*

**Interaction term (Diagnostic 2).** `I = F(W ∪ K) − F(W) − F(K)`, reported both
absolutely and as `I / F(W ∪ K)`, with its own noise floor beside it.

## The three branches

### Method paper

Survives **only** if, under the retrieval metric of Task D:

> the interaction term exceeds **three times its noise floor** on a **majority
> of pairings**, at **both** budget fractions.

Operationally: `|I| > 3·σ_I` for more than half the pairings run at budget
fraction 1, **and** for more than half the pairings run at budget fraction 2.
Both fractions must clear the bar on their own. A majority pooled across the two
fractions does not count.

### Null paper

If the interaction stays inside noise under **both** metrics — perplexity and
the retrieval metric. The claim is then three negative results with clean
measurement behind them:

1. Weight precision and KV retention are **separable** on OLMoE.
2. Both utility ladders **collapse to their bottom step**.
3. **Expert-level allocation carries no signal** above measurement noise.

### Dead

If Task C shows that **most cells cannot be classified at all** even after the
epsilon fix — operationally, resolved cells (`|D| > ε`) are **under half** of
the cells run — meaning the ladder above its bottom step has no measurable
structure in either axis.

*Amended by Amendment 2: a cell counts as resolved only if it passes both the
statistical and the effect gate, and the rate is called on the `σ₂` point
estimate with the two interval bounds reported beside it.*

## Precedence, and one ambiguity in the spec

Task C is measured under perplexity. Task D exists precisely because perplexity
is the metric H2O and SnapKV avoid, so a Dead reading from Task C is a statement
about the perplexity ladder, not about the model. The two rules can therefore
both fire.

Resolution used here, flagged for the user at the Task A stop:

* A Dead reading from Task C is recorded as **Dead under perplexity** and does
  **not** by itself cancel Task D. Task D changes the metric, which is the
  variable Task C could not vary.
* Dead becomes the final branch only if Task D's retrieval ladder is also
  unresolvable against its own measured floor.
* If Task C is Dead and the user prefers to stop there, that is the user's call
  and Task D is not run.

## Ground rules binding every run in this milestone

1. **No number goes in a table unless a run produced it.**
2. Every result record keeps the config, the git commit, the seed, the
   subsample list, and **the noise floor it was compared against**.
3. **Epsilon is not tuned after seeing the classifications.** Task B fixes it;
   Tasks C and D consume it.
4. Every utility used for a decision is deterministic, memoized on the
   allocation hash, and read from a calibration split disjoint from anything
   used for reporting.
5. A count reported as a headline and a count reported as secondary are labelled
   with the test that produced them, so the two can never be read as
   contradicting each other.
6. If a task's result makes a later task pointless, the run stops and says so
   rather than completing for the sake of completeness.

## Amendments

### Amendment 1 — epsilon is the standard error of the quantity classified

**Date:** 2026-08-21. **Recorded before:** Task B was run, therefore before any
`σ₂` was measured and before any Task C cell was classified. Nothing in this
amendment was chosen with a classification visible.

**Original rule:** `ε(modality) = σ₂(modality)`, where `σ₂` is the stdev of `D`
across single calibration subsamples.

**Amended rule:**

> `ε(modality) = σ₂(modality) / √k`, where `k` is the number of calibration
> subsamples averaged into one reported cell by the run being classified.

**Reason.** The original rule compares two different quantities. `σ₂` is the
spread of `D` measured on *one* draw, but Task C requires each cell to be
reported as the mean of at least three draws, and the standard error of a mean
of `k` draws is `σ₂/√k`. Classifying a mean-of-k value against a single-draw
spread under-resolves by a factor of `√k` — at `k = 3` it inflates the
tolerance by about 1.73×, systematically pushing real structure into the
"unresolved" bucket and biasing the milestone toward the Dead branch for a
reason that is arithmetic rather than physical.

**Binding consequences.**

1. `k` is fixed *before* the run it calibrates, is written into the config as
   `submodularity.epsilon_cell_subsamples`, and is written into every result
   record beside the tolerance.
2. A run whose actual subsamples-per-cell differs from the `k` its epsilon was
   built with must **fail loudly**, not silently classify against the wrong
   tolerance.
3. `σ₂` itself is still measured on single draws, exactly as Task B specifies.
   Only the conversion from `σ₂` to `ε` changes.
4. Everything else stands: epsilon is still not tuned after seeing
   classifications, still per modality, and still recorded next to every verdict.
5. The Method-paper rule is untouched. It is stated against the interaction
   term's own noise floor `σ_I` under the retrieval metric, not against `ε`.

### Amendment 2 — the verdict is a band, and resolution needs an effect gate

**Date:** 2026-08-21. **Recorded before:** any Task C cell was evaluated or
classified, and before the Step 2 conditioning check was run. Every number below
comes from a run already on disk:
`results/second_order_floor__20260821T200032Z__601002ea.json` (commit
`601002ea`) and
`results/diagnostic_0__merged__20260820T210527Z__75d4665a.json`. Nothing here
was chosen with a Task C output visible.

#### A. The resolved rate is reported as a band, and the branch is called on the point estimate

`σ₂` is estimated from six values per square. Its 95% chi-square interval spans
roughly a factor of four:

| modality | `σ₂` | 95% interval | `ε = σ₂/√3` |
| --- | --- | --- | --- |
| W\|W | 0.00455 | 0.00284 – 0.01117 | 0.00263 |
| KV\|KV | 0.00233 | 0.00145 – 0.00570 | 0.00134 |
| W\|KV and KV\|W | 0.00288 | 0.00180 – 0.00707 | 0.00166 |

That interval is wider than the `√3 = 1.73` correction Amendment 1 was written
about. A verdict that turns on a resolved-rate threshold cannot honestly be
reported against a point estimate alone.

**Rule.** Task C reports its resolved rate **three times per modality**: once at
the `σ₂` point estimate, once at the lower interval bound, and once at the
upper. Each is labelled with the `σ₂` it used.

* The **branch is called on the point estimate**, and only on the point
  estimate.
* The two bound readings are a **robustness band**. They are reported beside the
  verdict and are **never** used to select it.

**What the band is for.** If the verdict is the same at all three, the result is
stronger than the point estimate alone could justify: it survives the full range
of floors the six measurements are consistent with. If the verdict flips at a
bound, that is not a licence to pick the convenient reading — it is a finding
about the precision of the floor, and it is reported as one, naming the bound at
which it flips and what would be needed to tighten it (more subsamples, which
narrow the interval as `√(2(n−1))`). Declaring both outcomes here, in advance,
is what makes this a robustness report and not a choice made after the fact.

*Note for interpretation, not a rule:* the floor run's two cross-component
squares are the **same square**. The mixed second difference is symmetric in its
two components, so W\|KV and KV\|W over one layer with one weight move and one
KV move yield the identical `D`. The floor therefore rests on **three**
independent measurements, not four. Task C's cross-component cells are genuinely
distinct because their target and conditioning moves differ.

#### B. Two gates: statistical resolution and allocation relevance

`RESOLUTION_TEST` currently reads `|D| > ε` and nothing else. With `ε` for W\|W
at 0.00263 PPL — about 0.04% of a perplexity near 6.9 — statistical resolution
has become cheap while allocation relevance has not moved at all. A cell can now
resolve on a difference no allocator would ever act on. That is the same failure
the original report committed, with better statistics underneath it.

Two gates are defined. **Both are always reported. They are never merged into
one number.** A cell counts toward the branch decision only if it passes both.

**Statistical gate.** `|D| > ε(modality)`, with `ε = σ₂/√k` per Amendment 1.
Unchanged.

**Effect gate.** Two clauses, both on the target's own marginal gain
`m = max(|m(S_A)|, |m(S_B)|)`:

> **E1, the upgrade is worth buying:** `m > 3 · σ₁(kind)`, where
> `σ₁(weight) = 0.01153` and `σ₁(kv) = 0.00260` PPL.
> Thresholds: **0.03460 PPL for a weight target, 0.00781 PPL for a KV target.**
>
> **E2, the interaction is material:** `|D| ≥ φ · m` with **`φ = 0.10`**.

`max` and not `min` in `m`: if conditioning collapses an upgrade's value, `m(S_B)`
is near zero and that is precisely the interaction being hunted. Taking the
minimum would discard the most interesting cell in the matrix.

**Why `σ₁` is the single-draw first-order noise and is not divided by `√k`.**
The statistical gate asks "can this be distinguished from zero", so it correctly
gets cheaper as more data is averaged. The effect gate asks "is this worth
allocating for", and that must not get cheaper merely because more calibration
windows were read. `σ₁` is taken from the floor record's `stdev m(A)`, the
larger value across the squares of each kind, so the bar is the conservative one.

**Why `φ = 0.10`, derived and not asserted.** A greedy allocator ranks candidate
increments by benefit per byte. An interaction changes its decision only if it is
large enough to reorder that ranking. Diagnostic 0 gives the sixteen per-layer
gains for each real ladder step, so the spacing between adjacent competing
candidates is measurable. As a fraction of the median gain, that spacing is:

| ladder step | median adjacent-rank gap / median gain |
| --- | --- |
| W 3→4 | 6.6% |
| W 4→8 | 5.3% |
| KV 0.25→0.50 | 9.5% |
| KV 0.50→0.75 | 26.2% |
| **pooled median** | **8.1%** |

An interaction smaller than the gap between the candidates the allocator is
choosing between cannot change which one it picks. `φ` is set to **0.10**: just
above the pooled median of 8.1%, rounded so the gate is not tuned to a decimal
place, and above three of the four steps. Rounding up makes the gate **harder**
to pass, which is conservatism against this project's own hypothesis rather than
in favour of it.

**Consequences that follow arithmetically, stated now so they cannot later look
like excuses.** From Diagnostic 0's mean gains, `W 4→8` buys 0.02276 PPL against
an E1 threshold of 0.03460, so most `W 4→8` target cells are expected to fail
E1. `W 3→4` buys 0.09460, `KV 0.25→0.50` buys 0.04230, and `KV 0.50→0.75` buys
0.01113, all above their thresholds. E1 is evaluated **per cell** on that cell's
own measured gains, not on these means, so individual layers may fall either
way — the `W 4→8` per-layer gains run from 0.00475 to 0.04042 and the top of
that range clears the threshold. From the floor run, the W\|W square had
`|D|/m = 0.00183/0.06176 = 3.0%`, well under `φ = 0.10`, so a Task C cell that
behaves like it fails E2.

**Failure reasons are reported separately**, never pooled into one "unresolved"
count: failed the statistical gate; failed E1 (the target buys nothing); failed
E2 (two real gains happened to match). A cell that fails E1 and a cell that
fails E2 are different facts about the model.

#### C. The Step 2 conditioning check, decided before the number exists

The floor was measured with **adjacent** conditioning (`L4 W:3→4`). Task C
conditions on the **large** moves `W:3→16` and `KV:0.25→1.00`. If `σ₂` scales
with how much the conditioning move perturbs the model, a floor measured at the
smaller perturbation understates the tolerance Task C needs.

Step 2 runs one additional square at the exact Task C conditioning —
`L8 (W:3→4) | L4 (W:3→16)` — on the same six subsamples and the same
calibration configuration, and compares its `σ₂` against the matching
adjacent-conditioning square.

> **If the new `σ₂` lies inside 0.00284 – 0.01117**, the 95% interval of the
> adjacent-conditioning W\|W square, the measured floor **stands** and Task C
> proceeds on it unchanged.
>
> **If it lies outside that interval**, Task B is **not finished**. The floor is
> re-derived at the Task C conditioning, for every modality, before any cell is
> classified. Task C does not run in the meantime.

Either outcome is reported with the number that produced it. The expectation is
that it barely moves: on this ladder the large conditioning move is only
slightly larger than the adjacent one, 0.118 against 0.095 of removed
degradation for weights (ratio 1.24) and the same ratio 1.28 for KV. A large
shift would mean something other than perturbation size is driving `σ₂`, and
that would be chased before Task C rather than after.
