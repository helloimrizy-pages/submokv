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
