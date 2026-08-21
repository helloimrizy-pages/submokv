# Sub-MoKV diagnostics

This repository measures whether joint MoE expert-weight precision and KV-cache
retention exhibit diminishing returns or cross-component synergy. The main
experiment is designed to run on a CUDA pod; two A40s are a suitable setup.

## RunPod: 2×A40

Use persistent storage such as `/workspace` so the downloaded checkpoint,
evaluation cache, logs, and JSON results survive a pod restart.

```bash
cd /workspace
git clone https://github.com/helloimrizy-pages/submokv.git
cd submokv

scripts/setup_pod.sh
scripts/verify_device.sh
GPUS=2 scripts/run_submodularity.sh
```

The setup creates `.venv`, installs a CUDA-compatible PyTorch build and the
pinned Transformers version, runs the unit tests, and downloads
`allenai/OLMoE-1B-7B-0924`. It also prepares the default WikiText token windows
once, before the GPU workers launch. The verification command checks checkpoint
repacking, plausible perplexity, and bitwise determinism before any experimental
result is trusted.

By default, Hugging Face artifacts are stored in `.hf-cache` inside the cloned
repository, so placing the repository on `/workspace` also makes the model and
dataset cache persistent. Set `HF_HOME` before running setup if your pod uses a
different persistent-volume path.

The two-GPU runner launches one complete model replica on each A40. Each worker
evaluates a disjoint half of the 32-interaction matrix and writes to its own
memoization file. The final merge refuses missing/duplicate interactions or
shards that disagree on the full-precision reference. Outputs are written to:

```text
logs/submodularity_<run>_s0of2.log
logs/submodularity_<run>_s1of2.log
results/submodularity_<run>_s0of2.json
results/submodularity_<run>_s1of2.json
results/submodularity_<run>_merged.json
```

For a cheaper end-to-end smoke test before the 64-sequence run:

```bash
SEQUENCES=8 GPUS=2 scripts/run_submodularity.sh
```

The sequence count must be divisible by the configured batch size (four for
OLMoE). Rerunning the same command resumes from per-shard evaluation caches.

Useful overrides:

```bash
# Every adjacent move on the ground set's weight/KV ladders (96 interactions):
GPUS=2 scripts/run_submodularity.sh --all-upgrades

# Inspect the matrix without loading weights or data:
.venv/bin/python scripts/submodularity_diagnostic.py --dry-run

# Optional Mixtral configuration; requires substantially more memory:
.venv/bin/python scripts/submodularity_diagnostic.py \
  --config configs/mixtral.yaml --device cuda:0
```

The default OLMoE grid is `[2, 6, 10, 14]` because OLMoE has 16 decoder
layers. The optional 32-layer Mixtral grid is `[2, 10, 18, 26]`.

The tier ladders come from `ground_set.weight_tiers` and `ground_set.kv_tiers`
in the config, and from nowhere else. A requested transition that names a tier
the ground set does not contain is refused with an error naming the ladder;
pass `--allow-outside-ground-set` to record such a probe anyway, in which case
every affected row is labelled `in_ground_set: false`.

The report separates three tests that were previously mixed: the **epsilon
test** is the headline and is what the Classification column shows, the
**strict sign test** is a labelled secondary line, and the **resolution test**
reports how many cells cleared epsilon in either direction at all. A cell
inside epsilon is not counted as submodular evidence.

The paper decision rules for Milestone 3b are fixed in `DECISION.md` and are not
amended by any verdict the tool prints.

## The second-order noise floor

Epsilon is not a constant. The quantity the submodularity test classifies is
the second-order difference

```text
D = [F(S_A + j) - F(S_A)] - [F(S_B + j) - F(S_B)]
  = [PPL(S_A) - PPL(S_A+j)] - [PPL(S_B) - PPL(S_B+j)]
```

so the floor it must clear is the spread of `D` itself, not the spread of one
perplexity and not the spread of one delta. The full-precision reference
cancels out of `D`, so no reference evaluation is needed to measure it.

```bash
.venv/bin/python -m submokv.cli second-order-floor --subsamples 6 --layers 8,4
```

This evaluates all four corners of one square per modality on the same
calibration subsample, repeats that over non-overlapping subsamples, and reports
the stdev per modality with a chi-square interval. Weight-side and KV-side
scales differ by roughly a factor of seven, so the result is four numbers, not
one. It prints the YAML to paste under `submodularity.epsilon_ppl`, and every
classification the diagnostic later writes carries the `epsilon_used` and
`epsilon_source` it was compared against.

Evaluation order is plan-major: every subsample of one allocation is scored
before the next allocation is installed, because installing one costs a
checkpoint read and a requantization of each changed layer while changing the
subsample costs nothing.

**The floor and the run it calibrates must read the same windows.** All
calibration flags now default to the config rather than to a flag constant, so
`--sequence-length` and `--calibration-split` no longer silently move an
experiment off the windows its floor was measured on.
