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
# Every adjacent weight/KV tier upgrade (112 interactions):
GPUS=2 scripts/run_submodularity.sh --all-upgrades

# Inspect the matrix without loading weights or data:
.venv/bin/python scripts/submodularity_diagnostic.py --dry-run

# Optional Mixtral configuration; requires substantially more memory:
.venv/bin/python scripts/submodularity_diagnostic.py \
  --config configs/mixtral.yaml --device cuda:0
```

The default OLMoE grid is `[2, 6, 10, 14]` because OLMoE has 16 decoder
layers. The optional 32-layer Mixtral grid is `[2, 10, 18, 26]`.
