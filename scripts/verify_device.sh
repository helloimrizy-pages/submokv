#!/usr/bin/env bash
# Check a new device before any number is recorded from it.
#
# Three things have to hold: the master store reproduces the weights the model
# was loaded with, F is bitwise deterministic, and perplexity is in the range a
# working model gives. Any of them failing invalidates everything downstream.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

.venv/bin/python - <<'PY'
import glob, time, torch
from pathlib import Path
from submokv.cli import load_config
from submokv.utility import build_utility, CHEAP

snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*"))[0]
config = load_config(Path("configs/olmoe.yaml"))
config["calibration"]["cheap_sequences"] = 8
model, utility = build_utility(config, model_path=snap)
print(f"device {utility.device}  cache {utility.cache_path.name}")

ids = torch.randint(0, 50000, (4, 4096), device=utility.device)
with torch.no_grad():
    model(input_ids=ids[:, :256], use_cache=False)
torch.cuda.synchronize() if utility.device.type == "cuda" else None
start = time.perf_counter()
with torch.no_grad():
    model(input_ids=ids, use_cache=False)
torch.cuda.synchronize() if utility.device.type == "cuda" else None
seconds = time.perf_counter() - start
print(f"forward batch 4 x 4096: {seconds:.2f}s  ({4 * 4096 / seconds:.0f} tok/s)  "
      f"[8.02s on the Mac]")

top = utility.evaluate(utility.ground_set.full_allocation(), CHEAP, 0, use_cache=False)
print(f"top allocation perplexity {top.perplexity:.6f}  [5.844 on the Mac, expected to differ]")
assert 3.0 < top.perplexity < 20.0, "perplexity outside the range a working model gives"

check = utility.verify_determinism(utility.ground_set.full_allocation())
print(f"determinism: identical={check['identical']}  difference={check['absolute_difference']:.3e}")
assert check["identical"], (
    "F is not deterministic on this device; index_add_ on CUDA accumulates with "
    "atomics unless deterministic algorithms are enabled"
)
print("\nDEVICE VERIFIED")
PY
