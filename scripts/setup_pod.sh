#!/usr/bin/env bash
# Prepare a rented CUDA machine to run Sub-MoKV.
#
# Results from different accelerators are not comparable: the same allocation
# gives different floating point values on MPS and on CUDA. Everything reported
# in the paper has to come from one device, and the memoization file is named
# for the device that produced it so a cache cannot be carried across.
set -euo pipefail

REPO_DIR="${1:-$HOME/submokv}"
cd "$REPO_DIR"

echo "== python environment =="
command -v uv >/dev/null || pip install -q uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    numpy scipy pandas matplotlib pytest pyyaml tqdm \
    "torch" "transformers==5.15.1" datasets accelerate safetensors

echo "== versions =="
.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("transformers", transformers.__version__)
PY

echo "== unit tests, no accelerator needed =="
.venv/bin/python -m pytest tests/ -q

echo "== model =="
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print("SNAPSHOT:", snapshot_download("allenai/OLMoE-1B-7B-0924",
      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], max_workers=8))
PY

echo "== done. Next: scripts/verify_device.sh =="
