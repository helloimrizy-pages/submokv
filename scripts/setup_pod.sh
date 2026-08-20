#!/usr/bin/env bash
# Prepare a rented CUDA machine to run Sub-MoKV.
#
# Results from different accelerators are not comparable: the same allocation
# gives different floating point values on MPS and on CUDA. Everything reported
# in the paper has to come from one device, and the memoization file is named
# for the device that produced it so a cache cannot be carried across.
set -euo pipefail

# Default to the repository this script lives in, so it works wherever the
# machine puts the checkout.
REPO_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
echo "== repository: $REPO_DIR =="

# The wheel has to match the driver already on the machine. PyPI serves the
# newest CUDA build of torch, which on a machine with an older driver fails at
# import with "NVIDIA driver on your system is too old". Read the driver and
# pick the matching index rather than guessing.
if [[ -z "${TORCH_INDEX:-}" ]]; then
    DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1 || true)"
    case "${CUDA_MAJOR:-}" in
        13*) TORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
        128) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
        126|127) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
        *)   TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    esac
    echo "== driver ${DRIVER_CUDA:-unknown} (CUDA ${CUDA_MAJOR:-unknown}) -> ${TORCH_INDEX} =="
fi

echo "== python environment =="
command -v uv >/dev/null || pip install -q uv
uv venv --python 3.12 --clear .venv
# torch first, from the CUDA index, so the nvidia runtime packages it pulls are
# the ones the driver supports. Anything installed before it would fix them to
# the wrong line.
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" torch
uv pip install --python .venv/bin/python \
    numpy scipy pandas matplotlib pytest pyyaml tqdm \
    "transformers==5.15.1" datasets accelerate safetensors

echo "== versions =="
.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "torch cannot see a GPU. If the message above says the driver is too old, "
        "set TORCH_INDEX to a build matching the driver and run this script again."
    )
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
