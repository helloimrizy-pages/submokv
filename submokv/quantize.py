"""Fake quantization of MoE expert weights.

Quantization is applied by writing dequantized values back into the expert
parameters in place. Shapes and dtypes do not change, so the model runs
unmodified and only the numerics move. The footprint that a bit width implies is
computed analytically in memory.py and is never read from the device.

In transformers 5.x an OLMoE layer stores all of its experts in two packed
parameters rather than one Linear per expert:

    experts.gate_up_proj  (num_experts, 2 * intermediate_size, hidden_size)
    experts.down_proj     (num_experts, hidden_size, intermediate_size)

Groups run along the last dimension of each slice, which is the input dimension
of the matrix, so the group count matches the accounting in memory.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from .memory import QuantSpec

EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
_LAYER_INDEX_PATTERN = re.compile(r"layers\.(\d+)\.")

WeightPlan = Mapping[int, Mapping[int, int]]


def fake_quantize(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    unquantized_bits: int = 16,
) -> torch.Tensor:
    """Return a symmetric per-group quantized then dequantized copy of a weight tensor.

    Groups run along the last dimension. Each group is scaled by its own largest
    magnitude, rounded to a signed integer grid of 2 ** (bits - 1) - 1 levels
    either side of zero, and scaled back. The returned tensor has the same shape
    and dtype as the input. At unquantized_bits or above the input is returned
    unchanged, which makes the top tier an exact identity.
    """
    if bits >= unquantized_bits:
        return weight
    if bits < 2:
        raise ValueError(f"bits must be at least 2, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    original_shape = weight.shape
    original_dtype = weight.dtype
    in_features = original_shape[-1]

    values = weight.detach().to(torch.float32).reshape(-1, in_features)
    padding = (-in_features) % group_size
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    grouped = values.reshape(values.shape[0], -1, group_size)

    level = 2 ** (bits - 1) - 1
    scale = grouped.abs().amax(dim=-1, keepdim=True) / level
    # A group of exact zeros has no scale; leaving it at zero would divide by zero.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.clamp(torch.round(grouped / scale), -level, level)
    restored = (codes * scale).reshape(values.shape[0], -1)

    if padding:
        restored = restored[:, :in_features]
    return restored.reshape(original_shape).to(original_dtype)


@dataclass(frozen=True)
class ExpertModule:
    """One MoE layer's packed expert parameters, with the layer index it belongs to."""

    layer: int
    name: str
    module: nn.Module

    @property
    def num_experts(self) -> int:
        """Return how many experts this module holds."""
        return int(getattr(self.module, EXPERT_PARAMETER_NAMES[0]).shape[0])


def find_expert_modules(model: nn.Module) -> tuple[ExpertModule, ...]:
    """Return the packed expert modules of a model, ordered by layer index.

    A module qualifies when it holds every parameter in EXPERT_PARAMETER_NAMES
    as a three dimensional tensor whose first dimension is the expert axis.
    """
    found: list[ExpertModule] = []
    for name, module in model.named_modules():
        parameters = dict(module.named_parameters(recurse=False))
        if not all(key in parameters for key in EXPERT_PARAMETER_NAMES):
            continue
        if any(parameters[key].ndim != 3 for key in EXPERT_PARAMETER_NAMES):
            continue
        match = _LAYER_INDEX_PATTERN.search(name)
        if match is None:
            raise ValueError(f"cannot read a layer index from module name {name!r}")
        found.append(ExpertModule(layer=int(match.group(1)), name=name, module=module))
    found.sort(key=lambda entry: entry.layer)
    if not found:
        raise ValueError(
            "no packed expert modules found; expected modules holding "
            f"{EXPERT_PARAMETER_NAMES} as three dimensional parameters"
        )
    return tuple(found)


class ExpertQuantizer:
    """Applies a per expert bit width plan to a model's expert weights.

    The unmodified weights are kept in a master store so that any expert can be
    moved to any tier at runtime without reloading the model. Only experts whose
    bit width changes are rewritten, so moving one unit up its ladder touches one
    layer.
    """

    def __init__(
        self,
        model: nn.Module,
        quant: QuantSpec,
        master_device: str | torch.device = "cpu",
    ) -> None:
        if not quant.symmetric:
            raise NotImplementedError(
                "only symmetric quantization is implemented; memory.py can cost an "
                "asymmetric quantizer but quantize.py does not produce one"
            )
        self.model = model
        self.quant = quant
        self.master_device = torch.device(master_device)
        self.expert_modules = find_expert_modules(model)
        self._master: dict[tuple[int, str], torch.Tensor] = {}
        self._active_bits: dict[int, dict[int, int]] = {}
        self._attached = False

    @property
    def layers(self) -> tuple[int, ...]:
        """Return the layer indices that hold experts."""
        return tuple(entry.layer for entry in self.expert_modules)

    def attach(self) -> "ExpertQuantizer":
        """Copy the unmodified expert weights into the master store.

        The store holds one copy of every expert parameter, so it costs as many
        bytes as the expert weights themselves.
        """
        if self._attached:
            return self
        for entry in self.expert_modules:
            for name in EXPERT_PARAMETER_NAMES:
                parameter = getattr(entry.module, name)
                self._master[(entry.layer, name)] = parameter.detach().to(
                    self.master_device, copy=True
                )
            self._active_bits[entry.layer] = {
                expert: self.quant.unquantized_bits for expert in range(entry.num_experts)
            }
        self._attached = True
        return self

    def master_bytes(self) -> int:
        """Return the bytes held by the master store."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self._master.values())

    def current_plan(self) -> dict[int, dict[int, int]]:
        """Return the bit width currently applied to every expert."""
        self._require_attached()
        return {layer: dict(bits) for layer, bits in self._active_bits.items()}

    def set_plan(self, plan: WeightPlan) -> int:
        """Apply a layer to expert to bit width plan and return how many experts changed.

        Experts already at the requested bit width are left untouched, so
        repeated calls that move one unit up its ladder rewrite one layer.
        """
        self._require_attached()
        modules = {entry.layer: entry for entry in self.expert_modules}
        unknown = set(plan) - set(modules)
        if unknown:
            raise KeyError(f"plan names layers {sorted(unknown)} that hold no experts")

        changed = 0
        for layer, expert_bits in plan.items():
            entry = modules[layer]
            active = self._active_bits[layer]
            pending = {
                expert: bits for expert, bits in expert_bits.items() if active[expert] != bits
            }
            if not pending:
                continue
            # Experts that move to the same bit width are rewritten in one call.
            # Groups run along the last dimension, so quantizing a stack of
            # experts gives the same result as quantizing them one at a time.
            grouped: dict[int, list[int]] = {}
            for expert, bits in pending.items():
                grouped.setdefault(bits, []).append(expert)
            for name in EXPERT_PARAMETER_NAMES:
                parameter = getattr(entry.module, name)
                master = self._master[(entry.layer, name)]
                for bits, experts in grouped.items():
                    experts = sorted(experts)
                    source = master[experts].to(parameter.device, dtype=parameter.dtype)
                    with torch.no_grad():
                        parameter.data[experts] = fake_quantize(
                            source, bits, self.quant.group_size, self.quant.unquantized_bits
                        )
            for expert, bits in pending.items():
                active[expert] = bits
            changed += len(pending)
        return changed

    def set_uniform_bits(self, bits: int, layers: Iterable[int] | None = None) -> int:
        """Apply one bit width to every expert, or to the experts of the named layers."""
        self._require_attached()
        selected = set(self.layers) if layers is None else set(layers)
        modules = {entry.layer: entry for entry in self.expert_modules}
        plan = {
            layer: {expert: bits for expert in range(modules[layer].num_experts)}
            for layer in sorted(selected)
        }
        return self.set_plan(plan)

    def restore(self) -> int:
        """Return every expert to its unmodified weights."""
        return self.set_uniform_bits(self.quant.unquantized_bits)

    def describe(self) -> dict[str, object]:
        """Return the quantizer settings for result records."""
        self._require_attached()
        counts: dict[int, int] = {}
        for expert_bits in self._active_bits.values():
            for bits in expert_bits.values():
                counts[bits] = counts.get(bits, 0) + 1
        return {
            "group_size": self.quant.group_size,
            "symmetric": self.quant.symmetric,
            "unquantized_bits": self.quant.unquantized_bits,
            "num_expert_modules": len(self.expert_modules),
            "experts_per_bit_width": {str(k): v for k, v in sorted(counts.items())},
            "master_bytes": self.master_bytes(),
        }

    def _require_attached(self) -> None:
        if not self._attached:
            raise RuntimeError("call attach() before using the quantizer")
