"""Units, tiers, increments, and the chain constraint over upgrade ladders.

The search space is a set of ordered upgrade ladders, not a flat set. Each unit
starts at its lowest tier and can only be moved up one tier at a time, so the
candidate pool at any point in the search holds at most one increment per unit.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .memory import (
    Footprint,
    KVSpec,
    ModelSpec,
    QuantSpec,
    TierValue,
    expert_bytes,
    fixed_weight_bytes,
    format_bytes,
    kv_bytes_per_layer,
    reference_footprint,
)

# The floor is 3 bits, not 2. A symmetric grid of 2 ** (bits - 1) - 1 levels is
# ternary at 2 bits, so one of the four codes is unused and the tier delivers
# less than memory.py charges for it. 2 bits stays available as a diagnostic
# probe but is not part of the search space.
DEFAULT_WEIGHT_TIERS: tuple[int, ...] = (3, 4, 8, 16)
DEFAULT_KV_TIERS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


class ChainConstraintError(ValueError):
    """Raised when a selection skips a tier or repeats one on the same unit."""


class BudgetInfeasibleError(ValueError):
    """Raised when the base state already exceeds the budget."""


class UnknownIncrementError(KeyError):
    """Raised when an increment identifier is not part of this ground set."""


class UnitKind(Enum):
    """The two kinds of unit the budget is split between."""

    WEIGHT = "weight"
    KV = "kv"


@dataclass(frozen=True)
class Unit:
    """One expert weight group or one layer's KV cache, with its ordered tiers.

    Tiers run from lowest to highest. Weight tiers are bit widths, KV tiers are
    retention ratios.
    """

    unit_id: str
    kind: UnitKind
    layer: int
    tiers: tuple[TierValue, ...]
    expert_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.tiers) < 2:
            raise ValueError(f"unit {self.unit_id} needs at least two tiers, got {self.tiers}")
        if list(self.tiers) != sorted(self.tiers):
            raise ValueError(f"unit {self.unit_id} tiers must be ordered low to high, got {self.tiers}")
        if len(set(self.tiers)) != len(self.tiers):
            raise ValueError(f"unit {self.unit_id} tiers must be distinct, got {self.tiers}")

    @property
    def num_increments(self) -> int:
        """Return the number of upgrade steps on this unit's ladder."""
        return len(self.tiers) - 1

    @property
    def top_tier_index(self) -> int:
        """Return the index of the highest tier."""
        return len(self.tiers) - 1


@dataclass(frozen=True)
class Increment:
    """One move from tier step-1 to tier step on a single unit.

    cost_bytes is the byte delta of that move, not the absolute size of the
    resulting tier.
    """

    increment_id: str
    unit_id: str
    kind: UnitKind
    layer: int
    step: int
    from_tier: TierValue
    to_tier: TierValue
    cost_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return the increment as a plain dictionary for result records."""
        return {
            "increment_id": self.increment_id,
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "layer": self.layer,
            "step": self.step,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "cost_bytes": self.cost_bytes,
        }


@dataclass(frozen=True)
class Allocation:
    """A tier index per unit, which is the canonical state of the search.

    Storing a tier index per unit rather than a set of increments makes the
    chain constraint hold by construction. The equivalent set of increments is
    recovered with GroundSet.selection_from_allocation.
    """

    signature: str
    tier_index: tuple[tuple[str, int], ...]

    @classmethod
    def from_mapping(cls, signature: str, values: Mapping[str, int]) -> "Allocation":
        """Build an allocation from a unit identifier to tier index mapping."""
        return cls(signature=signature, tier_index=tuple(sorted(values.items())))

    def as_dict(self) -> dict[str, int]:
        """Return the tier index per unit as a plain dictionary."""
        return dict(self.tier_index)

    def canonical_hash(self) -> str:
        """Return a stable hash of this allocation for memoization.

        The ground set signature is part of the hash, so a cached value cannot
        be reused across incompatible ground sets.
        """
        payload = json.dumps(
            {"signature": self.signature, "tier_index": list(self.tier_index)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def num_selected_increments(self) -> int:
        """Return how many increments above the base state this allocation holds."""
        return sum(index for _, index in self.tier_index)


@dataclass(frozen=True)
class BudgetPlan:
    """The byte accounting for one budget fraction, resolved against a ground set."""

    fraction: float
    reference_bytes: int
    budget_bytes: int
    base_bytes: int
    max_bytes: int

    @property
    def slack_bytes(self) -> int:
        """Return the bytes available to the search after the base state is paid for."""
        return self.budget_bytes - self.base_bytes

    @property
    def base_fraction(self) -> float:
        """Return the base state footprint as a fraction of the reference footprint."""
        return self.base_bytes / self.reference_bytes

    def as_dict(self) -> dict[str, Any]:
        """Return the plan as a plain dictionary for result records."""
        return {
            "fraction": self.fraction,
            "reference_bytes": self.reference_bytes,
            "budget_bytes": self.budget_bytes,
            "base_bytes": self.base_bytes,
            "max_bytes": self.max_bytes,
            "slack_bytes": self.slack_bytes,
            "base_fraction": self.base_fraction,
        }


class GroundSet:
    """The ordered upgrade ladders for one model, budget context, and quantizer."""

    def __init__(
        self,
        model: ModelSpec,
        kv: KVSpec,
        quant: QuantSpec,
        weight_tiers: Sequence[int] = DEFAULT_WEIGHT_TIERS,
        kv_tiers: Sequence[float] = DEFAULT_KV_TIERS,
        weight_groups_per_layer: int = 1,
    ) -> None:
        self.model = model
        self.kv = kv
        self.quant = quant
        self.weight_tiers = tuple(int(t) for t in weight_tiers)
        self.kv_tiers = tuple(float(t) for t in kv_tiers)
        self.weight_groups_per_layer = int(weight_groups_per_layer)
        if self.weight_groups_per_layer < 1:
            raise ValueError(
                f"weight_groups_per_layer must be at least 1, got {self.weight_groups_per_layer}"
            )
        if self.weight_groups_per_layer > model.num_experts:
            raise ValueError(
                f"weight_groups_per_layer {self.weight_groups_per_layer} exceeds "
                f"num_experts {model.num_experts}"
            )
        self.units: tuple[Unit, ...] = self._build_units()
        self._unit_by_id = {unit.unit_id: unit for unit in self.units}
        self.increments: tuple[Increment, ...] = self._build_increments()
        self.increment_by_id = {inc.increment_id: inc for inc in self.increments}
        self._increments_by_unit: dict[str, tuple[Increment, ...]] = {
            unit.unit_id: tuple(
                inc for inc in self.increments if inc.unit_id == unit.unit_id
            )
            for unit in self.units
        }
        self.signature = self._compute_signature()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GroundSet":
        """Build a ground set from a parsed configuration mapping."""
        ground = dict(config.get("ground_set", {}))
        return cls(
            model=ModelSpec.from_mapping(config["model"]),
            kv=KVSpec.from_mapping(config["kv"]),
            quant=QuantSpec.from_mapping(config.get("quant", {})),
            weight_tiers=ground.get("weight_tiers", DEFAULT_WEIGHT_TIERS),
            kv_tiers=ground.get("kv_tiers", DEFAULT_KV_TIERS),
            weight_groups_per_layer=ground.get("weight_groups_per_layer", 1),
        )

    def _expert_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the expert indices belonging to each weight group of a layer."""
        total = self.model.num_experts
        groups = self.weight_groups_per_layer
        base, remainder = divmod(total, groups)
        result: list[tuple[int, ...]] = []
        start = 0
        for index in range(groups):
            size = base + (1 if index < remainder else 0)
            result.append(tuple(range(start, start + size)))
            start += size
        return tuple(result)

    def _build_units(self) -> tuple[Unit, ...]:
        units: list[Unit] = []
        expert_groups = self._expert_groups()
        for layer in range(self.model.num_hidden_layers):
            for group_index, expert_indices in enumerate(expert_groups):
                suffix = "" if self.weight_groups_per_layer == 1 else f".g{group_index}"
                units.append(
                    Unit(
                        unit_id=f"w.l{layer:02d}{suffix}",
                        kind=UnitKind.WEIGHT,
                        layer=layer,
                        tiers=self.weight_tiers,
                        expert_indices=expert_indices,
                    )
                )
        for layer in range(self.model.num_hidden_layers):
            units.append(
                Unit(
                    unit_id=f"kv.l{layer:02d}",
                    kind=UnitKind.KV,
                    layer=layer,
                    tiers=self.kv_tiers,
                )
            )
        return tuple(units)

    def unit_bytes(self, unit: Unit, tier_index: int) -> int:
        """Return the bytes held by one unit at one tier index."""
        tier = unit.tiers[tier_index]
        if unit.kind is UnitKind.WEIGHT:
            return expert_bytes(self.model, self.quant, int(tier), len(unit.expert_indices))
        return kv_bytes_per_layer(self.model, self.kv, float(tier))

    def _build_increments(self) -> tuple[Increment, ...]:
        increments: list[Increment] = []
        for unit in self.units:
            for step in range(1, len(unit.tiers)):
                cost = self.unit_bytes(unit, step) - self.unit_bytes(unit, step - 1)
                if cost <= 0:
                    raise ValueError(
                        f"increment {unit.unit_id} step {step} "
                        f"({unit.tiers[step - 1]} to {unit.tiers[step]}) has cost {cost} bytes; "
                        "every step up a ladder must cost bytes"
                    )
                increments.append(
                    Increment(
                        increment_id=f"{unit.unit_id}:{step}",
                        unit_id=unit.unit_id,
                        kind=unit.kind,
                        layer=unit.layer,
                        step=step,
                        from_tier=unit.tiers[step - 1],
                        to_tier=unit.tiers[step],
                        cost_bytes=cost,
                    )
                )
        return tuple(increments)

    def _compute_signature(self) -> str:
        payload = json.dumps(
            {
                "model": self.model.name,
                "num_hidden_layers": self.model.num_hidden_layers,
                "hidden_size": self.model.hidden_size,
                "intermediate_size": self.model.intermediate_size,
                "num_experts": self.model.num_experts,
                "num_key_value_heads": self.model.num_key_value_heads,
                "head_dim": self.model.head_dim,
                "vocab_size": self.model.vocab_size,
                "tie_word_embeddings": self.model.tie_word_embeddings,
                "has_qk_norm": self.model.has_qk_norm,
                "dtype_bytes": self.model.dtype_bytes,
                "kv": [self.kv.context_length, self.kv.batch_size, self.kv.dtype_bytes, self.kv.sink_tokens],
                "quant": [
                    self.quant.group_size,
                    self.quant.scale_bits,
                    self.quant.zero_point_bits,
                    self.quant.symmetric,
                    self.quant.unquantized_bits,
                ],
                "weight_tiers": list(self.weight_tiers),
                "kv_tiers": list(self.kv_tiers),
                "weight_groups_per_layer": self.weight_groups_per_layer,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def unit(self, unit_id: str) -> Unit:
        """Return the unit with the given identifier."""
        try:
            return self._unit_by_id[unit_id]
        except KeyError as error:
            raise UnknownIncrementError(f"unknown unit {unit_id!r}") from error

    def increment(self, increment_id: str) -> Increment:
        """Return the increment with the given identifier."""
        try:
            return self.increment_by_id[increment_id]
        except KeyError as error:
            raise UnknownIncrementError(f"unknown increment {increment_id!r}") from error

    def base_allocation(self) -> Allocation:
        """Return the allocation with every unit at its lowest tier."""
        return Allocation.from_mapping(self.signature, {unit.unit_id: 0 for unit in self.units})

    def full_allocation(self) -> Allocation:
        """Return the allocation with every unit at its highest tier."""
        return Allocation.from_mapping(
            self.signature, {unit.unit_id: unit.top_tier_index for unit in self.units}
        )

    def candidates(self, allocation: Allocation) -> tuple[Increment, ...]:
        """Return the next increment up the ladder for every unit not yet at its top tier.

        This is the whole candidate pool at one step of the search, so its size
        is at most the number of units.
        """
        self._check_signature(allocation)
        indices = allocation.as_dict()
        result: list[Increment] = []
        for unit in self.units:
            index = indices[unit.unit_id]
            if index < unit.top_tier_index:
                result.append(self._increments_by_unit[unit.unit_id][index])
        return tuple(result)

    def apply(self, allocation: Allocation, increment: Increment) -> Allocation:
        """Return the allocation with one increment applied.

        Raises ChainConstraintError when the increment is not the next step up
        that unit's ladder.
        """
        self._check_signature(allocation)
        indices = allocation.as_dict()
        current = indices[increment.unit_id]
        if increment.step != current + 1:
            raise ChainConstraintError(
                f"increment {increment.increment_id} moves to step {increment.step} "
                f"but unit {increment.unit_id} is at tier index {current}; "
                f"only step {current + 1} is selectable"
            )
        indices[increment.unit_id] = increment.step
        return Allocation.from_mapping(self.signature, indices)

    def validate_selection(self, increment_ids: Iterable[str]) -> None:
        """Raise ChainConstraintError unless the selection is a prefix of each unit's ladder."""
        steps: dict[str, set[int]] = defaultdict(set)
        for increment_id in increment_ids:
            increment = self.increment(increment_id)
            if increment.step in steps[increment.unit_id]:
                raise ChainConstraintError(
                    f"increment {increment_id} appears more than once in the selection"
                )
            steps[increment.unit_id].add(increment.step)
        for unit_id, selected in steps.items():
            expected = set(range(1, len(selected) + 1))
            if selected != expected:
                missing = sorted(expected - selected)
                raise ChainConstraintError(
                    f"unit {unit_id} selection {sorted(selected)} skips step(s) {missing}; "
                    "an increment is only selectable once the one below it is selected"
                )

    def allocation_from_selection(self, increment_ids: Iterable[str]) -> Allocation:
        """Return the allocation implied by a set of increments, after validating the chain."""
        increment_ids = list(increment_ids)
        self.validate_selection(increment_ids)
        indices = {unit.unit_id: 0 for unit in self.units}
        for increment_id in increment_ids:
            increment = self.increment(increment_id)
            indices[increment.unit_id] = max(indices[increment.unit_id], increment.step)
        return Allocation.from_mapping(self.signature, indices)

    def selection_from_allocation(self, allocation: Allocation) -> frozenset[str]:
        """Return the set of increments equivalent to an allocation."""
        self._check_signature(allocation)
        selected: list[str] = []
        for unit_id, index in allocation.tier_index:
            for step in range(1, index + 1):
                selected.append(f"{unit_id}:{step}")
        return frozenset(selected)

    def tier_values(self, allocation: Allocation) -> dict[str, TierValue]:
        """Return the tier value held by each unit under an allocation."""
        self._check_signature(allocation)
        return {
            unit_id: self._unit_by_id[unit_id].tiers[index]
            for unit_id, index in allocation.tier_index
        }

    def weight_bits_by_expert(self, allocation: Allocation) -> dict[int, dict[int, int]]:
        """Return layer to expert index to bit width, for driving the quantization hooks."""
        self._check_signature(allocation)
        indices = allocation.as_dict()
        plan: dict[int, dict[int, int]] = defaultdict(dict)
        for unit in self.units:
            if unit.kind is not UnitKind.WEIGHT:
                continue
            bits = int(unit.tiers[indices[unit.unit_id]])
            for expert_index in unit.expert_indices:
                plan[unit.layer][expert_index] = bits
        return dict(plan)

    def kv_retention_by_layer(self, allocation: Allocation) -> dict[int, float]:
        """Return layer to retention ratio, for driving the KV cache hooks."""
        self._check_signature(allocation)
        indices = allocation.as_dict()
        return {
            unit.layer: float(unit.tiers[indices[unit.unit_id]])
            for unit in self.units
            if unit.kind is UnitKind.KV
        }

    def footprint(self, allocation: Allocation) -> Footprint:
        """Return the analytic byte footprint of an allocation."""
        self._check_signature(allocation)
        indices = allocation.as_dict()
        expert_total = 0
        kv_total = 0
        for unit in self.units:
            size = self.unit_bytes(unit, indices[unit.unit_id])
            if unit.kind is UnitKind.WEIGHT:
                expert_total += size
            else:
                kv_total += size
        return Footprint(
            fixed_weight_bytes=fixed_weight_bytes(self.model),
            expert_weight_bytes=expert_total,
            kv_bytes=kv_total,
        )

    def cost_bytes(self, allocation: Allocation) -> int:
        """Return the total footprint of an allocation in bytes."""
        return self.footprint(allocation).total_bytes

    def base_cost_bytes(self) -> int:
        """Return the footprint of the base state, which is subtracted from every budget."""
        return self.cost_bytes(self.base_allocation())

    def reference_bytes(self) -> int:
        """Return the footprint with every expert at the model dtype and every cache kept in full."""
        return reference_footprint(self.model, self.kv, self.quant).total_bytes

    def plan_budget(self, fraction: float) -> BudgetPlan:
        """Return the byte plan for a budget fraction of the reference footprint.

        Raises BudgetInfeasibleError when the base state already exceeds the
        budget, because no allocation is feasible in that case.
        """
        if fraction <= 0.0:
            raise ValueError(f"budget fraction must be positive, got {fraction}")
        reference = self.reference_bytes()
        budget = int(fraction * reference)
        base = self.base_cost_bytes()
        plan = BudgetPlan(
            fraction=fraction,
            reference_bytes=reference,
            budget_bytes=budget,
            base_bytes=base,
            max_bytes=self.cost_bytes(self.full_allocation()),
        )
        if plan.slack_bytes < 0:
            raise BudgetInfeasibleError(
                f"base state costs {format_bytes(base)} "
                f"({plan.base_fraction:.4f} of the reference footprint) but the budget at "
                f"fraction {fraction} is {format_bytes(budget)}. "
                "No allocation is feasible. Raise the budget fraction, lower the bottom "
                "tier, or widen the set of quantized parameters."
            )
        return plan

    def describe(self) -> dict[str, Any]:
        """Return a summary of the ground set for result records."""
        weight_units = [u for u in self.units if u.kind is UnitKind.WEIGHT]
        kv_units = [u for u in self.units if u.kind is UnitKind.KV]
        weight_increments = [i for i in self.increments if i.kind is UnitKind.WEIGHT]
        kv_increments = [i for i in self.increments if i.kind is UnitKind.KV]
        return {
            "signature": self.signature,
            "model_name": self.model.name,
            "num_units": len(self.units),
            "num_weight_units": len(weight_units),
            "num_kv_units": len(kv_units),
            "num_increments": len(self.increments),
            "num_weight_increments": len(weight_increments),
            "num_kv_increments": len(kv_increments),
            "weight_tiers": list(self.weight_tiers),
            "kv_tiers": list(self.kv_tiers),
            "weight_groups_per_layer": self.weight_groups_per_layer,
            "context_length": self.kv.context_length,
            "batch_size": self.kv.batch_size,
            "quant_group_size": self.quant.group_size,
            "base_bytes": self.base_cost_bytes(),
            "reference_bytes": self.reference_bytes(),
        }

    def _check_signature(self, allocation: Allocation) -> None:
        if allocation.signature != self.signature:
            raise ValueError(
                f"allocation belongs to ground set {allocation.signature!r}, "
                f"not {self.signature!r}"
            )
