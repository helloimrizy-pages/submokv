"""Does a sensitivity-driven per-expert bit assignment beat random at matched budget?

MoPEQ, MxMoE and BT-MoE all claim it does.  The existing Sub-MoKV evidence
against per-expert allocation is sixteen probes that each drop one expert of
sixty-four to 2 bits, which perturbs about 1.5% of a layer's expert weights.  A
small delta there is the expected outcome and refutes nothing at all: it says a
single expert does not matter, not that a *ranking over all* experts does not.

This measures the claim the literature actually makes.  Every expert in the
model is a unit on its own ladder, a sensitivity-ranked assignment is built
under one budget, and it is scored against random feasible allocations at the
same budget with a measured floor beside the spread.

The sensitivity score is the standard one in that literature and needs no model
evaluations to compute: how much a tier costs an expert in reconstruction
error, weighted by how often the router actually sends tokens to it.  A weight
that is badly quantized but rarely read is cheap to damage; one that is read
constantly is not.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .ground_set import Allocation, GroundSet, Increment, UnitKind
from .quantize import EXPERT_PARAMETER_NAMES, fake_quantize, find_expert_modules
from .utility import CHEAP, UtilityFunction


@dataclass(frozen=True)
class ExpertSensitivity:
    """Per-expert router load and per-tier reconstruction error."""

    frequency: dict[tuple[int, int], float]
    relative_error: dict[tuple[int, int, int], float]
    num_layers: int
    num_experts: int
    tokens_routed: int
    top_k: int

    def score(self, layer: int, expert: int, before: int, after: int) -> float:
        """Return the estimated benefit of moving one expert up one tier.

        The error a tier leaves behind, minus the error the next tier leaves
        behind, weighted by how much of the model's routed traffic this expert
        carries.  Both factors are needed: reducing the error of an expert
        nothing routes to buys nothing.
        """
        gain = self.relative_error[(layer, expert, before)] - self.relative_error[
            (layer, expert, after)
        ]
        return max(0.0, gain) * self.frequency[(layer, expert)]

    def describe(self) -> dict[str, Any]:
        """Return the sensitivity settings and spread for result records."""
        loads = sorted(self.frequency.values())
        return {
            "metric": (
                "relative reconstruction error removed by the tier step, times the share of "
                "routed tokens the expert receives"
            ),
            "num_layers": self.num_layers,
            "num_experts_per_layer": self.num_experts,
            "tokens_routed": self.tokens_routed,
            "top_k": self.top_k,
            "router_load": {
                "min": loads[0],
                "median": loads[len(loads) // 2],
                "max": loads[-1],
                "uniform_would_be": 1.0 / (self.num_layers * self.num_experts),
                "max_over_min": loads[-1] / loads[0] if loads[0] > 0 else float("inf"),
                "zero_load_experts": sum(1 for value in loads if value == 0.0),
            },
        }


def measure_router_load(
    model: Any,
    sequences: torch.Tensor,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> tuple[dict[tuple[int, int], float], int]:
    """Count how often each expert is selected, over a fixed set of sequences.

    The router is read where it is computed rather than inferred, so the count
    is the one the model actually used.
    """
    modules = find_expert_modules(model)
    gates: list[tuple[int, Any]] = []
    for entry in modules:
        parent = model.get_submodule(entry.name.rsplit(".", 1)[0])
        gate = getattr(parent, "gate", None)
        if gate is None:
            raise ValueError(
                f"no router found next to expert module {entry.name!r}; expected a sibling "
                "'gate' linear layer"
            )
        gates.append((entry.layer, gate))

    num_experts = modules[0].num_experts
    counts = torch.zeros(len(modules), num_experts, dtype=torch.float64)
    handles = []

    def make_hook(position: int) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            flat = logits.reshape(-1, logits.shape[-1])
            chosen = torch.topk(flat, k=top_k, dim=-1).indices.reshape(-1)
            counts[position] += torch.bincount(
                chosen.detach().cpu(), minlength=num_experts
            ).to(torch.float64)

        return hook

    for position, (_layer, gate) in enumerate(gates):
        handles.append(gate.register_forward_hook(make_hook(position)))
    try:
        with torch.no_grad():
            for begin in range(0, sequences.shape[0], batch_size):
                batch = sequences[begin : begin + batch_size].to(device)
                if batch.shape[0] != batch_size:
                    break
                model(batch)
    finally:
        for handle in handles:
            handle.remove()

    total = float(counts.sum().item())
    if total <= 0:
        raise ValueError("the router selected no experts; the hook read nothing")
    frequency = {
        (modules[position].layer, expert): float(counts[position, expert].item()) / total
        for position in range(len(modules))
        for expert in range(num_experts)
    }
    return frequency, int(total)


def measure_expert_sensitivity(
    utility: UtilityFunction,
    tiers: Sequence[int],
    fidelity: str = CHEAP,
    subsample: int = 0,
    top_k: int | None = None,
) -> ExpertSensitivity:
    """Return router load and per-tier reconstruction error for every expert.

    Neither factor costs a perplexity evaluation: the error is tensor algebra on
    the master weights, and the load is one forward pass.
    """
    ground = utility.ground_set
    model = utility.model
    quantizer = utility.quantizer
    sequences = utility.store.subsample(fidelity, subsample)
    k = int(top_k if top_k is not None else getattr(model.config, "num_experts_per_tok", 8))
    frequency, tokens = measure_router_load(
        model, sequences, utility.device, utility.batch_size, k
    )

    modules = find_expert_modules(model)
    num_experts = modules[0].num_experts
    relative: dict[tuple[int, int, int], float] = {}
    with torch.no_grad():
        for entry in modules:
            # A whole layer's expert stack is read and quantized in one call.
            # Quantization groups run along the last dimension and stay inside
            # one expert's rows, so quantizing the stack gives bit-for-bit what
            # quantizing each expert separately would, at a fraction of the
            # checkpoint reads: 2 per layer instead of 2 per expert.
            experts = list(range(num_experts))
            squared_error = {tier: torch.zeros(num_experts, dtype=torch.float64) for tier in tiers}
            squared_norm = torch.zeros(num_experts, dtype=torch.float64)
            for name in EXPERT_PARAMETER_NAMES:
                original = quantizer.master.read(
                    entry.layer, name, experts, torch.device("cpu"), torch.float32
                )
                flat = original.reshape(num_experts, -1).to(torch.float64)
                squared_norm += (flat**2).sum(dim=1)
                for tier in tiers:
                    approximated = fake_quantize(
                        original,
                        int(tier),
                        ground.quant.group_size,
                        ground.quant.unquantized_bits,
                    )
                    residual = (original - approximated).reshape(num_experts, -1)
                    squared_error[tier] += (residual.to(torch.float64) ** 2).sum(dim=1)
            for expert in range(num_experts):
                scale = math.sqrt(float(squared_norm[expert].item())) or 1.0
                for tier in tiers:
                    relative[(entry.layer, expert, int(tier))] = (
                        math.sqrt(float(squared_error[tier][expert].item())) / scale
                    )
    return ExpertSensitivity(
        frequency=frequency,
        relative_error=relative,
        num_layers=len(modules),
        num_experts=num_experts,
        tokens_routed=tokens,
        top_k=k,
    )


def _expert_of(unit_id: str, ground: GroundSet) -> tuple[int, int] | None:
    """Return the (layer, expert) a weight unit covers, when it covers exactly one."""
    unit = ground.unit(unit_id)
    if unit.kind is not UnitKind.WEIGHT or len(unit.expert_indices) != 1:
        return None
    return unit.layer, unit.expert_indices[0]


def sensitivity_ranked_allocation(
    ground: GroundSet,
    budget_bytes: int,
    sensitivity: ExpertSensitivity,
    kinds: Sequence[UnitKind] | None = None,
) -> tuple[Allocation, list[dict[str, Any]]]:
    """Spend the budget on the highest benefit-per-byte increment still affordable.

    This is the allocator the literature describes: rank every candidate tier
    step by estimated benefit divided by what it costs, and buy down the list.
    The chain constraint holds by construction because the candidate pool holds
    at most one increment per unit.
    """
    allowed = set(kinds) if kinds is not None else None
    allocation = ground.base_allocation()
    spent = ground.cost_bytes(allocation)
    trace: list[dict[str, Any]] = []

    def benefit(increment: Increment) -> float:
        if increment.kind is not UnitKind.WEIGHT:
            return 0.0
        located = _expert_of(increment.unit_id, ground)
        if located is None:
            return 0.0
        layer, expert = located
        return sensitivity.score(
            layer, expert, int(increment.from_tier), int(increment.to_tier)
        )

    while True:
        candidates = [
            candidate
            for candidate in ground.candidates(allocation)
            if spent + candidate.cost_bytes <= budget_bytes
            and (allowed is None or candidate.kind in allowed)
        ]
        if not candidates:
            break
        scored = [
            (benefit(candidate) / candidate.cost_bytes, candidate) for candidate in candidates
        ]
        # Ties are broken by increment id so the allocation is deterministic.
        best_score, chosen = max(scored, key=lambda pair: (pair[0], pair[1].increment_id))
        if best_score <= 0.0:
            break
        allocation = ground.apply(allocation, chosen)
        spent += chosen.cost_bytes
        trace.append(
            {
                "increment_id": chosen.increment_id,
                "benefit_per_byte": best_score,
                "cost_bytes": chosen.cost_bytes,
                "spent_bytes": spent,
            }
        )
    return allocation, trace


def _tier_histogram(ground: GroundSet, allocation: Allocation) -> dict[str, int]:
    values = ground.tier_values(allocation)
    histogram: dict[str, int] = {}
    for unit_id, tier in values.items():
        if unit_id.startswith("w."):
            key = str(int(tier))
            histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def diagnostic_1_per_expert(
    utility: UtilityFunction,
    budget_fraction: float,
    sensitivity: ExpertSensitivity,
    num_random: int = 30,
    seed: int = 0,
    fidelity: str = CHEAP,
    subsamples: Sequence[int] = (0, 1, 2),
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Score a sensitivity-ranked per-expert assignment against random ones.

    Every allocation is evaluated on the same calibration draws, so the
    comparison is paired and the floor it must clear is the spread of the
    *difference*, not the spread of either perplexity.
    """
    from .diagnostics import random_feasible_allocation, uniform_allocation

    ground = utility.ground_set
    if ground.weight_groups_per_layer != ground.model.num_experts:
        raise ValueError(
            f"this experiment is about per-expert granularity, but the ground set puts "
            f"{ground.weight_groups_per_layer} group(s) on each layer's {ground.model.num_experts} "
            "experts; set ground_set.weight_groups_per_layer to the expert count"
        )
    draws = tuple(int(index) for index in subsamples)
    plan = ground.plan_budget(budget_fraction)

    ranked, trace = sensitivity_ranked_allocation(ground, plan.budget_bytes, sensitivity)
    uniform = uniform_allocation(ground, plan.budget_bytes)
    contenders: list[tuple[str, Allocation]] = [("sensitivity_ranked", ranked), ("uniform", uniform)]
    for index in range(num_random):
        generator = torch.Generator().manual_seed(seed * 1000003 + index)
        contenders.append(
            (f"random_{index:02d}", random_feasible_allocation(ground, plan.budget_bytes, generator))
        )

    rows: list[dict[str, Any]] = []
    perplexities: dict[str, list[float]] = {}
    for position, (label, allocation) in enumerate(contenders, start=1):
        if progress is not None:
            progress(label, position, len(contenders))
        values = [
            utility.evaluate(allocation, fidelity, draw).perplexity for draw in draws
        ]
        perplexities[label] = values
        rows.append(
            {
                "label": label,
                "kind": "ranked" if label == "sensitivity_ranked" else (
                    "uniform" if label == "uniform" else "random"
                ),
                "perplexity_per_subsample": values,
                "perplexity_mean": statistics.fmean(values),
                "perplexity_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "cost_bytes": ground.cost_bytes(allocation),
                "within_budget": ground.cost_bytes(allocation) <= plan.budget_bytes,
                "num_increments": allocation.num_selected_increments,
                "weight_tier_histogram": _tier_histogram(ground, allocation),
            }
        )

    ranked_values = perplexities["sensitivity_ranked"]
    randoms = [row for row in rows if row["kind"] == "random"]
    random_means = [row["perplexity_mean"] for row in randoms]

    # Paired against each random allocation on the same draws, so the
    # calibration draw cancels exactly as it does in the interaction matrix.
    paired: list[dict[str, Any]] = []
    for row in randoms:
        differences = [
            other - mine
            for mine, other in zip(ranked_values, perplexities[row["label"]])
        ]
        paired.append(
            {
                "against": row["label"],
                # Positive means the ranked allocation has the lower perplexity.
                "advantage_mean": statistics.fmean(differences),
                "advantage_stdev": statistics.stdev(differences) if len(differences) > 1 else 0.0,
                "per_subsample": differences,
            }
        )
    advantages = [entry["advantage_mean"] for entry in paired]
    paired_floor = (
        statistics.fmean(entry["advantage_stdev"] for entry in paired) if paired else 0.0
    )
    ranked_mean = statistics.fmean(ranked_values)
    beaten = sum(1 for value in random_means if ranked_mean < value)

    return {
        "budget_fraction": budget_fraction,
        "budget_bytes": plan.budget_bytes,
        "slack_bytes": plan.slack_bytes,
        "granularity": {
            "weight_groups_per_layer": ground.weight_groups_per_layer,
            "num_weight_units": sum(
                1 for unit in ground.units if unit.kind is UnitKind.WEIGHT
            ),
            "one_unit_per_expert": True,
        },
        "calibration": {
            **utility.store.spec.describe(),
            "fidelity": fidelity,
            "subsamples": list(draws),
        },
        "sensitivity": sensitivity.describe(),
        "ranked_trace_head": trace[:20],
        "ranked_increments_bought": len(trace),
        "allocations": rows,
        "paired_against_random": paired,
        "summary": {
            "claim_under_test": (
                "MoPEQ / MxMoE / BT-MoE: a sensitivity-driven per-expert assignment beats "
                "uniform and random at matched budget"
            ),
            "ranked_perplexity_mean": ranked_mean,
            "uniform_perplexity_mean": statistics.fmean(perplexities["uniform"]),
            "random_perplexity_mean": statistics.fmean(random_means) if random_means else None,
            "random_perplexity_best": min(random_means) if random_means else None,
            "random_perplexity_worst": max(random_means) if random_means else None,
            "random_perplexity_stdev": (
                statistics.stdev(random_means) if len(random_means) > 1 else 0.0
            ),
            "ranked_beats_random_count": beaten,
            "ranked_beats_random_of": len(random_means),
            "ranked_beats_uniform": ranked_mean < statistics.fmean(perplexities["uniform"]),
            "mean_advantage_over_random": statistics.fmean(advantages) if advantages else None,
            "paired_noise_floor": paired_floor,
            "advantage_in_floor_units": (
                statistics.fmean(advantages) / paired_floor
                if paired_floor > 0 and advantages
                else None
            ),
        },
    }
