"""Diagnostics 0, 1, and 2: is there anything here for an allocator to allocate?

These run before any solver exists, because each one can end the line of work on
its own.

    0  sensitivity spread. Start at the top allocation, drop one unit at a time,
       and measure how much perplexity moves. If every layer's sensitivity curve
       is the same to within the noise floor, no allocation can beat a uniform
       one and nothing downstream matters.
    1  achievable headroom. Fix a budget, sample feasible allocations, and take
       the spread of perplexity across them. Best minus worst bounds what a
       perfect solver could ever win. Compared against the noise floor, and
       reported again with the worst decile dropped, because a floor tier that
       is catastrophic inflates the spread with a cliff that any heuristic
       avoids.
    2  cross-component interaction. Take a weight-only and a KV-only allocation
       and measure F(W and K) minus F(W) minus F(K). Near zero means the two
       axes are separable and a joint allocator is two independent sweeps. Large
       and positive means real synergy, which is also the supermodularity that
       breaks the greedy bound.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from .ground_set import Allocation, GroundSet, Increment, UnitKind
from .utility import CHEAP, UtilityFunction

# The 2-bit tier is outside the search space because its symmetric grid is
# ternary. It is probed here anyway, as the most aggressive setting available,
# so that a per-expert signal has the best chance of clearing the noise floor.
WEIGHT_PROBE_BITS: tuple[int, ...] = (2, 3, 4, 8)
KV_PROBE_RETENTION: tuple[float, ...] = (0.25, 0.50, 0.75)


def top_weight_plan(ground_set: GroundSet) -> dict[int, dict[int, int]]:
    """Return the plan holding every expert at the unquantized tier."""
    bits = ground_set.quant.unquantized_bits
    return {
        layer: {expert: bits for expert in range(ground_set.model.num_experts)}
        for layer in range(ground_set.model.num_hidden_layers)
    }


def top_retention(ground_set: GroundSet) -> dict[int, float]:
    """Return the retention holding every layer's cache in full."""
    return {layer: 1.0 for layer in range(ground_set.model.num_hidden_layers)}


def random_feasible_allocation(
    ground_set: GroundSet,
    budget_bytes: int,
    generator: torch.Generator,
    kinds: Iterable[UnitKind] | None = None,
) -> Allocation:
    """Return an allocation built by taking affordable increments at random until none fit.

    Every step respects the chain constraint, because the candidate pool holds
    at most one increment per unit.
    """
    allowed = set(kinds) if kinds is not None else None
    allocation = ground_set.base_allocation()
    spent = ground_set.cost_bytes(allocation)
    while True:
        candidates = [
            candidate
            for candidate in ground_set.candidates(allocation)
            if spent + candidate.cost_bytes <= budget_bytes
            and (allowed is None or candidate.kind in allowed)
        ]
        if not candidates:
            return allocation
        index = int(torch.randint(len(candidates), (1,), generator=generator).item())
        chosen = candidates[index]
        allocation = ground_set.apply(allocation, chosen)
        spent += chosen.cost_bytes


def uniform_allocation(
    ground_set: GroundSet,
    budget_bytes: int,
    kinds: Iterable[UnitKind] | None = None,
) -> Allocation:
    """Return the highest allocation that raises every unit of a kind in lockstep.

    A sweep is taken only when the whole sweep fits, so every unit of the kind
    ends at the same tier.
    """
    allowed = set(kinds) if kinds is not None else None
    allocation = ground_set.base_allocation()
    while True:
        candidates = [
            candidate
            for candidate in ground_set.candidates(allocation)
            if allowed is None or candidate.kind in allowed
        ]
        if not candidates:
            return allocation
        sweep = sum(candidate.cost_bytes for candidate in candidates)
        if ground_set.cost_bytes(allocation) + sweep > budget_bytes:
            return allocation
        for candidate in candidates:
            allocation = ground_set.apply(allocation, candidate)


def _spread(values: Sequence[float]) -> dict[str, float]:
    """Return the range, standard deviation, and extremes of a set of measurements."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def noise_floor_sweep(
    utility: UtilityFunction,
    sizes: Sequence[int],
    num_subsamples: int = 6,
) -> dict[str, Any]:
    """Measure the noise floor at several calibration sizes.

    The size that F runs at should be the smallest one whose noise floor sits
    well below the effects the diagnostics need to resolve, because every
    evaluation is paid for many times over.
    """
    import dataclasses

    ground_set = utility.ground_set
    original = utility.store.spec
    results: dict[str, Any] = {}
    try:
        for size in sizes:
            utility.store.spec = dataclasses.replace(original, cheap_sequences=size)
            floor = utility.noise_floor(ground_set.full_allocation(), num_subsamples, CHEAP)
            floor["sequences"] = size
            results[str(size)] = floor
    finally:
        utility.store.spec = original
    return results


def diagnostic_0_probes(
    ground_set: GroundSet,
    per_expert_layers: Sequence[int] = (0, 8),
    per_expert_sample: int = 8,
    expert_group_size: int = 8,
) -> list[dict[str, Any]]:
    """Return every probe Diagnostic 0 measures, in a fixed order.

    The order does not depend on which shard runs it, so splitting the list
    across workers gives each one a disjoint slice of the same experiment.
    """
    layers = list(range(ground_set.model.num_hidden_layers))
    num_experts = ground_set.model.num_experts
    top_bits = top_weight_plan(ground_set)
    top_kv = top_retention(ground_set)
    probes: list[dict[str, Any]] = []

    for layer in layers:
        for bits in WEIGHT_PROBE_BITS:
            plan = top_weight_plan(ground_set)
            plan[layer] = {expert: bits for expert in range(num_experts)}
            probes.append(
                {
                    "group": "per_layer_weight",
                    "key": f"d0.w.l{layer:02d}.b{bits}",
                    "weight_bits": plan,
                    "retention": dict(top_kv),
                    "row": {"layer": layer, "bits": bits, "in_ground_set": bits in ground_set.weight_tiers},
                }
            )

    for layer in layers:
        for ratio in KV_PROBE_RETENTION:
            retention = dict(top_kv)
            retention[layer] = ratio
            probes.append(
                {
                    "group": "per_layer_kv",
                    "key": f"d0.kv.l{layer:02d}.r{ratio}",
                    "weight_bits": top_bits,
                    "retention": retention,
                    "row": {"layer": layer, "retention": ratio},
                }
            )

    # A single expert of sixty four carries little of a layer, so the most
    # aggressive bit width is used to give the signal its best chance, and
    # contiguous groups are measured alongside single experts.
    for layer in per_expert_layers:
        for expert in range(min(per_expert_sample, num_experts)):
            plan = top_weight_plan(ground_set)
            plan[layer] = dict(plan[layer])
            plan[layer][expert] = WEIGHT_PROBE_BITS[0]
            probes.append(
                {
                    "group": "per_expert_weight",
                    "key": f"d0.we.l{layer:02d}.e{expert:02d}.b{WEIGHT_PROBE_BITS[0]}",
                    "weight_bits": plan,
                    "retention": dict(top_kv),
                    "row": {"layer": layer, "expert": expert, "bits": WEIGHT_PROBE_BITS[0]},
                }
            )
        for start in range(0, num_experts, expert_group_size):
            members = list(range(start, min(start + expert_group_size, num_experts)))
            plan = top_weight_plan(ground_set)
            plan[layer] = dict(plan[layer])
            for expert in members:
                plan[layer][expert] = WEIGHT_PROBE_BITS[1]
            probes.append(
                {
                    "group": "expert_group_weight",
                    "key": f"d0.wg.l{layer:02d}.g{start // expert_group_size}.b{WEIGHT_PROBE_BITS[1]}",
                    "weight_bits": plan,
                    "retention": dict(top_kv),
                    "row": {
                        "layer": layer,
                        "group": start // expert_group_size,
                        "experts": members,
                        "bits": WEIGHT_PROBE_BITS[1],
                    },
                }
            )
    return probes


def summarize_diagnostic_0(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the spread of every probe family, computed over whatever rows are present."""
    weight = payload.get("per_layer_weight", [])
    kv = payload.get("per_layer_kv", [])
    experts = payload.get("per_expert_weight", [])
    groups = payload.get("expert_group_weight", [])
    expert_layers = sorted({row["layer"] for row in experts} | {row["layer"] for row in groups})
    return {
        "weight_spread_by_bits": {
            str(bits): _spread([r["delta"] for r in weight if r["bits"] == bits])
            for bits in WEIGHT_PROBE_BITS
        },
        "kv_spread_by_retention": {
            str(ratio): _spread([r["delta"] for r in kv if r["retention"] == ratio])
            for ratio in KV_PROBE_RETENTION
        },
        "expert_spread_by_layer": {
            str(layer): _spread([r["delta"] for r in experts if r["layer"] == layer])
            for layer in expert_layers
        },
        "expert_group_spread_by_layer": {
            str(layer): _spread([r["delta"] for r in groups if r["layer"] == layer])
            for layer in expert_layers
        },
    }


def diagnostic_0_sensitivity(
    utility: UtilityFunction,
    fidelity: str = CHEAP,
    subsample: int = 0,
    per_expert_layers: Sequence[int] = (0, 8),
    per_expert_sample: int = 8,
    expert_group_size: int = 8,
    shard: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Measure how much perplexity moves when one unit is dropped from the top allocation.

    This is the necessary condition. If the sensitivity curves of the layers lie
    on top of each other to within the noise floor, there is nothing for any
    allocator to allocate.

    With num_shards above one, this measures only its own slice of the probe
    list. The slices are disjoint and their union is the whole experiment, so
    merge_diagnostic_0 rebuilds the full result from the shard records.
    """
    ground_set = utility.ground_set
    reference = utility.evaluate_plan(
        "d0.top", top_weight_plan(ground_set), top_retention(ground_set), fidelity, subsample
    ).perplexity

    probes = diagnostic_0_probes(
        ground_set, per_expert_layers, per_expert_sample, expert_group_size
    )
    payload: dict[str, Any] = {
        "per_layer_weight": [],
        "per_layer_kv": [],
        "per_expert_weight": [],
        "expert_group_weight": [],
    }
    for index, probe in enumerate(probes):
        if index % num_shards != shard:
            continue
        result = utility.evaluate_plan(
            probe["key"], probe["weight_bits"], probe["retention"], fidelity, subsample
        )
        row = {
            **probe["row"],
            "probe": probe["key"],
            "perplexity": result.perplexity,
            "delta": result.perplexity - reference,
        }
        payload[probe["group"]].append(row)

    return {
        "reference_perplexity": reference,
        "fidelity": fidelity,
        "subsample": subsample,
        "shard": shard,
        "num_shards": num_shards,
        "num_probes_total": len(probes),
        "num_probes_measured": sum(len(v) for v in payload.values()),
        **payload,
        "summary": summarize_diagnostic_0(payload),
    }


def merge_diagnostic_0(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine Diagnostic 0 shard payloads into the result the whole experiment would give."""
    if not shards:
        raise ValueError("no shards to merge")
    references = {round(s["reference_perplexity"], 12) for s in shards}
    if len(references) > 1:
        raise ValueError(
            f"shards disagree on the reference perplexity: {sorted(references)}; "
            "they were not measured on the same model, device, or subsample"
        )
    payload: dict[str, Any] = {
        group: sorted(
            (row for s in shards for row in s.get(group, [])),
            key=lambda r: r["probe"],
        )
        for group in ("per_layer_weight", "per_layer_kv", "per_expert_weight", "expert_group_weight")
    }
    measured = sum(len(v) for v in payload.values())
    expected = shards[0]["num_probes_total"]
    return {
        "reference_perplexity": shards[0]["reference_perplexity"],
        "fidelity": shards[0]["fidelity"],
        "subsample": shards[0]["subsample"],
        "merged_from_shards": sorted(s["shard"] for s in shards),
        "num_probes_total": expected,
        "num_probes_measured": measured,
        "complete": measured == expected,
        **payload,
        "summary": summarize_diagnostic_0(payload),
    }


def paired_noise_floor(
    utility: UtilityFunction,
    probes: Mapping[str, tuple[Mapping[int, Mapping[int, int]], Mapping[int, float]]],
    num_subsamples: int = 4,
    fidelity: str = CHEAP,
) -> dict[str, Any]:
    """Measure how far a drop-from-top difference moves with the calibration draw.

    Diagnostic 0 measures every probe on one calibration subsample, so the probe
    and the reference share their data and much of the noise cancels. The floor
    a difference must clear is the spread of the difference, not the spread of
    either perplexity on its own, and the two are not the same number.
    """
    ground_set = utility.ground_set
    available = min(num_subsamples, utility.store.max_subsamples(fidelity))
    references = [
        utility.evaluate_plan(
            "d0.top", top_weight_plan(ground_set), top_retention(ground_set), fidelity, index
        ).perplexity
        for index in range(available)
    ]
    rows: dict[str, Any] = {}
    for name, (weight_plan, retention) in probes.items():
        deltas = [
            utility.evaluate_plan(name, weight_plan, retention, fidelity, index).perplexity
            - references[index]
            for index in range(available)
        ]
        rows[name] = {"deltas": deltas, **_spread(deltas)}
    return {
        "num_subsamples": available,
        "reference_perplexity": _spread(references),
        "probes": rows,
    }


def diagnostic_1_headroom(
    utility: UtilityFunction,
    budget_fraction: float,
    num_samples: int = 30,
    seed: int = 0,
    fidelity: str = CHEAP,
    subsample: int = 0,
    drop_worst_fraction: float = 0.1,
    shard: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Measure the spread of perplexity across random feasible allocations at one budget.

    Best minus worst is the ceiling on what a perfect solver could win. The
    spread with the worst decile dropped is reported alongside it, because a
    floor tier that is catastrophic turns the headline spread into a measure of
    cliff avoidance rather than of allocation quality.
    """
    ground_set = utility.ground_set
    plan = ground_set.plan_budget(budget_fraction)

    rows: list[dict[str, Any]] = []
    for index in range(num_samples):
        if index % num_shards != shard:
            continue
        # Each sample carries its own seed, so sample i is the same allocation
        # whichever shard draws it and however many shards there are.
        generator = torch.Generator().manual_seed(seed * 1000003 + index)
        allocation = random_feasible_allocation(ground_set, plan.budget_bytes, generator)
        result = utility.evaluate(allocation, fidelity, subsample)
        rows.append(
            {
                "sample": index,
                "perplexity": result.perplexity,
                "cost_bytes": ground_set.cost_bytes(allocation),
                "num_increments": allocation.num_selected_increments,
                "weight_tiers": sorted(
                    {int(v) for k, v in ground_set.tier_values(allocation).items() if k.startswith("w.")}
                ),
                "kv_tiers": sorted(
                    {float(v) for k, v in ground_set.tier_values(allocation).items() if k.startswith("kv.")}
                ),
                "allocation": allocation.as_dict(),
            }
        )

    return {
        "budget_fraction": budget_fraction,
        "budget_bytes": plan.budget_bytes,
        "slack_bytes": plan.slack_bytes,
        "fidelity": fidelity,
        "subsample": subsample,
        "seed": seed,
        "shard": shard,
        "num_shards": num_shards,
        "num_samples_requested": num_samples,
        "drop_worst_fraction": drop_worst_fraction,
        "samples": rows,
        **summarize_diagnostic_1(rows, drop_worst_fraction),
    }


def summarize_diagnostic_1(
    rows: Sequence[Mapping[str, Any]], drop_worst_fraction: float = 0.1
) -> dict[str, Any]:
    """Return the spread of perplexity over sampled allocations, whole and trimmed.

    The trimmed spread drops the worst allocations, because a floor tier that is
    catastrophic turns the headline spread into a measure of cliff avoidance
    rather than of allocation quality.
    """
    values = sorted(row["perplexity"] for row in rows)
    if not values:
        return {"full_spread": _spread([]), "trimmed_spread": _spread([]), "dropped_worst": 0}
    keep = max(2, len(values) - max(1, int(round(drop_worst_fraction * len(values)))))
    # Higher perplexity is worse, so the worst allocations sit at the end.
    return {
        "full_spread": _spread(values),
        "trimmed_spread": _spread(values[:keep]),
        "dropped_worst": len(values) - keep,
    }


def merge_diagnostic_1(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine Diagnostic 1 shard payloads into the result the whole experiment would give."""
    if not shards:
        raise ValueError("no shards to merge")
    budgets = {s["budget_fraction"] for s in shards}
    if len(budgets) > 1:
        raise ValueError(f"shards disagree on the budget fraction: {sorted(budgets)}")
    rows = sorted((row for s in shards for row in s["samples"]), key=lambda r: r["sample"])
    drop = shards[0].get("drop_worst_fraction", 0.1)
    return {
        "budget_fraction": shards[0]["budget_fraction"],
        "budget_bytes": shards[0]["budget_bytes"],
        "slack_bytes": shards[0]["slack_bytes"],
        "fidelity": shards[0]["fidelity"],
        "subsample": shards[0]["subsample"],
        "seed": shards[0]["seed"],
        "merged_from_shards": sorted(s["shard"] for s in shards),
        "num_samples_requested": shards[0]["num_samples_requested"],
        "num_samples_measured": len(rows),
        "complete": len(rows) == shards[0]["num_samples_requested"],
        "samples": rows,
        **summarize_diagnostic_1(rows, drop),
    }


def diagnostic_2_interaction(
    utility: UtilityFunction,
    budget_fraction: float,
    fidelity: str = CHEAP,
    subsample: int = 0,
) -> dict[str, Any]:
    """Measure whether the weight axis and the KV axis add up or interact.

    The interaction term is F(W and K) minus F(W) minus F(K). Near zero means
    the axes are separable, so a joint allocator is two independent sweeps and
    is not a step beyond a KV-only method plus off the shelf weight
    quantization. Large and positive means real synergy, which is the same
    supermodularity that breaks the greedy bound.

    Two pairings are reported. The shared pairing splits the slack between the
    axes so that the union is feasible at the budget. The full pairing gives
    each axis the whole slack, so the union sits above the budget but the term
    measures separability without a budget getting in the way.
    """
    ground_set = utility.ground_set
    plan = ground_set.plan_budget(budget_fraction)
    base_cost = plan.base_bytes

    def union(weight_only: Allocation, kv_only: Allocation) -> Allocation:
        values = weight_only.as_dict()
        for unit_id, index in kv_only.as_dict().items():
            if unit_id.startswith("kv."):
                values[unit_id] = index
        return Allocation.from_mapping(ground_set.signature, values)

    pairings: dict[str, Any] = {}
    for label, share in (("shared_slack", 0.5), ("full_slack", 1.0)):
        allowance = base_cost + int(share * plan.slack_bytes)
        weight_only = uniform_allocation(ground_set, allowance, [UnitKind.WEIGHT])
        kv_only = uniform_allocation(ground_set, allowance, [UnitKind.KV])
        joint = union(weight_only, kv_only)

        utility_w = utility.utility(weight_only, fidelity, subsample)
        utility_k = utility.utility(kv_only, fidelity, subsample)
        utility_joint = utility.utility(joint, fidelity, subsample)
        interaction = utility_joint - utility_w - utility_k
        pairings[label] = {
            "allowance_bytes": allowance,
            "weight_only_tiers": sorted(
                {int(v) for k, v in ground_set.tier_values(weight_only).items() if k.startswith("w.")}
            ),
            "kv_only_tiers": sorted(
                {float(v) for k, v in ground_set.tier_values(kv_only).items() if k.startswith("kv.")}
            ),
            "joint_cost_bytes": ground_set.cost_bytes(joint),
            "over_budget": ground_set.cost_bytes(joint) > plan.budget_bytes,
            "utility_weight_only": utility_w,
            "utility_kv_only": utility_k,
            "utility_joint": utility_joint,
            "interaction": interaction,
            "interaction_share_of_joint": (
                interaction / utility_joint if utility_joint else float("nan")
            ),
        }

    return {
        "budget_fraction": budget_fraction,
        "budget_bytes": plan.budget_bytes,
        "slack_bytes": plan.slack_bytes,
        "batch_size": ground_set.kv.batch_size,
        "fidelity": fidelity,
        "subsample": subsample,
        "base_perplexity": utility.base_perplexity(fidelity, subsample),
        "pairings": pairings,
    }
