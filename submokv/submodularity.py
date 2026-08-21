"""Four-state tests for diminishing returns in joint weight/KV allocation.

Each interaction holds every unmentioned layer at the full 16-bit/100% state,
compresses a target and a conditioning component, and evaluates the four
corners of the resulting lattice square::

    S_A             S_A + j
    S_B             S_B + j

``S_B`` differs from ``S_A`` only by upgrading the conditioning component and
``j`` upgrades the target component.  Consequently ``S_A`` is a subset of
``S_B`` in the allocation lattice.  With utility defined as negative
perplexity degradation, the diminishing-returns difference is

    [F(S_A + j) - F(S_A)] - [F(S_B + j) - F(S_B)].

Positive values are submodular; negative values are synergy spikes.  The
implementation uses raw per-layer plans rather than the allocator ground set
because the diagnostic deliberately probes 2-bit weights even when that tier
is too destructive to include in the eventual allocator.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .utility import CHEAP, UtilityFunction

WEIGHT = "weight"
KV = "kv"

WEIGHT_TIERS: tuple[int, ...] = (2, 3, 4, 8, 16)
KV_TIERS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

WEIGHT_TO_WEIGHT = "weight_to_weight"
KV_TO_KV = "kv_to_kv"
WEIGHT_GIVEN_KV = "weight_given_kv"
KV_GIVEN_WEIGHT = "kv_given_weight"

MODALITY_LABELS = {
    WEIGHT_TO_WEIGHT: "W|W",
    KV_TO_KV: "KV|KV",
    WEIGHT_GIVEN_KV: "W|KV",
    KV_GIVEN_WEIGHT: "KV|W",
}


def _number(value: int | float) -> int | float:
    """Return integral values as ints so identifiers and JSON stay readable."""
    return int(value) if float(value).is_integer() else float(value)


def _tier_text(value: int | float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}"


def _tier_slug(value: int | float) -> str:
    return _tier_text(value).replace(".", "p")


@dataclass(frozen=True)
class TierUpgrade:
    """One target or conditioning move on a weight or KV tier ladder."""

    kind: str
    before: int | float
    after: int | float

    def __post_init__(self) -> None:
        if self.kind not in (WEIGHT, KV):
            raise ValueError(f"unknown component kind {self.kind!r}")
        allowed: tuple[int | float, ...] = WEIGHT_TIERS if self.kind == WEIGHT else KV_TIERS
        before = _number(self.before)
        after = _number(self.after)
        if before not in allowed or after not in allowed:
            raise ValueError(
                f"{self.kind} upgrade {before}->{after} is outside supported tiers {allowed}"
            )
        if before >= after:
            raise ValueError(f"an upgrade must increase its tier, got {before}->{after}")
        if self.kind == WEIGHT and (not isinstance(before, int) or not isinstance(after, int)):
            raise ValueError("weight bit widths must be integers")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    @property
    def short_kind(self) -> str:
        return "W" if self.kind == WEIGHT else "KV"

    @property
    def label(self) -> str:
        return f"{self.short_kind}:{_tier_text(self.before)}->{_tier_text(self.after)}"

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "from": self.before, "to": self.after}


@dataclass(frozen=True)
class InteractionSpec:
    """A target upgrade measured with one conditioning component low and high."""

    target_layer: int
    target: TierUpgrade
    conditioning_layer: int
    conditioning: TierUpgrade

    def __post_init__(self) -> None:
        if self.target_layer < 0 or self.conditioning_layer < 0:
            raise ValueError("layer indices must be non-negative")
        if (
            self.target.kind == self.conditioning.kind
            and self.target_layer == self.conditioning_layer
        ):
            raise ValueError("an intra-component interaction needs two distinct layers")

    @property
    def modality(self) -> str:
        pair = (self.target.kind, self.conditioning.kind)
        return {
            (WEIGHT, WEIGHT): WEIGHT_TO_WEIGHT,
            (KV, KV): KV_TO_KV,
            (WEIGHT, KV): WEIGHT_GIVEN_KV,
            (KV, WEIGHT): KV_GIVEN_WEIGHT,
        }[pair]

    @property
    def interaction_id(self) -> str:
        target = (
            f"{self.target.short_kind.lower()}{_tier_slug(self.target.before)}"
            f"to{_tier_slug(self.target.after)}"
        )
        context = (
            f"{self.conditioning.short_kind.lower()}{_tier_slug(self.conditioning.before)}"
            f"to{_tier_slug(self.conditioning.after)}"
        )
        return (
            f"{self.modality}.t{self.target_layer:02d}.{target}."
            f"c{self.conditioning_layer:02d}.{context}"
        )

    @property
    def label(self) -> str:
        return (
            f"L{self.target_layer} ({self.target.label}) | "
            f"L{self.conditioning_layer} ({self.conditioning.label})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "modality": self.modality,
            "target_layer": self.target_layer,
            "target_upgrade": self.target.as_dict(),
            "conditioning_layer": self.conditioning_layer,
            "conditioning_upgrade": self.conditioning.as_dict(),
            "label": self.label,
        }


@dataclass(frozen=True)
class DiagnosticPlan:
    """A uniform expert bit width and KV ratio for every decoder layer."""

    weight_bits: tuple[int, ...]
    kv_retention: tuple[float, ...]

    @classmethod
    def full(cls, num_layers: int) -> "DiagnosticPlan":
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        return cls((16,) * num_layers, (1.0,) * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.weight_bits)

    def with_value(self, layer: int, kind: str, value: int | float) -> "DiagnosticPlan":
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} is outside [0, {self.num_layers - 1}]")
        if kind == WEIGHT:
            bits = list(self.weight_bits)
            bits[layer] = int(value)
            return DiagnosticPlan(tuple(bits), self.kv_retention)
        if kind == KV:
            retention = list(self.kv_retention)
            retention[layer] = float(value)
            return DiagnosticPlan(self.weight_bits, tuple(retention))
        raise ValueError(f"unknown component kind {kind!r}")

    def expanded_weight_plan(self, num_experts: int) -> dict[int, dict[int, int]]:
        return {
            layer: {expert: bits for expert in range(num_experts)}
            for layer, bits in enumerate(self.weight_bits)
        }

    def expanded_retention_plan(self) -> dict[int, float]:
        return {layer: ratio for layer, ratio in enumerate(self.kv_retention)}

    def compact_dict(self) -> dict[str, dict[str, int | float]]:
        """Return only layers that differ from the full-precision reference."""
        return {
            "weight_bits": {
                str(layer): bits
                for layer, bits in enumerate(self.weight_bits)
                if bits != WEIGHT_TIERS[-1]
            },
            "kv_retention": {
                str(layer): ratio
                for layer, ratio in enumerate(self.kv_retention)
                if ratio != KV_TIERS[-1]
            },
        }

    def is_no_greater_than(self, other: "DiagnosticPlan") -> bool:
        if self.num_layers != other.num_layers:
            return False
        return all(a <= b for a, b in zip(self.weight_bits, other.weight_bits)) and all(
            a <= b for a, b in zip(self.kv_retention, other.kv_retention)
        )


@dataclass(frozen=True)
class FourStatePlans:
    """The four allocations needed for one second-order marginal difference."""

    s_a: DiagnosticPlan
    s_a_union_j: DiagnosticPlan
    s_b: DiagnosticPlan
    s_b_union_j: DiagnosticPlan

    def __post_init__(self) -> None:
        if not self.s_a.is_no_greater_than(self.s_b):
            raise ValueError("S_A must be no greater than S_B")
        if not self.s_a_union_j.is_no_greater_than(self.s_b_union_j):
            raise ValueError("S_A+j must be no greater than S_B+j")


def four_state_plans(spec: InteractionSpec, num_layers: int) -> FourStatePlans:
    """Construct the lattice square for an interaction around the full model."""
    base = DiagnosticPlan.full(num_layers)
    s_a = base.with_value(
        spec.target_layer, spec.target.kind, spec.target.before
    ).with_value(
        spec.conditioning_layer,
        spec.conditioning.kind,
        spec.conditioning.before,
    )
    s_a_union_j = s_a.with_value(spec.target_layer, spec.target.kind, spec.target.after)
    s_b = s_a.with_value(
        spec.conditioning_layer,
        spec.conditioning.kind,
        spec.conditioning.after,
    )
    s_b_union_j = s_b.with_value(spec.target_layer, spec.target.kind, spec.target.after)
    return FourStatePlans(s_a, s_a_union_j, s_b, s_b_union_j)


def adjacent_upgrades(kind: str) -> tuple[TierUpgrade, ...]:
    """Return every adjacent move on one supported tier ladder."""
    tiers: Sequence[int | float]
    if kind == WEIGHT:
        tiers = WEIGHT_TIERS
    elif kind == KV:
        tiers = KV_TIERS
    else:
        raise ValueError(f"unknown component kind {kind!r}")
    return tuple(TierUpgrade(kind, before, after) for before, after in zip(tiers, tiers[1:]))


def parse_upgrades(text: str, kind: str) -> tuple[TierUpgrade, ...]:
    """Parse comma-separated ``from:to`` transitions for a CLI."""
    parsed: list[TierUpgrade] = []
    for part in text.split(","):
        values = part.strip().split(":")
        if len(values) != 2:
            raise ValueError(
                f"invalid {kind} transition {part!r}; expected comma-separated FROM:TO pairs"
            )
        try:
            before, after = (float(value) for value in values)
        except ValueError as error:
            raise ValueError(f"invalid numeric transition {part!r}") from error
        parsed.append(TierUpgrade(kind, before, after))
    if not parsed:
        raise ValueError(f"at least one {kind} transition is required")
    return tuple(parsed)


def build_interaction_matrix(
    layers: Sequence[int],
    weight_upgrades: Sequence[TierUpgrade] | None = None,
    kv_upgrades: Sequence[TierUpgrade] | None = None,
) -> tuple[InteractionSpec, ...]:
    """Return cross- and intra-component pairwise tests in deterministic order.

    For four layers and one target transition on each ladder this produces 32
    interactions: 12 directed W|W pairs, 12 directed KV|KV pairs, and both
    cross-component directions at each of the four layers.
    """
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("at least one test layer is required")
    if len(set(selected)) != len(selected):
        raise ValueError(f"test layers must be unique, got {selected}")
    if any(layer < 0 for layer in selected):
        raise ValueError(f"test layers must be non-negative, got {selected}")

    weight_moves = tuple(weight_upgrades or (TierUpgrade(WEIGHT, 2, 4),))
    kv_moves = tuple(kv_upgrades or (TierUpgrade(KV, 0.25, 0.50),))
    if any(move.kind != WEIGHT for move in weight_moves):
        raise ValueError("weight_upgrades contains a non-weight transition")
    if any(move.kind != KV for move in kv_moves):
        raise ValueError("kv_upgrades contains a non-KV transition")

    full_weight_move = TierUpgrade(WEIGHT, WEIGHT_TIERS[0], WEIGHT_TIERS[-1])
    full_kv_move = TierUpgrade(KV, KV_TIERS[0], KV_TIERS[-1])
    interactions: list[InteractionSpec] = []

    # Cross-component tests first: these are the central Sub-MoKV hypothesis.
    for layer in selected:
        interactions.extend(
            InteractionSpec(layer, move, layer, full_kv_move) for move in weight_moves
        )
        interactions.extend(
            InteractionSpec(layer, move, layer, full_weight_move) for move in kv_moves
        )

    # Directed pairs matter: layer A conditioned on B need not equal B conditioned on A.
    for target_layer in selected:
        for conditioning_layer in selected:
            if target_layer == conditioning_layer:
                continue
            interactions.extend(
                InteractionSpec(target_layer, move, conditioning_layer, full_weight_move)
                for move in weight_moves
            )
    for target_layer in selected:
        for conditioning_layer in selected:
            if target_layer == conditioning_layer:
                continue
            interactions.extend(
                InteractionSpec(target_layer, move, conditioning_layer, full_kv_move)
                for move in kv_moves
            )
    return tuple(interactions)


def validate_test_layers(layers: Sequence[int], num_layers: int) -> tuple[int, ...]:
    """Validate user/config layer indices against the loaded architecture."""
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("at least one test layer is required")
    if len(set(selected)) != len(selected):
        raise ValueError(f"test layers must be unique, got {selected}")
    invalid = [layer for layer in selected if not 0 <= layer < num_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} do not exist in a {num_layers}-layer model; valid indices are "
            f"0 through {num_layers - 1}"
        )
    return selected


def classify_difference(difference: float, epsilon: float) -> str:
    """Classify one second-order difference using the requested tolerance."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    return "submodular" if difference >= -epsilon else "supermodular"


def _pairwise_gamma(rows: Sequence[Mapping[str, Any]], epsilon: float) -> float | None:
    """Return a conservative empirical pairwise diminishing-returns ratio.

    This is deliberately called *pairwise* gamma: it is not the global
    submodularity ratio over arbitrary sets.  Only cases with a positive
    enhanced-state marginal constrain the ratio.  A non-positive base-state
    marginal against a positive enhanced marginal yields zero.
    """
    ratios: list[float] = []
    for row in rows:
        marginal_a = float(row["marginal_gain_s_a"])
        marginal_b = float(row["marginal_gain_s_b"])
        if marginal_b > epsilon:
            ratios.append(max(0.0, min(1.0, marginal_a / marginal_b)))
    return min(ratios) if ratios else None


def summarize_interactions(
    rows: Sequence[Mapping[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    """Aggregate classifications, monotonicity checks, and the paper decision."""
    total = len(rows)
    submodular = sum(row["classification"] == "submodular" for row in rows)
    significant_supermodular = total - submodular
    strict_supermodular = sum(float(row["second_order_difference"]) < 0.0 for row in rows)
    epsilon_band = strict_supermodular - significant_supermodular
    negative_marginals = sum(
        float(row[key]) < -epsilon
        for row in rows
        for key in ("marginal_gain_s_a", "marginal_gain_s_b")
    )
    denominator = 2 * total

    by_modality: dict[str, Any] = {}
    for modality in MODALITY_LABELS:
        subset = [row for row in rows if row["modality"] == modality]
        if not subset:
            continue
        count = len(subset)
        passes = sum(row["classification"] == "submodular" for row in subset)
        by_modality[modality] = {
            "total": count,
            "submodular": passes,
            "significant_supermodular": count - passes,
            "strict_supermodular": sum(
                float(row["second_order_difference"]) < 0.0 for row in subset
            ),
            "submodular_rate": passes / count,
            "significant_supermodular_rate": (count - passes) / count,
            "strict_supermodular_rate": sum(
                float(row["second_order_difference"]) < 0.0 for row in subset
            )
            / count,
            "pairwise_gamma": _pairwise_gamma(subset, epsilon),
        }

    submodular_rate = submodular / total if total else 0.0
    significant_violation_rate = significant_supermodular / total if total else 0.0
    strict_violation_rate = strict_supermodular / total if total else 0.0
    core_cross = by_modality.get(WEIGHT_GIVEN_KV)
    core_submodular_rate = (
        float(core_cross["submodular_rate"]) if core_cross else submodular_rate
    )
    core_strict_violation_rate = (
        float(core_cross["strict_supermodular_rate"])
        if core_cross
        else strict_violation_rate
    )
    core_significant_violation_rate = (
        float(core_cross["significant_supermodular_rate"])
        if core_cross
        else significant_violation_rate
    )
    # A core W|KV synergy signal must not be diluted by the more numerous
    # intra-component rows. Use the conservative side of the overall and core
    # rates for the paper decision.
    decision_violation_rate = max(strict_violation_rate, core_strict_violation_rate)
    decision_significant_rate = max(
        significant_violation_rate, core_significant_violation_rate
    )
    decision_submodular_rate = min(submodular_rate, core_submodular_rate)
    if not total:
        verdict = "INCONCLUSIVE / NO INTERACTIONS"
        action = "No paper decision: run at least one interaction."
    elif decision_violation_rate > 0.30:
        verdict = (
            "SUPERMODULAR / SYNERGY-DOMINATED"
            if decision_significant_rate > 0.30
            else "SUPERMODULAR BY STRICT RULE / EPSILON-SENSITIVE"
        )
        action = (
            "Pivot to an empirical interaction paper; the strict >30% supermodularity "
            "criterion is met."
        )
    elif decision_submodular_rate > 0.70:
        verdict = "NEAR-SUBMODULAR / WEAKLY SUBMODULAR"
        action = (
            "Proceed with the method-paper hypothesis, then estimate a formal submodularity "
            "ratio before claiming greedy bounds."
        )
    else:
        verdict = "BORDERLINE / INCONCLUSIVE"
        action = "Increase the sample size or epsilon calibration before choosing a paper direction."

    return {
        "total_interactions": total,
        "submodular_pairs": submodular,
        "supermodular_pairs": strict_supermodular,
        "significant_supermodular_pairs": significant_supermodular,
        "submodular_rate": submodular_rate,
        "supermodular_violation_rate": strict_violation_rate,
        "significant_supermodular_rate": significant_violation_rate,
        "strict_supermodular_pairs": strict_supermodular,
        "strict_supermodular_rate": strict_violation_rate,
        "epsilon_band_pairs": epsilon_band,
        "negative_marginals": negative_marginals,
        "total_marginals": denominator,
        "monotonicity_violation_rate": negative_marginals / denominator if denominator else 0.0,
        "pairwise_gamma": _pairwise_gamma(rows, epsilon),
        "by_modality": by_modality,
        "decision_scope": {
            "rule": "conservative maximum violation / minimum compliance over overall and W|KV",
            "submodular_rate": decision_submodular_rate,
            "supermodular_violation_rate": decision_violation_rate,
            "significant_supermodular_rate": decision_significant_rate,
        },
        "verdict": verdict,
        "action": action,
    }


def _cache_namespace(utility: UtilityFunction) -> str:
    """Fingerprint settings omitted from UtilityFunction's raw-plan cache key."""
    payload = {
        "schema": "submokv.submodularity.v1",
        "ground_set": utility.ground_set.signature,
        "model": utility.ground_set.model.name,
        "model_source": str(getattr(utility.model.config, "_name_or_path", "unknown")),
        "calibration": utility.store.spec.describe(),
        "tokenizer_source": utility.store.tokenizer_source,
        "policy": utility.controller.policy.describe(),
        "protocol": utility.controller.protocol.describe(),
        "master_store": utility.quantizer.master.describe(),
        "batch_size": utility.batch_size,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _plan_cache_key(namespace: str, plan: DiagnosticPlan) -> str:
    payload = json.dumps(
        {
            "namespace": namespace,
            "weight_bits": plan.weight_bits,
            "kv_retention": plan.kv_retention,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "submod.v1." + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_submodularity_diagnostic(
    utility: UtilityFunction,
    interactions: Sequence[InteractionSpec],
    *,
    epsilon: float = 0.01,
    fidelity: str = CHEAP,
    subsample: int = 0,
    use_cache: bool = True,
    progress: Callable[[int, int, InteractionSpec], None] | None = None,
) -> dict[str, Any]:
    """Evaluate an interaction matrix and return a JSON-serializable report."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    ground = utility.ground_set
    num_layers = ground.model.num_hidden_layers
    num_experts = ground.model.num_experts
    matrix = tuple(interactions)
    for spec in matrix:
        validate_test_layers(
            tuple(dict.fromkeys((spec.target_layer, spec.conditioning_layer))), num_layers
        )

    namespace = _cache_namespace(utility)
    measurements: dict[DiagnosticPlan, Any] = {}
    evaluations_before = utility.evaluation_count

    def evaluate(plan: DiagnosticPlan) -> Any:
        result = measurements.get(plan)
        if result is None:
            result = utility.evaluate_plan(
                _plan_cache_key(namespace, plan),
                plan.expanded_weight_plan(num_experts),
                plan.expanded_retention_plan(),
                fidelity=fidelity,
                subsample=subsample,
                use_cache=use_cache,
            )
            measurements[plan] = result
        return result

    reference_plan = DiagnosticPlan.full(num_layers)
    rows: list[dict[str, Any]] = []
    try:
        reference = evaluate(reference_plan)
        for index, spec in enumerate(matrix, start=1):
            if progress is not None:
                progress(index, len(matrix), spec)
            plans = four_state_plans(spec, num_layers)
            state_plans = {
                "s_a": plans.s_a,
                "s_a_union_j": plans.s_a_union_j,
                "s_b": plans.s_b,
                "s_b_union_j": plans.s_b_union_j,
            }
            state_results = {name: evaluate(plan) for name, plan in state_plans.items()}

            states: dict[str, Any] = {}
            for name, result in state_results.items():
                degradation = result.perplexity - reference.perplexity
                states[name] = {
                    **result.as_dict(),
                    "perplexity_degradation": degradation,
                    "utility": -degradation,
                    "allocation_delta_from_full": state_plans[name].compact_dict(),
                }

            marginal_a = states["s_a_union_j"]["utility"] - states["s_a"]["utility"]
            marginal_b = states["s_b_union_j"]["utility"] - states["s_b"]["utility"]
            difference = marginal_a - marginal_b
            rows.append(
                {
                    **spec.as_dict(),
                    "states": states,
                    "marginal_gain_s_a": marginal_a,
                    "marginal_gain_s_b": marginal_b,
                    "second_order_difference": difference,
                    "classification": classify_difference(difference, epsilon),
                    "strict_supermodular": difference < 0.0,
                    "within_epsilon": abs(difference) <= epsilon,
                    "monotone_s_a": marginal_a >= -epsilon,
                    "monotone_s_b": marginal_b >= -epsilon,
                }
            )
    finally:
        # Leave the shared model in its unmodified state even if an evaluation fails.
        utility.quantizer.restore()
        utility.controller.set_uniform_retention(1.0)

    summary = summarize_interactions(rows, epsilon)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": {
            "utility": "F(S) = PPL(full 16-bit/100% KV) - PPL(S) = -DeltaPPL",
            "marginal_s_a": "F(S_A union {j}) - F(S_A)",
            "marginal_s_b": "F(S_B union {j}) - F(S_B)",
            "second_order_difference": "marginal_s_a - marginal_s_b",
            "submodular_test": "second_order_difference >= -epsilon",
        },
        "model": {
            "name": ground.model.name,
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
        },
        "calibration": {
            **utility.store.spec.describe(),
            "fidelity": fidelity,
            "subsample": subsample,
        },
        "test_layers": sorted(
            {spec.target_layer for spec in matrix}
            | {spec.conditioning_layer for spec in matrix}
        ),
        "epsilon_ppl": epsilon,
        "reference": {
            **reference.as_dict(),
            "utility": 0.0,
            "allocation_delta_from_full": reference_plan.compact_dict(),
        },
        "interactions": rows,
        "summary": summary,
        "execution": {
            "matrix_interactions": len(matrix),
            "unique_allocation_states": len(measurements),
            "new_model_evaluations": utility.evaluation_count - evaluations_before,
            "evaluation_cache_enabled": use_cache,
            "cache_namespace": namespace,
            "setup": utility.describe(),
        },
    }


def merge_submodularity_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge disjoint interaction shards after checking experiment identity.

    Each accelerator evaluates the same full-precision reference and a
    modulo-indexed slice of the interaction matrix.  The reference check is a
    guard against combining results from different model revisions, datasets,
    numerical environments, or subsamples.
    """
    shards = tuple(reports)
    if not shards:
        raise ValueError("no submodularity shard reports were provided")

    executions = [report.get("execution", {}) for report in shards]
    shard_counts = {int(execution.get("num_shards", 1)) for execution in executions}
    if len(shard_counts) != 1:
        raise ValueError(f"shards disagree on num_shards: {sorted(shard_counts)}")
    num_shards = shard_counts.pop()
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    shard_indices = [int(execution.get("shard", 0)) for execution in executions]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError(f"duplicate shard indices: {sorted(shard_indices)}")
    expected_shards = set(range(num_shards))
    if set(shard_indices) != expected_shards:
        missing = sorted(expected_shards - set(shard_indices))
        extra = sorted(set(shard_indices) - expected_shards)
        raise ValueError(f"incomplete shard set; missing={missing}, unexpected={extra}")

    def canonical(report: Mapping[str, Any], key: str) -> str:
        return json.dumps(report.get(key), sort_keys=True, separators=(",", ":"))

    for key in ("definition", "model", "calibration", "test_layers", "epsilon_ppl"):
        values = {canonical(report, key) for report in shards}
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}")

    references = {
        round(float(report["reference"]["perplexity"]), 12) for report in shards
    }
    reference_nlls = {
        round(float(report["reference"]["mean_nll"]), 12) for report in shards
    }
    if len(references) != 1 or len(reference_nlls) != 1:
        raise ValueError(
            "shards disagree on the full-precision reference; do not merge results "
            "from different models, devices, or calibration samples"
        )

    orders = [tuple(execution.get("full_matrix_order", ())) for execution in executions]
    if not orders[0] or any(order != orders[0] for order in orders[1:]):
        raise ValueError("shards disagree on the full interaction matrix order")
    full_order = orders[0]
    expected_total = len(full_order)
    declared_totals = {
        int(execution.get("full_matrix_interactions", -1)) for execution in executions
    }
    if declared_totals != {expected_total}:
        raise ValueError(
            f"shards disagree on the full matrix size: {sorted(declared_totals)} "
            f"versus order length {expected_total}"
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    for report in shards:
        for row in report.get("interactions", []):
            interaction_id = str(row["interaction_id"])
            if interaction_id in by_id:
                raise ValueError(f"interaction {interaction_id!r} appears in more than one shard")
            by_id[interaction_id] = row
    missing_interactions = [interaction_id for interaction_id in full_order if interaction_id not in by_id]
    unexpected_interactions = sorted(set(by_id) - set(full_order))
    if missing_interactions or unexpected_interactions:
        raise ValueError(
            "incomplete interaction union; "
            f"missing={missing_interactions}, unexpected={unexpected_interactions}"
        )
    rows = [deepcopy(by_id[interaction_id]) for interaction_id in full_order]

    first = shards[0]
    epsilon = float(first["epsilon_ppl"])
    merged = {
        "schema_version": first.get("schema_version", 1),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": deepcopy(first["definition"]),
        "model": deepcopy(first["model"]),
        "calibration": deepcopy(first["calibration"]),
        "test_layers": deepcopy(first["test_layers"]),
        "epsilon_ppl": epsilon,
        "reference": deepcopy(first["reference"]),
        "interactions": rows,
        "summary": summarize_interactions(rows, epsilon),
        "execution": {
            "matrix_interactions": expected_total,
            "full_matrix_interactions": expected_total,
            "merged_from_shards": sorted(shard_indices),
            "num_shards": num_shards,
            "new_model_evaluations": sum(
                int(execution.get("new_model_evaluations", 0)) for execution in executions
            ),
            "shard_unique_allocation_states_sum": sum(
                int(execution.get("unique_allocation_states", 0)) for execution in executions
            ),
            "evaluation_cache_enabled": all(
                bool(execution.get("evaluation_cache_enabled", False))
                for execution in executions
            ),
            "full_matrix_order": list(full_order),
            "shards": [deepcopy(execution) for execution in executions],
        },
    }
    return merged


def _signed(value: float, width: int = 10) -> str:
    if not math.isfinite(value):
        return f"{value:+g}".rjust(width)
    return f"{value:+.4f}".rjust(width)


def format_diagnostic_report(report: Mapping[str, Any]) -> str:
    """Render the requested terminal table from a diagnostic report."""
    width = 124
    model = report["model"]
    calibration = report["calibration"]
    summary = report["summary"]
    lines = [
        "=" * width,
        "SUB-MoKV SUBMODULARITY DIAGNOSTIC REPORT".center(width),
        "=" * width,
    ]
    execution = report.get("execution", {})
    if int(execution.get("num_shards", 1)) > 1 and "merged_from_shards" not in execution:
        lines.append(
            (
                f"PARTIAL SHARD {int(execution.get('shard', 0)) + 1}/"
                f"{execution['num_shards']} — VERDICT IS PRELIMINARY UNTIL MERGED"
            ).center(width)
        )
    lines.extend(
        [
            (
                f"Model: {model['name']} | Sequences: {report['reference']['num_sequences']} | "
                f"Length: {calibration['sequence_length']} | "
                f"epsilon: {report['epsilon_ppl']:.4g} PPL"
            ),
            "-" * width,
            (
                f"{'Layer Pair / Modality':<59} | {'Delta(j|S_A)':>12} | "
                f"{'Delta(j|S_B)':>12} | {'Diff (A-B)':>12} | Classification"
            ),
            "-" * width,
        ]
    )
    for row in report["interactions"]:
        descriptor = f"[{MODALITY_LABELS[row['modality']]}] {row['label']}"
        classification = (
            "Submodular" if row["classification"] == "submodular" else "SUPERMODULAR *"
        )
        lines.append(
            f"{descriptor:<59} | {_signed(row['marginal_gain_s_a'], 12)} | "
            f"{_signed(row['marginal_gain_s_b'], 12)} | "
            f"{_signed(row['second_order_difference'], 12)} | {classification}"
        )
    lines.extend(["-" * width, "SUMMARY:"])
    total = summary["total_interactions"]
    lines.extend(
        [
            f"- Total Interactions Tested: {total}",
            (
                f"- Submodular Pairs: {summary['submodular_pairs']} "
                f"({100.0 * summary['submodular_rate']:.2f}%)"
            ),
            (
                f"- Supermodular Pairs (strict violations): "
                f"{summary['supermodular_pairs']} "
                f"({100.0 * summary['supermodular_violation_rate']:.2f}%)"
            ),
            (
                f"- Significant Violations Beyond Epsilon: "
                f"{summary['significant_supermodular_pairs']} "
                f"({100.0 * summary['significant_supermodular_rate']:.2f}%; "
                f"{summary['epsilon_band_pairs']} strict violations fall inside epsilon)"
            ),
            (
                f"- Negative Marginal Gains: {summary['negative_marginals']}/"
                f"{summary['total_marginals']}"
            ),
        ]
    )
    gamma = summary["pairwise_gamma"]
    core = summary["by_modality"].get(WEIGHT_GIVEN_KV)
    if core is not None:
        lines.append(
            f"- Core W|KV Strict Violations: {core['strict_supermodular']}/{core['total']} "
            f"({100.0 * core['strict_supermodular_rate']:.2f}%)"
        )
    lines.append(
        "- Empirical Pairwise Gamma: " + ("n/a" if gamma is None else f"{gamma:.4f}")
    )
    lines.extend(
        [
            "",
            f"DECISION VERDICT: {summary['verdict']}",
            f"-> Action: {summary['action']}",
            "=" * width,
        ]
    )
    return "\n".join(lines)


def matrix_as_dict(interactions: Iterable[InteractionSpec]) -> list[dict[str, Any]]:
    """Return a compact JSON-ready matrix, useful for a no-model dry run."""
    return [spec.as_dict() for spec in interactions]
