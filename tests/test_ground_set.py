"""Tests for units, tiers, increments, and the chain constraint."""

from __future__ import annotations

import pytest

from submokv.ground_set import (
    BudgetInfeasibleError,
    ChainConstraintError,
    GroundSet,
    UnitKind,
    UnknownIncrementError,
)
from submokv.memory import KVSpec, ModelSpec, QuantSpec

LAYERS = 16
WEIGHT_TIERS = (3, 4, 8, 16)
KV_TIERS = (0.25, 0.50, 0.75, 1.00)

# Base state, recomputed by hand. A 3-bit weight with one 16-bit scale per group
# of 128 costs 3.125 bits per element, which is 25/64 of a byte.
HAND_EXPERT_PARAMS_PER_LAYER = 64 * 3 * 1024 * 2048
HAND_BASE_EXPERT_BYTES = LAYERS * HAND_EXPERT_PARAMS_PER_LAYER * 25 // 64
HAND_FIXED_PARAMS = LAYERS * (4 * 2048 * 2048 + 2048 * 64 + 4 * 2048) + 2 * 50304 * 2048 + 2048
HAND_FIXED_BYTES = HAND_FIXED_PARAMS * 2
HAND_BASE_KV_BYTES = 4 * LAYERS * 2 * 1024 * 16 * 128 * 2
HAND_BASE_BYTES = HAND_BASE_EXPERT_BYTES + HAND_FIXED_BYTES + HAND_BASE_KV_BYTES


def test_unit_and_increment_counts(ground_set: GroundSet) -> None:
    """OLMoE-1B-7B has 16 layers, so the ground set holds 32 units and 96 increments."""
    assert len(ground_set.units) == 2 * LAYERS
    weight_units = [u for u in ground_set.units if u.kind is UnitKind.WEIGHT]
    kv_units = [u for u in ground_set.units if u.kind is UnitKind.KV]
    assert len(weight_units) == LAYERS
    assert len(kv_units) == LAYERS
    assert len(ground_set.increments) == LAYERS * (len(WEIGHT_TIERS) - 1 + len(KV_TIERS) - 1)
    assert len(ground_set.increments) == 96


def test_every_weight_unit_covers_all_experts_of_its_layer(ground_set: GroundSet) -> None:
    for unit in ground_set.units:
        if unit.kind is UnitKind.WEIGHT:
            assert unit.expert_indices == tuple(range(64))


def test_base_allocation_cost_matches_hand_computation(ground_set: GroundSet) -> None:
    footprint = ground_set.footprint(ground_set.base_allocation())
    assert footprint.expert_weight_bytes == HAND_BASE_EXPERT_BYTES
    assert footprint.fixed_weight_bytes == HAND_FIXED_BYTES
    assert footprint.kv_bytes == HAND_BASE_KV_BYTES
    assert ground_set.base_cost_bytes() == HAND_BASE_BYTES


def test_full_allocation_equals_the_reference_footprint(ground_set: GroundSet) -> None:
    assert ground_set.cost_bytes(ground_set.full_allocation()) == ground_set.reference_bytes()


def test_increment_costs_are_byte_deltas_that_telescope(ground_set: GroundSet) -> None:
    """Summing every increment cost must close the gap between the base and top states."""
    total = sum(increment.cost_bytes for increment in ground_set.increments)
    assert total == ground_set.cost_bytes(ground_set.full_allocation()) - ground_set.base_cost_bytes()


def test_increment_cost_is_a_delta_not_an_absolute_size(ground_set: GroundSet) -> None:
    increment = ground_set.increment("w.l00:1")
    unit = ground_set.unit("w.l00")
    assert increment.cost_bytes == ground_set.unit_bytes(unit, 1) - ground_set.unit_bytes(unit, 0)
    assert increment.cost_bytes < ground_set.unit_bytes(unit, 1)


def test_every_increment_costs_bytes(ground_set: GroundSet) -> None:
    assert all(increment.cost_bytes > 0 for increment in ground_set.increments)


def test_zero_cost_increment_is_rejected_at_construction(
    olmoe_model: ModelSpec, quant_spec: QuantSpec
) -> None:
    """Two retention ratios that round to the same token count would give a free increment."""
    kv = KVSpec(context_length=8, batch_size=1, dtype_bytes=2, sink_tokens=4)
    with pytest.raises(ValueError, match="must cost bytes"):
        GroundSet(model=olmoe_model, kv=kv, quant=quant_spec, kv_tiers=(0.25, 0.30, 1.0))


def test_candidates_hold_exactly_one_increment_per_unit(ground_set: GroundSet) -> None:
    candidates = ground_set.candidates(ground_set.base_allocation())
    assert len(candidates) == len(ground_set.units)
    assert len({increment.unit_id for increment in candidates}) == len(ground_set.units)
    assert all(increment.step == 1 for increment in candidates)


def test_candidates_advance_up_the_ladder(ground_set: GroundSet) -> None:
    allocation = ground_set.base_allocation()
    allocation = ground_set.apply(allocation, ground_set.increment("w.l00:1"))
    by_unit = {increment.unit_id: increment for increment in ground_set.candidates(allocation)}
    assert by_unit["w.l00"].increment_id == "w.l00:2"
    assert by_unit["w.l01"].increment_id == "w.l01:1"


def test_candidates_drop_units_that_reach_the_top_tier(ground_set: GroundSet) -> None:
    assert ground_set.candidates(ground_set.full_allocation()) == ()
    allocation = ground_set.base_allocation()
    for step in range(1, len(KV_TIERS)):
        allocation = ground_set.apply(allocation, ground_set.increment(f"kv.l00:{step}"))
    unit_ids = {increment.unit_id for increment in ground_set.candidates(allocation)}
    assert "kv.l00" not in unit_ids
    assert len(unit_ids) == len(ground_set.units) - 1


def test_apply_rejects_an_out_of_order_increment(ground_set: GroundSet) -> None:
    base = ground_set.base_allocation()
    with pytest.raises(ChainConstraintError, match="only step 1 is selectable"):
        ground_set.apply(base, ground_set.increment("w.l00:3"))


def test_apply_rejects_repeating_an_increment(ground_set: GroundSet) -> None:
    allocation = ground_set.apply(ground_set.base_allocation(), ground_set.increment("w.l00:1"))
    with pytest.raises(ChainConstraintError):
        ground_set.apply(allocation, ground_set.increment("w.l00:1"))


def test_validate_selection_accepts_a_ladder_prefix(ground_set: GroundSet) -> None:
    ground_set.validate_selection(["w.l00:1", "w.l00:2", "kv.l03:1"])


def test_validate_selection_rejects_a_skipped_step(ground_set: GroundSet) -> None:
    with pytest.raises(ChainConstraintError, match=r"skips step\(s\) \[1\]"):
        ground_set.validate_selection(["w.l00:2"])


def test_validate_selection_rejects_a_gap_in_the_middle(ground_set: GroundSet) -> None:
    with pytest.raises(ChainConstraintError, match=r"skips step\(s\) \[2\]"):
        ground_set.validate_selection(["w.l00:1", "w.l00:3"])


def test_validate_selection_rejects_a_duplicate(ground_set: GroundSet) -> None:
    with pytest.raises(ChainConstraintError, match="more than once"):
        ground_set.validate_selection(["w.l00:1", "w.l00:1"])


def test_validate_selection_rejects_an_unknown_increment(ground_set: GroundSet) -> None:
    with pytest.raises(UnknownIncrementError):
        ground_set.validate_selection(["w.l99:1"])


def test_allocation_and_selection_are_two_views_of_one_state(ground_set: GroundSet) -> None:
    selection = ["w.l00:1", "w.l00:2", "kv.l03:1", "kv.l03:2"]
    allocation = ground_set.allocation_from_selection(selection)
    assert ground_set.selection_from_allocation(allocation) == frozenset(selection)
    assert allocation.num_selected_increments == len(selection)


def test_allocation_from_selection_is_order_independent(ground_set: GroundSet) -> None:
    forward = ground_set.allocation_from_selection(["w.l00:1", "w.l00:2", "kv.l03:1"])
    reversed_order = ground_set.allocation_from_selection(["kv.l03:1", "w.l00:2", "w.l00:1"])
    assert forward == reversed_order
    assert forward.canonical_hash() == reversed_order.canonical_hash()


def test_canonical_hash_separates_distinct_allocations(ground_set: GroundSet) -> None:
    base = ground_set.base_allocation()
    stepped = ground_set.apply(base, ground_set.increment("w.l00:1"))
    assert base.canonical_hash() != stepped.canonical_hash()


def test_canonical_hash_separates_ground_sets(
    olmoe_model: ModelSpec, quant_spec: QuantSpec
) -> None:
    """A different declared context length is a different problem, so caches must not be shared."""
    short = GroundSet(olmoe_model, KVSpec(context_length=4096), quant_spec)
    long = GroundSet(olmoe_model, KVSpec(context_length=8192), quant_spec)
    assert short.signature != long.signature
    assert short.base_allocation().canonical_hash() != long.base_allocation().canonical_hash()


def test_allocation_from_another_ground_set_is_rejected(
    olmoe_model: ModelSpec, quant_spec: QuantSpec, ground_set: GroundSet
) -> None:
    other = GroundSet(olmoe_model, KVSpec(context_length=8192), quant_spec)
    with pytest.raises(ValueError, match="belongs to ground set"):
        ground_set.footprint(other.base_allocation())


def test_tier_plans_drive_the_hooks(ground_set: GroundSet) -> None:
    allocation = ground_set.allocation_from_selection(["w.l00:1", "w.l00:2", "kv.l03:1"])
    weight_plan = ground_set.weight_bits_by_expert(allocation)
    assert weight_plan[0][0] == 8
    assert weight_plan[1][0] == 3
    retention = ground_set.kv_retention_by_layer(allocation)
    assert retention[3] == 0.50
    assert retention[0] == 0.25
    assert len(retention) == LAYERS


def test_weight_groups_split_a_layer_into_separate_ladders(
    olmoe_model: ModelSpec, kv_spec: KVSpec, quant_spec: QuantSpec
) -> None:
    grouped = GroundSet(olmoe_model, kv_spec, quant_spec, weight_groups_per_layer=4)
    weight_units = [u for u in grouped.units if u.kind is UnitKind.WEIGHT]
    assert len(weight_units) == 4 * LAYERS
    assert weight_units[0].unit_id == "w.l00.g0"
    covered = [i for u in weight_units[:4] for i in u.expert_indices]
    assert sorted(covered) == list(range(64))
    assert grouped.reference_bytes() == GroundSet(olmoe_model, kv_spec, quant_spec).reference_bytes()


def test_plan_budget_reports_slack(ground_set: GroundSet) -> None:
    plan = ground_set.plan_budget(0.35)
    assert plan.base_bytes == HAND_BASE_BYTES
    assert plan.budget_bytes == int(0.35 * ground_set.reference_bytes())
    assert plan.slack_bytes == plan.budget_bytes - plan.base_bytes
    assert plan.slack_bytes > 0


def test_plan_budget_fails_loudly_when_the_base_state_exceeds_the_budget(
    ground_set: GroundSet,
) -> None:
    with pytest.raises(BudgetInfeasibleError, match="No allocation is feasible"):
        ground_set.plan_budget(0.20)


def test_the_lowest_requested_budget_is_infeasible_by_a_small_margin(
    ground_set: GroundSet,
) -> None:
    """The 3-bit floor puts the base at 0.2507, just above the 0.25 budget."""
    reference = ground_set.reference_bytes()
    shortfall = ground_set.base_cost_bytes() - int(0.25 * reference)
    assert 0 < shortfall < 0.001 * reference
    with pytest.raises(BudgetInfeasibleError):
        ground_set.plan_budget(0.25)


def test_base_state_fraction_bounds_the_usable_budget_range(ground_set: GroundSet) -> None:
    """The unquantized parameters put a hard floor under every budget."""
    base_fraction = ground_set.base_cost_bytes() / ground_set.reference_bytes()
    assert 0.250 < base_fraction < 0.251


def test_ground_set_from_config_matches_direct_construction(ground_set: GroundSet) -> None:
    config = {
        "model": {
            "name": "allenai/OLMoE-1B-7B-0924",
            "num_hidden_layers": 16,
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "vocab_size": 50304,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
            "num_experts_per_tok": 8,
        },
        "kv": {"context_length": 4096, "batch_size": 4, "dtype_bytes": 2, "sink_tokens": 4},
        "quant": {"group_size": 128, "scale_bits": 16, "zero_point_bits": 0},
        "ground_set": {"weight_tiers": [3, 4, 8, 16], "kv_tiers": [0.25, 0.5, 0.75, 1.0]},
    }
    assert GroundSet.from_config(config).signature == ground_set.signature


def test_shipped_config_builds_the_expected_ground_set() -> None:
    """The config on disk is a deliverable, so it is checked against the same numbers."""
    from pathlib import Path

    from submokv.cli import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "olmoe.yaml")
    built = GroundSet.from_config(config)
    assert len(built.increments) == 96
    assert built.base_cost_bytes() == HAND_BASE_BYTES
    assert config["budgets"]["fractions"] == [0.25, 0.35, 0.50, 0.70]
