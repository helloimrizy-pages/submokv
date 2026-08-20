"""Tests for the diagnostic helpers that do not need a model."""

from __future__ import annotations

import pytest
import torch

from submokv.diagnostics import (
    _spread,
    random_feasible_allocation,
    top_retention,
    top_weight_plan,
    uniform_allocation,
)
from submokv.ground_set import GroundSet, UnitKind

BUDGET_FRACTION = 0.35


def budget_bytes(ground_set: GroundSet) -> int:
    """Return the byte budget used across these tests."""
    return ground_set.plan_budget(BUDGET_FRACTION).budget_bytes


def test_random_feasible_allocation_stays_within_the_budget(ground_set: GroundSet) -> None:
    limit = budget_bytes(ground_set)
    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        allocation = random_feasible_allocation(ground_set, limit, generator)
        assert ground_set.cost_bytes(allocation) <= limit


def test_random_feasible_allocation_never_breaks_the_chain(ground_set: GroundSet) -> None:
    generator = torch.Generator().manual_seed(0)
    allocation = random_feasible_allocation(ground_set, budget_bytes(ground_set), generator)
    ground_set.validate_selection(ground_set.selection_from_allocation(allocation))


def test_random_feasible_allocation_spends_what_it_can(ground_set: GroundSet) -> None:
    """Sampling stops only when no next increment fits, so the budget is filled."""
    limit = budget_bytes(ground_set)
    generator = torch.Generator().manual_seed(0)
    allocation = random_feasible_allocation(ground_set, limit, generator)
    spent = ground_set.cost_bytes(allocation)
    remaining = ground_set.candidates(allocation)
    assert all(spent + increment.cost_bytes > limit for increment in remaining)


def test_random_feasible_allocation_is_reproducible(ground_set: GroundSet) -> None:
    limit = budget_bytes(ground_set)
    first = random_feasible_allocation(ground_set, limit, torch.Generator().manual_seed(7))
    second = random_feasible_allocation(ground_set, limit, torch.Generator().manual_seed(7))
    assert first == second


def test_random_feasible_allocation_can_be_held_to_one_axis(ground_set: GroundSet) -> None:
    generator = torch.Generator().manual_seed(0)
    allocation = random_feasible_allocation(
        ground_set, budget_bytes(ground_set), generator, kinds=[UnitKind.KV]
    )
    tiers = ground_set.tier_values(allocation)
    assert all(tiers[unit_id] == 3 for unit_id in tiers if unit_id.startswith("w."))
    assert any(tiers[unit_id] > 0.25 for unit_id in tiers if unit_id.startswith("kv."))


def test_uniform_allocation_puts_every_unit_of_a_kind_on_one_tier(ground_set: GroundSet) -> None:
    allocation = uniform_allocation(ground_set, budget_bytes(ground_set), [UnitKind.WEIGHT])
    tiers = ground_set.tier_values(allocation)
    weight_tiers = {tiers[unit_id] for unit_id in tiers if unit_id.startswith("w.")}
    assert len(weight_tiers) == 1
    assert ground_set.cost_bytes(allocation) <= budget_bytes(ground_set)


def test_uniform_allocation_leaves_the_other_axis_at_the_floor(ground_set: GroundSet) -> None:
    allocation = uniform_allocation(ground_set, budget_bytes(ground_set), [UnitKind.KV])
    tiers = ground_set.tier_values(allocation)
    assert {tiers[u] for u in tiers if u.startswith("w.")} == {3}
    assert len({tiers[u] for u in tiers if u.startswith("kv.")}) == 1


def test_top_plans_cover_every_layer_and_expert(ground_set: GroundSet) -> None:
    plan = top_weight_plan(ground_set)
    assert len(plan) == ground_set.model.num_hidden_layers
    assert set(plan[0].values()) == {16}
    assert len(plan[0]) == ground_set.model.num_experts
    assert set(top_retention(ground_set).values()) == {1.0}


def test_spread_reports_range_and_deviation() -> None:
    result = _spread([1.0, 2.0, 4.0])
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["range"] == 3.0
    assert result["count"] == 3
    assert _spread([])["count"] == 0
