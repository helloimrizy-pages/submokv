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


def test_diagnostic_0_runs_and_reports_a_spread(tiny_utility) -> None:
    """The whole diagnostic path runs, including the per expert granularity."""
    from submokv.diagnostics import diagnostic_0_sensitivity

    result = diagnostic_0_sensitivity(
        tiny_utility, per_expert_layers=(0,), per_expert_sample=2, expert_group_size=2
    )
    layers = tiny_utility.ground_set.model.num_hidden_layers
    assert len(result["per_layer_weight"]) == layers * 4
    assert len(result["per_layer_kv"]) == layers * 3
    assert len(result["per_expert_weight"]) == 2
    assert len(result["expert_group_weight"]) == 2
    assert result["reference_perplexity"] > 0
    assert set(result["summary"]["weight_spread_by_bits"]) == {"2", "3", "4", "8"}
    # The 2-bit probe sits outside the search space and is labelled as such.
    two_bit = [r for r in result["per_layer_weight"] if r["bits"] == 2]
    assert all(not r["in_ground_set"] for r in two_bit)
    assert all(r["in_ground_set"] for r in result["per_layer_weight"] if r["bits"] == 4)


def test_dropping_a_unit_from_the_top_does_not_improve_perplexity(tiny_utility) -> None:
    """A random model has no reason to prefer a lower tier, but the sign is worth pinning."""
    from submokv.diagnostics import diagnostic_0_sensitivity

    result = diagnostic_0_sensitivity(
        tiny_utility, per_expert_layers=(0,), per_expert_sample=1, expert_group_size=4
    )
    assert all(row["perplexity"] > 0 for row in result["per_layer_weight"])


def test_diagnostic_1_reports_full_and_trimmed_spread(tiny_utility) -> None:
    from submokv.diagnostics import diagnostic_1_headroom

    result = diagnostic_1_headroom(tiny_utility, budget_fraction=0.9, num_samples=6, seed=0)
    assert len(result["samples"]) == 6
    assert result["full_spread"]["count"] == 6
    assert result["dropped_worst"] >= 1
    assert result["trimmed_spread"]["count"] < result["full_spread"]["count"]
    assert result["trimmed_spread"]["range"] <= result["full_spread"]["range"]
    assert all(s["cost_bytes"] <= result["budget_bytes"] for s in result["samples"])


def test_diagnostic_2_reports_an_interaction_term(tiny_utility) -> None:
    from submokv.diagnostics import diagnostic_2_interaction

    result = diagnostic_2_interaction(tiny_utility, budget_fraction=0.9)
    assert set(result["pairings"]) == {"shared_slack", "full_slack"}
    shared = result["pairings"]["shared_slack"]
    expected = shared["utility_joint"] - shared["utility_weight_only"] - shared["utility_kv_only"]
    assert shared["interaction"] == pytest.approx(expected)
    assert not shared["over_budget"]
    assert result["batch_size"] == 1


def test_the_utility_is_zero_at_the_base_state(tiny_utility) -> None:
    """F(base) must be zero by construction, or every gain is measured from a moving point."""
    assert tiny_utility.utility(tiny_utility.ground_set.base_allocation()) == 0.0


def test_the_utility_memoizes_on_the_allocation(tiny_utility) -> None:
    allocation = tiny_utility.ground_set.full_allocation()
    tiny_utility.evaluate(allocation)
    before = tiny_utility.evaluation_count
    tiny_utility.evaluate(allocation)
    assert tiny_utility.evaluation_count == before
