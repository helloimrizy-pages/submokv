"""Tests for the per-expert allocation experiment."""

from __future__ import annotations

import json

import pytest
import torch

from submokv.ground_set import GroundSet, UnitKind
from submokv.memory import KVSpec, ModelSpec, QuantSpec
from submokv.perexpert import (
    ExpertSensitivity,
    diagnostic_1_per_expert,
    measure_expert_sensitivity,
    measure_router_load,
    sensitivity_ranked_allocation,
)

from conftest import TINY_ARCHITECTURE, _tiny_utility, build_tiny_model


def _per_expert_utility(model):
    """A tiny utility whose ground set puts every expert on its own ladder."""
    utility = _tiny_utility(model, cheap_sequences=1, pool_windows=8)
    spec = utility.ground_set.model
    utility.ground_set = GroundSet(
        model=spec,
        kv=utility.ground_set.kv,
        quant=utility.ground_set.quant,
        weight_groups_per_layer=spec.num_experts,
    )
    return utility


def test_the_router_is_read_where_it_is_computed(tiny_model) -> None:
    utility = _tiny_utility(tiny_model, cheap_sequences=2, pool_windows=6)
    sequences = utility.store.subsample("cheap", 0)
    frequency, tokens = measure_router_load(
        tiny_model, sequences, torch.device("cpu"), utility.batch_size, top_k=2
    )
    layers = TINY_ARCHITECTURE["num_hidden_layers"]
    experts = TINY_ARCHITECTURE["num_experts"]
    assert len(frequency) == layers * experts
    assert sum(frequency.values()) == pytest.approx(1.0)
    # Every position picks top_k experts in every layer, so the total is exact.
    positions = sequences.shape[0] * sequences.shape[1]
    assert tokens == positions * 2 * layers


def test_sensitivity_needs_both_error_and_load() -> None:
    """An expert nothing routes to scores zero however badly it quantizes."""
    sensitivity = ExpertSensitivity(
        frequency={(0, 0): 0.9, (0, 1): 0.0},
        relative_error={
            (0, 0, 3): 0.30, (0, 0, 4): 0.10,
            (0, 1, 3): 0.90, (0, 1, 4): 0.10,
        },
        num_layers=1, num_experts=2, tokens_routed=100, top_k=1,
    )
    assert sensitivity.score(0, 0, 3, 4) == pytest.approx(0.2 * 0.9)
    # Much larger error, but no traffic, so no benefit.
    assert sensitivity.score(0, 1, 3, 4) == 0.0
    assert sensitivity.describe()["router_load"]["zero_load_experts"] == 1


def test_the_ranked_allocation_buys_the_most_sensitive_experts_first(tiny_model) -> None:
    utility = _per_expert_utility(tiny_model)
    ground = utility.ground_set
    experts = ground.model.num_experts
    # Expert 0 of layer 0 is the only one worth anything.
    frequency = {(l, e): (1.0 if (l, e) == (0, 0) else 0.0)
                 for l in range(ground.model.num_hidden_layers) for e in range(experts)}
    error = {(l, e, t): {3: 0.4, 4: 0.2, 8: 0.05, 16: 0.0}[t]
             for l in range(ground.model.num_hidden_layers)
             for e in range(experts) for t in (3, 4, 8, 16)}
    sensitivity = ExpertSensitivity(frequency, error, ground.model.num_hidden_layers,
                                    experts, 100, 1)
    plan = ground.plan_budget(0.99)
    allocation, trace = sensitivity_ranked_allocation(ground, plan.budget_bytes, sensitivity)
    assert trace, "the ranked allocator bought nothing"
    # Only the unit with nonzero benefit is ever bought.
    assert {entry["increment_id"].split(":")[0] for entry in trace} == {"w.l00.g0"}
    # Benefit per byte is non-increasing down the list.
    scores = [entry["benefit_per_byte"] for entry in trace]
    assert scores == sorted(scores, reverse=True)
    assert ground.cost_bytes(allocation) <= plan.budget_bytes


def test_the_experiment_refuses_a_ground_set_that_is_not_per_expert(tiny_utility_wide) -> None:
    sensitivity = ExpertSensitivity({}, {}, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="per-expert granularity"):
        diagnostic_1_per_expert(tiny_utility_wide, 0.9, sensitivity, num_random=1)


def test_the_experiment_pairs_the_ranked_allocation_against_each_random_one(tiny_model) -> None:
    utility = _per_expert_utility(tiny_model)
    ground = utility.ground_set
    sensitivity = measure_expert_sensitivity(utility, ground.weight_tiers, "cheap", 0)
    report = diagnostic_1_per_expert(
        utility, 0.99, sensitivity, num_random=3, subsamples=(0, 1, 2)
    )
    labels = [row["label"] for row in report["allocations"]]
    assert labels[:2] == ["sensitivity_ranked", "uniform"]
    assert len([l for l in labels if l.startswith("random")]) == 3
    for row in report["allocations"]:
        assert len(row["perplexity_per_subsample"]) == 3
        assert row["within_budget"], f"{row['label']} exceeded the budget"
    # The comparison is paired: one difference per shared draw, per opponent.
    assert len(report["paired_against_random"]) == 3
    for entry in report["paired_against_random"]:
        assert len(entry["per_subsample"]) == 3
    summary = report["summary"]
    assert summary["ranked_beats_random_of"] == 3
    assert 0 <= summary["ranked_beats_random_count"] <= 3
    assert summary["paired_noise_floor"] >= 0.0
    json.dumps(report)


def test_every_expert_gets_its_own_ladder(tiny_model) -> None:
    ground = _per_expert_utility(tiny_model).ground_set
    weight_units = [u for u in ground.units if u.kind is UnitKind.WEIGHT]
    assert len(weight_units) == ground.model.num_hidden_layers * ground.model.num_experts
    assert all(len(u.expert_indices) == 1 for u in weight_units)
