"""Tests that an allocation from the ground set drives both sets of hooks."""

from __future__ import annotations

import torch

from submokv.ground_set import GroundSet
from submokv.kvcache import RecencySinkPolicy, RetentionController, forward_with_retention
from submokv.memory import KVSpec, ModelSpec, QuantSpec
from submokv.quantize import ExpertQuantizer

from conftest import TINY_ARCHITECTURE

TINY_KV = KVSpec(context_length=16, batch_size=1, dtype_bytes=2, sink_tokens=2)
TINY_QUANT = QuantSpec(group_size=8, scale_bits=16, zero_point_bits=0)


def tiny_ground_set() -> GroundSet:
    """Return a ground set whose model spec matches the tiny test model."""
    model = ModelSpec(
        name="tiny",
        num_hidden_layers=TINY_ARCHITECTURE["num_hidden_layers"],
        hidden_size=TINY_ARCHITECTURE["hidden_size"],
        intermediate_size=TINY_ARCHITECTURE["intermediate_size"],
        num_attention_heads=TINY_ARCHITECTURE["num_attention_heads"],
        num_key_value_heads=TINY_ARCHITECTURE["num_key_value_heads"],
        num_experts=TINY_ARCHITECTURE["num_experts"],
        vocab_size=TINY_ARCHITECTURE["vocab_size"],
        max_position_embeddings=TINY_ARCHITECTURE["max_position_embeddings"],
    )
    return GroundSet(model=model, kv=TINY_KV, quant=TINY_QUANT)


def apply(ground_set, allocation, quantizer, controller) -> None:
    """Push an allocation into the quantizer and the retention controller."""
    quantizer.set_plan(ground_set.weight_bits_by_expert(allocation))
    controller.set_retention(ground_set.kv_retention_by_layer(allocation))


def test_the_top_allocation_reproduces_the_unmodified_model_exactly(
    tiny_model, tiny_ids
) -> None:
    """Every unit at its top tier is 16-bit weights and full retention, so nothing changes."""
    ground_set = tiny_ground_set()
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits

    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), TINY_KV
    ).attach()
    apply(ground_set, ground_set.full_allocation(), quantizer, controller)

    after = forward_with_retention(tiny_model, tiny_ids, controller)
    assert torch.equal(before, after)


def test_the_base_allocation_sets_the_lowest_tier_everywhere(tiny_model, tiny_ids) -> None:
    ground_set = tiny_ground_set()
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), TINY_KV
    ).attach()
    apply(ground_set, ground_set.base_allocation(), quantizer, controller)

    assert {bits for layer in quantizer.current_plan().values() for bits in layer.values()} == {2}
    assert set(controller.retention().values()) == {0.25}


def test_one_increment_moves_exactly_one_unit(tiny_model, tiny_ids) -> None:
    """A greedy step touches one ladder, so one layer is rewritten and the rest are untouched."""
    ground_set = tiny_ground_set()
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), TINY_KV
    ).attach()

    allocation = ground_set.base_allocation()
    apply(ground_set, allocation, quantizer, controller)

    allocation = ground_set.apply(allocation, ground_set.increment("w.l01:1"))
    changed = quantizer.set_plan(ground_set.weight_bits_by_expert(allocation))
    assert changed == TINY_ARCHITECTURE["num_experts"]
    assert set(quantizer.current_plan()[1].values()) == {3}
    assert set(quantizer.current_plan()[0].values()) == {2}

    allocation = ground_set.apply(allocation, ground_set.increment("kv.l02:1"))
    controller.set_retention(ground_set.kv_retention_by_layer(allocation))
    assert controller.retention() == {0: 0.25, 1: 0.25, 2: 0.50}


def test_a_heterogeneous_allocation_reaches_the_hooks(tiny_model, tiny_ids) -> None:
    ground_set = tiny_ground_set()
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), TINY_KV
    ).attach()
    allocation = ground_set.allocation_from_selection(
        ["w.l00:1", "w.l00:2", "w.l00:3", "kv.l02:1", "kv.l02:2"]
    )
    apply(ground_set, allocation, quantizer, controller)

    assert set(quantizer.current_plan()[0].values()) == {8}
    assert set(quantizer.current_plan()[1].values()) == {2}
    assert controller.retention() == {0: 0.25, 1: 0.25, 2: 0.75}
    logits = forward_with_retention(tiny_model, tiny_ids, controller)
    assert logits.shape == (1, 16, TINY_ARCHITECTURE["vocab_size"])
    assert bool(torch.isfinite(logits).all())
