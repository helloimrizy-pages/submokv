"""Tests for fake quantization of expert weights."""

from __future__ import annotations

import pytest
import torch

from submokv.memory import ModelSpec, QuantSpec, expert_bytes, matrix_bytes
from submokv.quantize import ExpertQuantizer, fake_quantize, find_expert_modules

TINY_QUANT = QuantSpec(group_size=8, scale_bits=16, zero_point_bits=0)


def test_fake_quantize_at_the_unquantized_tier_is_the_identity() -> None:
    weight = torch.randn(8, 32)
    assert torch.equal(fake_quantize(weight, 16, group_size=8), weight)


def test_fake_quantize_keeps_shape_and_dtype() -> None:
    weight = torch.randn(4, 8, 32, dtype=torch.float16)
    result = fake_quantize(weight, 4, group_size=8)
    assert result.shape == weight.shape
    assert result.dtype == weight.dtype


def test_fake_quantize_error_falls_as_bits_rise() -> None:
    torch.manual_seed(0)
    weight = torch.randn(16, 64)
    errors = [
        (fake_quantize(weight, bits, group_size=16) - weight).abs().mean().item()
        for bits in (2, 3, 4, 8)
    ]
    assert errors == sorted(errors, reverse=True)


def test_fake_quantize_uses_one_scale_per_group() -> None:
    """A large value in one group must not widen the scale of another group."""
    weight = torch.zeros(1, 8)
    weight[0, :4] = torch.tensor([1.0, -1.0, 0.5, -0.5])
    weight[0, 4:] = torch.tensor([1000.0, 0.0, 0.0, 0.0])
    result = fake_quantize(weight, 4, group_size=4)
    assert result[0, 0].item() == pytest.approx(1.0, rel=1e-3)
    assert result[0, 4].item() == pytest.approx(1000.0, rel=1e-3)


def test_fake_quantize_leaves_an_all_zero_group_at_zero() -> None:
    result = fake_quantize(torch.zeros(2, 8), 2, group_size=4)
    assert torch.equal(result, torch.zeros(2, 8))
    assert torch.isfinite(result).all()


def test_fake_quantize_pads_a_group_that_does_not_divide_the_row() -> None:
    torch.manual_seed(0)
    weight = torch.randn(2, 10)
    result = fake_quantize(weight, 4, group_size=4)
    assert result.shape == weight.shape
    assert torch.isfinite(result).all()


def test_two_bit_quantization_leaves_three_levels_per_group() -> None:
    torch.manual_seed(0)
    weight = torch.randn(1, 8)
    result = fake_quantize(weight, 2, group_size=8)
    assert len(torch.unique(result)) <= 3


def test_packed_expert_layout_costs_the_same_as_the_three_matrix_layout() -> None:
    """transformers packs gate and up into one tensor, which must not change the accounting."""
    model = ModelSpec(
        name="test",
        num_hidden_layers=16,
        hidden_size=2048,
        intermediate_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=64,
        vocab_size=50304,
        max_position_embeddings=4096,
    )
    quant = QuantSpec(group_size=128, scale_bits=16, zero_point_bits=0)
    for bits in (2, 3, 4, 8, 16):
        packed = matrix_bytes(2 * 1024, 2048, bits, quant) + matrix_bytes(2048, 1024, bits, quant)
        assert packed == expert_bytes(model, quant, bits, num_experts=1)


def test_find_expert_modules_returns_one_module_per_layer(tiny_model) -> None:
    modules = find_expert_modules(tiny_model)
    assert [entry.layer for entry in modules] == [0, 1, 2]
    assert all(entry.num_experts == 4 for entry in modules)


def test_the_unquantized_tier_reproduces_the_unmodified_logits_exactly(
    tiny_model, tiny_ids
) -> None:
    """Every expert at 16 bits must leave the model bit for bit unchanged."""
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    quantizer.set_uniform_bits(16)
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert torch.equal(before, after)


def test_two_bit_experts_change_the_logits(tiny_model, tiny_ids) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    quantizer.set_uniform_bits(2)
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert not torch.equal(before, after)


def test_restore_returns_the_model_to_its_unmodified_weights(tiny_model, tiny_ids) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    quantizer.set_uniform_bits(2)
    quantizer.restore()
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert torch.equal(before, after)


def test_setting_the_same_plan_twice_rewrites_nothing(tiny_model) -> None:
    """Moving one unit up its ladder must not requantize the whole model."""
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    assert quantizer.set_uniform_bits(4) == 12
    assert quantizer.set_uniform_bits(4) == 0
    assert quantizer.set_uniform_bits(4, layers=[1]) == 0
    assert quantizer.set_uniform_bits(8, layers=[1]) == 4


def test_quantizing_one_layer_leaves_everything_upstream_unchanged(
    tiny_model, tiny_ids
) -> None:
    """Layer 1 at 2 bits must not move the hidden states that feed it."""
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids, output_hidden_states=True).hidden_states
    quantizer.set_uniform_bits(2, layers=[1])
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids, output_hidden_states=True).hidden_states

    # hidden_states[i] is the input to layer i, so entries up to and including
    # the input of layer 1 are upstream of the change.
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert not torch.equal(before[2], after[2])
    assert not torch.equal(before[3], after[3])


def test_bit_widths_can_differ_between_experts_of_one_layer(tiny_model, tiny_ids) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    quantizer.set_plan({0: {0: 2, 1: 4, 2: 8, 3: 16}})
    plan = quantizer.current_plan()
    assert plan[0] == {0: 2, 1: 4, 2: 8, 3: 16}
    assert set(plan[1].values()) == {16}


def test_master_store_holds_one_copy_of_the_expert_weights(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    expected = sum(
        parameter.numel() * parameter.element_size()
        for entry in quantizer.expert_modules
        for name, parameter in entry.module.named_parameters(recurse=False)
    )
    assert quantizer.master_bytes() == expected


def test_plan_rejects_a_layer_that_holds_no_experts(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    with pytest.raises(KeyError):
        quantizer.set_plan({9: {0: 4}})


def test_quantizer_requires_attach_first(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    with pytest.raises(RuntimeError, match="attach"):
        quantizer.set_uniform_bits(4)


def test_asymmetric_quantization_is_costed_but_not_produced(tiny_model) -> None:
    asymmetric = QuantSpec(group_size=8, scale_bits=16, zero_point_bits=16, symmetric=False)
    with pytest.raises(NotImplementedError, match="symmetric"):
        ExpertQuantizer(tiny_model, asymmetric)


def test_describe_records_the_active_bit_widths(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT).attach()
    quantizer.set_uniform_bits(4)
    quantizer.set_uniform_bits(8, layers=[2])
    summary = quantizer.describe()
    assert summary["experts_per_bit_width"] == {"4": 8, "8": 4}
    assert summary["group_size"] == 8


def test_quantizing_a_stack_of_experts_matches_quantizing_them_one_at_a_time() -> None:
    """set_plan rewrites experts in batches, which must not change the numerics."""
    torch.manual_seed(0)
    stack = torch.randn(4, 8, 32)
    batched = fake_quantize(stack, 3, group_size=8)
    for index in range(stack.shape[0]):
        assert torch.equal(batched[index], fake_quantize(stack[index], 3, group_size=8))
