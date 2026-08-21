"""Tests for fake quantization of expert weights."""

from __future__ import annotations

import pytest
import torch

from submokv.memory import ModelSpec, QuantSpec, expert_bytes, matrix_bytes
from submokv.quantize import (
    CheckpointMasterStore,
    ExpertQuantizer,
    fake_quantize,
    find_expert_modules,
)

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
    """Packing gate and up into one tensor must not change the byte count.

    This checks bytes only. It cannot detect a wrong grouping axis, because
    OLMoE's dimensions make the group count come out the same either way:
    gate_up_proj is square, and for down_proj 2048 * ceil(1024 / 128) equals
    1024 * ceil(2048 / 128). The axis is pinned by the numerics test below.
    """
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
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    quantizer.set_uniform_bits(16)
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert torch.equal(before, after)


def test_two_bit_experts_change_the_logits(tiny_model, tiny_ids) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    quantizer.set_uniform_bits(2)
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert not torch.equal(before, after)


def test_restore_returns_the_model_to_its_unmodified_weights(tiny_model, tiny_ids) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    quantizer.set_uniform_bits(2)
    quantizer.restore()
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert torch.equal(before, after)


def test_setting_the_same_plan_twice_rewrites_nothing(tiny_model) -> None:
    """Moving one unit up its ladder must not requantize the whole model."""
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    assert quantizer.set_uniform_bits(4) == 12
    assert quantizer.set_uniform_bits(4) == 0
    assert quantizer.set_uniform_bits(4, layers=[1]) == 0
    assert quantizer.set_uniform_bits(8, layers=[1]) == 4


def test_quantizing_one_layer_leaves_everything_upstream_unchanged(
    tiny_model, tiny_ids
) -> None:
    """Layer 1 at 2 bits must not move the hidden states that feed it."""
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
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
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    quantizer.set_plan({0: {0: 2, 1: 4, 2: 8, 3: 16}})
    plan = quantizer.current_plan()
    assert plan[0] == {0: 2, 1: 4, 2: 8, 3: 16}
    assert set(plan[1].values()) == {16}


def test_memory_master_store_holds_one_copy_of_the_expert_weights(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    expected = sum(
        parameter.numel() * parameter.element_size()
        for entry in quantizer.expert_modules
        for name, parameter in entry.module.named_parameters(recurse=False)
    )
    assert quantizer.master.resident_bytes() == expected


def test_plan_rejects_a_layer_that_holds_no_experts(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
    with pytest.raises(KeyError):
        quantizer.set_plan({9: {0: 4}})


def test_asymmetric_quantization_is_costed_but_not_produced(tiny_model) -> None:
    asymmetric = QuantSpec(group_size=8, scale_bits=16, zero_point_bits=16, symmetric=False)
    with pytest.raises(NotImplementedError, match="symmetric"):
        ExpertQuantizer(tiny_model, asymmetric)


def test_describe_records_the_active_bit_widths(tiny_model) -> None:
    quantizer = ExpertQuantizer(tiny_model, TINY_QUANT)
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


def test_quantization_groups_run_along_the_input_dimension() -> None:
    """An outlier must widen the scale of its own input group and no other.

    Grouping along the output dimension would give the same byte count for this
    model, so only the numerics can tell the two apart.
    """
    torch.manual_seed(0)
    weight = torch.randn(4, 256) * 0.01
    clean = fake_quantize(weight, 4, group_size=128)

    spiked = weight.clone()
    spiked[0, 0] = 100.0
    result = fake_quantize(spiked, 4, group_size=128)

    # The outlier shares a group with columns 0 to 127 of its own row, so those
    # entries lose precision.
    spoiled = (result[0, 1:128] - weight[0, 1:128]).abs().max()
    intact = (clean[0, 1:128] - weight[0, 1:128]).abs().max()
    assert spoiled > intact

    # Columns 128 onward are a different group of the same row, so they are
    # untouched. A single scale for the whole tensor would fail here.
    assert torch.equal(result[0, 128:], clean[0, 128:])

    # Every other row is untouched. Grouping along the output dimension would
    # put these rows in the outlier's group and fail here.
    assert torch.equal(result[1:], clean[1:])


def test_the_two_bit_tier_is_ternary() -> None:
    """A symmetric grid of 2 ** (bits - 1) - 1 levels leaves one of four codes unused."""
    torch.manual_seed(0)
    weight = torch.randn(1, 128)
    levels = torch.unique(fake_quantize(weight, 2, group_size=128))
    assert len(levels) == 3


def test_checkpoint_store_detects_and_repacks_legacy_mixtral_weights(tmp_path) -> None:
    prefix = "model.layers.0.block_sparse_moe.experts.0"
    tensors = {
        f"{prefix}.w1.weight": torch.full((2, 3), 1.0),
        f"{prefix}.w3.weight": torch.full((2, 3), 3.0),
        f"{prefix}.w2.weight": torch.full((3, 2), 2.0),
    }
    index = {"weight_map": {key: "model.safetensors" for key in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(__import__("json").dumps(index))
    store = CheckpointMasterStore(tmp_path)
    store._tensor = tensors.__getitem__  # type: ignore[method-assign]

    gate_up = store.read(0, "gate_up_proj", [0], torch.device("cpu"), torch.float32)
    down = store.read(0, "down_proj", [0], torch.device("cpu"), torch.float32)
    assert store.layout == "mixtral_split"
    assert gate_up.shape == (1, 4, 3)
    assert gate_up[0, :2].eq(1.0).all()
    assert gate_up[0, 2:].eq(3.0).all()
    assert torch.equal(down, tensors[f"{prefix}.w2.weight"].unsqueeze(0))


def test_checkpoint_store_reads_already_packed_expert_weights(tmp_path) -> None:
    prefix = "model.layers.0.mlp.experts"
    tensors = {
        f"{prefix}.gate_up_proj": torch.randn(3, 4, 5),
        f"{prefix}.down_proj": torch.randn(3, 5, 2),
    }
    index = {"weight_map": {key: "model.safetensors" for key in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(__import__("json").dumps(index))
    store = CheckpointMasterStore(tmp_path)
    store._tensor = tensors.__getitem__  # type: ignore[method-assign]

    actual = store.read(0, "gate_up_proj", [0, 2], torch.device("cpu"), torch.float32)
    assert store.layout == "packed"
    assert torch.equal(actual, tensors[f"{prefix}.gate_up_proj"][[0, 2]])


def test_transformers_mixtral_experts_are_discovered_and_quantized() -> None:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralConfig,
        MixtralForCausalLM,
    )

    config = MixtralConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(9)
    model = MixtralForCausalLM(config).eval()
    modules = find_expert_modules(model)
    assert [entry.layer for entry in modules] == [0, 1]
    assert [entry.num_experts for entry in modules] == [4, 4]

    original = modules[1].module.gate_up_proj.detach().clone()
    quantizer = ExpertQuantizer(model, TINY_QUANT)
    assert quantizer.set_uniform_bits(2, layers=[1]) == 4
    assert not torch.equal(original, modules[1].module.gate_up_proj)
    quantizer.restore()
    assert torch.equal(original, modules[1].module.gate_up_proj)
