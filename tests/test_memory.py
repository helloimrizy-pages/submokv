"""Tests for analytic byte accounting.

The reference numbers in this module are recomputed from the OLMoE-1B-7B
architecture by hand rather than read back from the code under test.
"""

from __future__ import annotations

import pytest

from submokv.memory import (
    KVSpec,
    ModelSpec,
    QuantSpec,
    attention_params_per_layer,
    embedding_params,
    expert_bytes,
    expert_params_per_layer,
    fixed_weight_bytes,
    kv_bytes_per_layer,
    matrix_bytes,
    reference_footprint,
    retained_tokens,
    total_params,
)

# Hand computation from the published architecture of OLMoE-1B-7B-0924.
LAYERS = 16
HIDDEN = 2048
INTERMEDIATE = 1024
HEADS = 16
KV_HEADS = 16
HEAD_DIM = 128
EXPERTS = 64
VOCAB = 50304
CONTEXT = 4096
DTYPE_BYTES = 2

# Each expert holds gate_proj and up_proj of shape (1024, 2048) and down_proj of
# shape (2048, 1024).
HAND_EXPERT_PARAMS_PER_LAYER = EXPERTS * 3 * INTERMEDIATE * HIDDEN
# Four square projections, no bias.
HAND_ATTENTION_PARAMS_PER_LAYER = 4 * HIDDEN * HIDDEN
HAND_ROUTER_PARAMS_PER_LAYER = HIDDEN * EXPERTS
# Two decoder norms plus the query and key norms.
HAND_NORM_PARAMS_PER_LAYER = 2 * HIDDEN + HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM
# Untied embedding and output head, plus the final norm.
HAND_EMBEDDING_PARAMS = 2 * VOCAB * HIDDEN + HIDDEN

HAND_TOTAL_PARAMS = (
    LAYERS
    * (
        HAND_EXPERT_PARAMS_PER_LAYER
        + HAND_ATTENTION_PARAMS_PER_LAYER
        + HAND_ROUTER_PARAMS_PER_LAYER
        + HAND_NORM_PARAMS_PER_LAYER
    )
    + HAND_EMBEDDING_PARAMS
)
HAND_WEIGHT_BYTES_16BIT = HAND_TOTAL_PARAMS * DTYPE_BYTES
HAND_KV_BYTES_FULL = LAYERS * 2 * CONTEXT * KV_HEADS * HEAD_DIM * DTYPE_BYTES
HAND_REFERENCE_BYTES = HAND_WEIGHT_BYTES_16BIT + HAND_KV_BYTES_FULL

TOLERANCE = 1e-3


def _relative_error(measured: int, expected: int) -> float:
    return abs(measured - expected) / expected


def test_total_params_matches_hand_computation(olmoe_model: ModelSpec) -> None:
    assert total_params(olmoe_model) == HAND_TOTAL_PARAMS


def test_component_params_match_hand_computation(olmoe_model: ModelSpec) -> None:
    assert expert_params_per_layer(olmoe_model) == HAND_EXPERT_PARAMS_PER_LAYER
    assert attention_params_per_layer(olmoe_model) == HAND_ATTENTION_PARAMS_PER_LAYER
    assert embedding_params(olmoe_model) == HAND_EMBEDDING_PARAMS


def test_reference_footprint_matches_hand_computed_16bit_baseline(
    olmoe_model: ModelSpec, kv_spec: KVSpec, quant_spec: QuantSpec
) -> None:
    """The 16-bit reference footprint must match the hand computation within 0.1 percent."""
    footprint = reference_footprint(olmoe_model, kv_spec, quant_spec)
    assert _relative_error(footprint.weight_bytes, HAND_WEIGHT_BYTES_16BIT) < TOLERANCE
    assert _relative_error(footprint.kv_bytes, HAND_KV_BYTES_FULL) < TOLERANCE
    assert _relative_error(footprint.total_bytes, HAND_REFERENCE_BYTES) < TOLERANCE


def test_reference_footprint_carries_no_quantization_overhead(
    olmoe_model: ModelSpec, kv_spec: KVSpec, quant_spec: QuantSpec
) -> None:
    """At the unquantized tier no scales are stored, so weights are exactly two bytes each."""
    footprint = reference_footprint(olmoe_model, kv_spec, quant_spec)
    assert footprint.weight_bytes == HAND_TOTAL_PARAMS * DTYPE_BYTES


def test_fixed_weight_bytes_excludes_experts(olmoe_model: ModelSpec) -> None:
    expected = (HAND_TOTAL_PARAMS - LAYERS * HAND_EXPERT_PARAMS_PER_LAYER) * DTYPE_BYTES
    assert fixed_weight_bytes(olmoe_model) == expected


def test_experts_dominate_the_parameter_count(olmoe_model: ModelSpec) -> None:
    """Only the experts are quantizable, so their share bounds what quantization can save."""
    share = LAYERS * expert_params_per_layer(olmoe_model) / total_params(olmoe_model)
    assert 0.92 < share < 0.94


def test_matrix_bytes_unquantized_has_no_group_overhead(quant_spec: QuantSpec) -> None:
    assert matrix_bytes(1024, 2048, 16, quant_spec) == 1024 * 2048 * 2


def test_matrix_bytes_group_overhead_is_one_scale_per_group(quant_spec: QuantSpec) -> None:
    out_features, in_features, bits = 1024, 2048, 4
    groups = out_features * (in_features // quant_spec.group_size)
    expected = out_features * in_features * bits // 8 + groups * quant_spec.scale_bits // 8
    assert matrix_bytes(out_features, in_features, bits, quant_spec) == expected


@pytest.mark.parametrize(
    "bits,expected_effective_bits",
    [(2, 2.125), (3, 3.125), (4, 4.125), (8, 8.125), (16, 16.0)],
)
def test_matrix_bytes_effective_bits_per_element(
    quant_spec: QuantSpec, bits: int, expected_effective_bits: float
) -> None:
    """A 16-bit scale over a group of 128 adds exactly 0.125 bits per element."""
    elements = 1024 * 2048
    measured = matrix_bytes(1024, 2048, bits, quant_spec) * 8 / elements
    assert measured == pytest.approx(expected_effective_bits)


def test_matrix_bytes_partial_group_is_charged_in_full() -> None:
    """A row shorter than the group size still stores one scale."""
    quant = QuantSpec(group_size=128, scale_bits=16, zero_point_bits=0)
    assert matrix_bytes(4, 100, 4, quant) == 4 * 100 * 4 // 8 + 4 * 2


def test_matrix_bytes_rises_with_bit_width(quant_spec: QuantSpec) -> None:
    sizes = [matrix_bytes(1024, 2048, bits, quant_spec) for bits in (2, 3, 4, 8, 16)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_expert_bytes_scales_with_expert_count(
    olmoe_model: ModelSpec, quant_spec: QuantSpec
) -> None:
    one = expert_bytes(olmoe_model, quant_spec, 4, 1)
    assert expert_bytes(olmoe_model, quant_spec, 4, 64) == 64 * one
    assert expert_bytes(olmoe_model, quant_spec, 4, 0) == 0


@pytest.mark.parametrize(
    "retention,expected",
    [(0.25, 1024), (0.50, 2048), (0.75, 3072), (1.00, 4096), (0.10, 410)],
)
def test_retained_tokens(kv_spec: KVSpec, retention: float, expected: int) -> None:
    assert retained_tokens(kv_spec, retention) == expected


def test_retained_tokens_never_falls_below_the_sink() -> None:
    kv = KVSpec(context_length=4096, sink_tokens=64)
    assert retained_tokens(kv, 0.0) == 64
    assert retained_tokens(kv, 1.0) == 4096


def test_retained_tokens_rejects_ratio_outside_unit_interval(kv_spec: KVSpec) -> None:
    with pytest.raises(ValueError):
        retained_tokens(kv_spec, 1.5)


def test_kv_bytes_per_layer_matches_hand_computation(
    olmoe_model: ModelSpec, kv_spec: KVSpec
) -> None:
    expected = 2 * CONTEXT * KV_HEADS * HEAD_DIM * DTYPE_BYTES
    assert kv_bytes_per_layer(olmoe_model, kv_spec, 1.0) == expected
    assert kv_bytes_per_layer(olmoe_model, kv_spec, 0.25) == expected // 4


def test_kv_bytes_scale_with_batch_and_context(olmoe_model: ModelSpec) -> None:
    single = kv_bytes_per_layer(olmoe_model, KVSpec(context_length=4096, batch_size=1), 1.0)
    batched = kv_bytes_per_layer(olmoe_model, KVSpec(context_length=4096, batch_size=4), 1.0)
    longer = kv_bytes_per_layer(olmoe_model, KVSpec(context_length=8192, batch_size=1), 1.0)
    assert batched == 4 * single
    assert longer == 2 * single


def test_model_spec_derives_head_dim(olmoe_model: ModelSpec) -> None:
    assert olmoe_model.head_dim == HEAD_DIM
    assert olmoe_model.query_dim == HIDDEN
    assert olmoe_model.key_value_dim == HIDDEN


def test_model_spec_rejects_nonpositive_fields() -> None:
    with pytest.raises(ValueError):
        ModelSpec(
            name="broken",
            num_hidden_layers=0,
            hidden_size=2048,
            intermediate_size=1024,
            num_attention_heads=16,
            num_key_value_heads=16,
            num_experts=64,
            vocab_size=50304,
            max_position_embeddings=4096,
        )


def test_quant_spec_rejects_a_zero_point_it_would_not_store() -> None:
    with pytest.raises(ValueError):
        QuantSpec(group_size=128, zero_point_bits=8, symmetric=True)


def test_quant_spec_charges_an_asymmetric_zero_point() -> None:
    asymmetric = QuantSpec(group_size=128, scale_bits=16, zero_point_bits=16, symmetric=False)
    symmetric = QuantSpec(group_size=128, scale_bits=16, zero_point_bits=0, symmetric=True)
    assert matrix_bytes(1024, 2048, 4, asymmetric) > matrix_bytes(1024, 2048, 4, symmetric)
