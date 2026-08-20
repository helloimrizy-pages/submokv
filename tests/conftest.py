"""Shared fixtures for the Sub-MoKV test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submokv.ground_set import GroundSet  # noqa: E402
from submokv.memory import KVSpec, ModelSpec, QuantSpec  # noqa: E402


@pytest.fixture
def olmoe_model() -> ModelSpec:
    """Return the OLMoE-1B-7B architecture spec."""
    return ModelSpec(
        name="allenai/OLMoE-1B-7B-0924",
        num_hidden_layers=16,
        hidden_size=2048,
        intermediate_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=64,
        vocab_size=50304,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        attention_bias=False,
        has_qk_norm=True,
        dtype_bytes=2,
    )


@pytest.fixture
def kv_spec() -> KVSpec:
    """Return the declared KV cache shape used across the tests."""
    return KVSpec(context_length=4096, batch_size=1, dtype_bytes=2, sink_tokens=4)


@pytest.fixture
def quant_spec() -> QuantSpec:
    """Return the default grouped quantizer storage spec."""
    return QuantSpec(group_size=128, scale_bits=16, zero_point_bits=0)


@pytest.fixture
def ground_set(olmoe_model: ModelSpec, kv_spec: KVSpec, quant_spec: QuantSpec) -> GroundSet:
    """Return the default ground set for OLMoE-1B-7B."""
    return GroundSet(model=olmoe_model, kv=kv_spec, quant=quant_spec)
