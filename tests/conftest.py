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
    return KVSpec(context_length=4096, batch_size=4, dtype_bytes=2, sink_tokens=4)


@pytest.fixture
def quant_spec() -> QuantSpec:
    """Return the default grouped quantizer storage spec."""
    return QuantSpec(group_size=128, scale_bits=16, zero_point_bits=0)


@pytest.fixture
def ground_set(olmoe_model: ModelSpec, kv_spec: KVSpec, quant_spec: QuantSpec) -> GroundSet:
    """Return the default ground set for OLMoE-1B-7B."""
    return GroundSet(model=olmoe_model, kv=kv_spec, quant=quant_spec)


TINY_ARCHITECTURE = dict(
    hidden_size=32,
    intermediate_size=16,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=4,
    num_experts_per_tok=2,
    vocab_size=64,
    max_position_embeddings=64,
    eos_token_id=1,
    pad_token_id=0,
)


def build_tiny_model(attn_implementation: str = "sdpa", seed: int = 0):
    """Return a small randomly initialized OLMoE, which exercises the hooks without a download."""
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(attn_implementation=attn_implementation, **TINY_ARCHITECTURE)
    torch.manual_seed(seed)
    return OlmoeForCausalLM(config).eval()


@pytest.fixture
def tiny_model():
    """Return a small OLMoE using the default attention implementation."""
    return build_tiny_model("sdpa")


@pytest.fixture
def tiny_model_eager():
    """Return a small OLMoE using eager attention, which returns attention weights."""
    return build_tiny_model("eager")


@pytest.fixture
def tiny_ids():
    """Return one fixed token sequence of length 16."""
    import torch

    generator = torch.Generator().manual_seed(1234)
    return torch.randint(0, TINY_ARCHITECTURE["vocab_size"], (1, 16), generator=generator)


@pytest.fixture
def tiny_kv() -> KVSpec:
    """Return a KV spec whose context length matches the tiny sequence length."""
    return KVSpec(context_length=16, batch_size=1, dtype_bytes=2, sink_tokens=2)
