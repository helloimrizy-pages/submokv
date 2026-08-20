"""Analytic byte accounting for model weights and the KV cache.

Fake quantization keeps FP16 tensors resident, so process memory does not move
when a tier changes. Every footprint number in this project comes from the
formulas in this module and never from a device memory query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Union

TierValue = Union[int, float]

BITS_PER_BYTE = 8

# Absorbs floating point error in ratio * length before the ceiling is taken.
_ROUNDING_TOLERANCE = 1e-9


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return the ceiling of an integer division."""
    return -(-numerator // denominator)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture fields needed for analytic byte accounting.

    Field names mirror a Hugging Face model config. Nothing here is read from a
    loaded model, so accounting runs without downloading weights.
    """

    name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    has_qk_norm: bool = True
    head_dim: int | None = None
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if self.head_dim is None:
            derived = self.hidden_size // self.num_attention_heads
            object.__setattr__(self, "head_dim", derived)
        for field_name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_experts",
            "vocab_size",
            "head_dim",
            "dtype_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"ModelSpec.{field_name} must be a positive integer, got {value!r}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads, "
                f"got {self.num_attention_heads} and {self.num_key_value_heads}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelSpec":
        """Build a ModelSpec from a config mapping, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})

    @property
    def query_dim(self) -> int:
        """Return the total dimension across all query heads."""
        return self.num_attention_heads * int(self.head_dim)

    @property
    def key_value_dim(self) -> int:
        """Return the total dimension across all key or value heads."""
        return self.num_key_value_heads * int(self.head_dim)


@dataclass(frozen=True)
class KVSpec:
    """Declared cache shape used for KV byte accounting.

    context_length is a declared property of the experiment, not a property of
    the model, and it is recorded in every result.
    """

    context_length: int
    batch_size: int = 1
    dtype_bytes: int = 2
    sink_tokens: int = 4

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes must be positive, got {self.dtype_bytes}")
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must not be negative, got {self.sink_tokens}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "KVSpec":
        """Build a KVSpec from a config mapping, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass(frozen=True)
class QuantSpec:
    """Storage parameters for grouped weight quantization.

    Symmetric quantization stores no zero point, so zero_point_bits defaults to
    zero. The field exists so an asymmetric variant can be costed without a code
    change. A tier at unquantized_bits stores no scales at all, because that
    tier is the unmodified 16-bit weight rather than a quantized one.
    """

    group_size: int = 128
    scale_bits: int = 16
    zero_point_bits: int = 0
    symmetric: bool = True
    unquantized_bits: int = 16

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.scale_bits < 0 or self.zero_point_bits < 0:
            raise ValueError("scale_bits and zero_point_bits must not be negative")
        if self.symmetric and self.zero_point_bits != 0:
            raise ValueError(
                "symmetric quantization stores no zero point; set symmetric=False "
                f"to charge zero_point_bits={self.zero_point_bits}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QuantSpec":
        """Build a QuantSpec from a config mapping, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass(frozen=True)
class Footprint:
    """Byte totals for one allocation, split into the parts that can change."""

    fixed_weight_bytes: int
    expert_weight_bytes: int
    kv_bytes: int

    @property
    def weight_bytes(self) -> int:
        """Return all weight bytes, quantized and unquantized."""
        return self.fixed_weight_bytes + self.expert_weight_bytes

    @property
    def total_bytes(self) -> int:
        """Return the full footprint in bytes."""
        return self.fixed_weight_bytes + self.expert_weight_bytes + self.kv_bytes

    def as_dict(self) -> dict[str, int]:
        """Return the breakdown and total as a plain dictionary for result records."""
        return {
            "fixed_weight_bytes": self.fixed_weight_bytes,
            "expert_weight_bytes": self.expert_weight_bytes,
            "kv_bytes": self.kv_bytes,
            "weight_bytes": self.weight_bytes,
            "total_bytes": self.total_bytes,
        }


def matrix_bytes(
    out_features: int,
    in_features: int,
    bits: int,
    quant: QuantSpec,
    dtype_bytes: int = 2,
) -> int:
    """Return the stored bytes for one weight matrix at the given bit width.

    Groups run along the input dimension of each output row, so a matrix of
    shape (out_features, in_features) holds out_features * ceil(in_features /
    group_size) groups. Each group stores one scale and, when the quantizer is
    asymmetric, one zero point. At unquantized_bits the matrix is stored as a
    plain dtype tensor with no group overhead.
    """
    if out_features <= 0 or in_features <= 0:
        raise ValueError(f"matrix shape must be positive, got ({out_features}, {in_features})")
    elements = out_features * in_features
    if bits >= quant.unquantized_bits:
        return elements * dtype_bytes
    payload_bytes = _ceil_div(elements * bits, BITS_PER_BYTE)
    groups = out_features * _ceil_div(in_features, quant.group_size)
    overhead_bits = groups * (quant.scale_bits + quant.zero_point_bits)
    return payload_bytes + _ceil_div(overhead_bits, BITS_PER_BYTE)


def expert_matrix_shapes(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    """Return (out_features, in_features) for the three matrices of one expert."""
    return (
        (model.intermediate_size, model.hidden_size),
        (model.intermediate_size, model.hidden_size),
        (model.hidden_size, model.intermediate_size),
    )


def expert_params_per_layer(model: ModelSpec) -> int:
    """Return the parameter count of all experts in one layer."""
    per_expert = sum(out * inp for out, inp in expert_matrix_shapes(model))
    return per_expert * model.num_experts


def attention_params_per_layer(model: ModelSpec) -> int:
    """Return the parameter count of one layer's attention projections."""
    params = (
        model.hidden_size * model.query_dim
        + 2 * model.hidden_size * model.key_value_dim
        + model.query_dim * model.hidden_size
    )
    if model.attention_bias:
        params += model.query_dim + 2 * model.key_value_dim + model.hidden_size
    return params


def router_params_per_layer(model: ModelSpec) -> int:
    """Return the parameter count of one layer's expert router."""
    return model.hidden_size * model.num_experts


def norm_params_per_layer(model: ModelSpec) -> int:
    """Return the parameter count of one layer's normalization weights."""
    params = 2 * model.hidden_size
    if model.has_qk_norm:
        params += model.query_dim + model.key_value_dim
    return params


def embedding_params(model: ModelSpec) -> int:
    """Return the parameter count of the token embedding, output head, and final norm."""
    params = model.vocab_size * model.hidden_size + model.hidden_size
    if not model.tie_word_embeddings:
        params += model.vocab_size * model.hidden_size
    return params


def fixed_params(model: ModelSpec) -> int:
    """Return the parameter count that stays at the model dtype in every allocation."""
    per_layer = (
        attention_params_per_layer(model)
        + router_params_per_layer(model)
        + norm_params_per_layer(model)
    )
    return model.num_hidden_layers * per_layer + embedding_params(model)


def total_params(model: ModelSpec) -> int:
    """Return the total parameter count of the model."""
    return fixed_params(model) + model.num_hidden_layers * expert_params_per_layer(model)


def fixed_weight_bytes(model: ModelSpec) -> int:
    """Return the bytes held by parameters that are never quantized."""
    return fixed_params(model) * model.dtype_bytes


def expert_bytes(model: ModelSpec, quant: QuantSpec, bits: int, num_experts: int) -> int:
    """Return the stored bytes for a number of experts held at one bit width."""
    if num_experts < 0:
        raise ValueError(f"num_experts must not be negative, got {num_experts}")
    per_expert = sum(
        matrix_bytes(out, inp, bits, quant, model.dtype_bytes)
        for out, inp in expert_matrix_shapes(model)
    )
    return per_expert * num_experts


def retained_tokens(kv: KVSpec, retention: float, length: int | None = None) -> int:
    """Return the number of cached token positions kept at a retention ratio.

    The always-kept sink tokens count inside the retained budget rather than on
    top of it, and the result never exceeds the length it is measured against.
    Byte accounting passes no length and so uses the declared context length.
    The retention hooks pass the length of the sequence being processed, so a
    ratio means the same share of the cache in both places.
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError(f"retention must lie in [0, 1], got {retention}")
    if length is None:
        length = kv.context_length
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    tokens = math.ceil(retention * length - _ROUNDING_TOLERANCE)
    tokens = max(tokens, min(kv.sink_tokens, length))
    return min(tokens, length)


def kv_bytes_per_layer(model: ModelSpec, kv: KVSpec, retention: float) -> int:
    """Return the cache bytes held by one layer at a retention ratio."""
    per_token = 2 * model.key_value_dim * kv.dtype_bytes
    return kv.batch_size * retained_tokens(kv, retention) * per_token


def reference_footprint(model: ModelSpec, kv: KVSpec, quant: QuantSpec) -> Footprint:
    """Return the footprint with every expert at the model dtype and every cache kept in full.

    This is the denominator for budget fractions.
    """
    experts = model.num_hidden_layers * expert_bytes(
        model, quant, quant.unquantized_bits, model.num_experts
    )
    cache = model.num_hidden_layers * kv_bytes_per_layer(model, kv, 1.0)
    return Footprint(
        fixed_weight_bytes=fixed_weight_bytes(model),
        expert_weight_bytes=experts,
        kv_bytes=cache,
    )


def format_bytes(value: int) -> str:
    """Return a byte count rendered in GiB with three decimal places."""
    return f"{value / (1024 ** 3):.3f} GiB"
