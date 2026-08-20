"""Per-layer KV cache retention with a configurable eviction policy.

Retention is simulated the same way quantization is: the cache holds every
entry, and an attention mask hides the entries a policy has evicted. Nothing is
physically dropped, so cache indices stay equal to absolute token positions,
rotary embeddings are untouched, and the retained set can be any subset rather
than a contiguous window. The bytes a retention ratio implies are computed
analytically in memory.py.

Two policies are implemented and the one in use is recorded in every result,
because the two do not agree:

    recency_sink     keeps the first sink_tokens positions and the most recent
                     positions up to the budget. The retained set does not
                     depend on the data, so one forward pass over the whole
                     sequence simulates it exactly.
    attention_score  keeps the positions with the highest accumulated attention
                     score, in the style of H2O and SnapKV, alongside the sinks
                     and a recent window. The retained set depends on attention
                     weights, so the sequence is processed in chunks and the
                     policy evicts between them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from .memory import KVSpec, retained_tokens

_LAYER_INDEX_PATTERN = re.compile(r"layers\.(\d+)\.")
ATTENTION_PARAMETER_NAMES = ("q_proj", "k_proj", "v_proj")


@dataclass(frozen=True)
class AttentionModule:
    """One layer's attention module, with the layer index it belongs to."""

    layer: int
    name: str
    module: nn.Module


def find_attention_modules(model: nn.Module) -> tuple[AttentionModule, ...]:
    """Return the attention modules of a model, ordered by layer index."""
    found: list[AttentionModule] = []
    for name, module in model.named_modules():
        children = dict(module.named_children())
        if not all(key in children for key in ATTENTION_PARAMETER_NAMES):
            continue
        match = _LAYER_INDEX_PATTERN.search(name)
        if match is None:
            raise ValueError(f"cannot read a layer index from module name {name!r}")
        found.append(AttentionModule(layer=int(match.group(1)), name=name, module=module))
    found.sort(key=lambda entry: entry.layer)
    if not found:
        raise ValueError(
            "no attention modules found; expected modules holding "
            f"{ATTENTION_PARAMETER_NAMES} as children"
        )
    return tuple(found)


class RetentionPolicy(ABC):
    """Decides which cached positions a query may attend to."""

    name: str
    needs_attention_scores: bool

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return the policy name and its parameters for result records."""

    @abstractmethod
    def allowed(
        self,
        query_positions: torch.Tensor,
        key_length: int,
        budget: int,
        state: "PolicyState | None",
    ) -> torch.Tensor:
        """Return a boolean mask over (heads, queries, keys) or (queries, keys).

        The result must broadcast against (batch, query heads, queries, keys).
        An entry is True when the query may attend to that key. The policy must
        not admit a key that lies after its query.
        """


@dataclass
class PolicyState:
    """Per-sequence state for a policy whose retained set depends on the data.

    scores and retained both have shape (batch, key value heads, sequence
    length), so every sequence in a batch keeps its own set.
    """

    batch_size: int
    num_key_value_heads: int
    num_key_value_groups: int
    sequence_length: int
    device: torch.device
    scores: torch.Tensor | None = None
    retained: torch.Tensor | None = None
    positions_seen: int = 0


@dataclass(frozen=True)
class RecencySinkPolicy(RetentionPolicy):
    """Keeps the first sink_tokens positions and the most recent positions.

    The retained set is a function of position alone, so the mask for the whole
    sequence can be built once and the sequence needs only one forward pass.
    """

    sink_tokens: int = 4
    name: str = "recency_sink"
    needs_attention_scores: bool = False

    def describe(self) -> dict[str, Any]:
        """Return the policy name and its parameters for result records."""
        return {"policy": self.name, "sink_tokens": self.sink_tokens}

    def allowed(
        self,
        query_positions: torch.Tensor,
        key_length: int,
        budget: int,
        state: PolicyState | None = None,
    ) -> torch.Tensor:
        """Return a boolean mask of shape (queries, keys)."""
        keys = torch.arange(key_length, device=query_positions.device)
        queries = query_positions.reshape(-1, 1)
        sink = min(self.sink_tokens, budget)
        recent = budget - sink
        causal = keys.reshape(1, -1) <= queries
        is_sink = keys.reshape(1, -1) < sink
        is_recent = keys.reshape(1, -1) > queries - recent
        return causal & (is_sink | is_recent)


@dataclass(frozen=True)
class AttentionScorePolicy(RetentionPolicy):
    """Keeps the positions with the highest accumulated attention score.

    This follows H2O and SnapKV: attention probabilities are summed over the
    queries seen so far, and the positions with the largest totals are kept
    alongside the sinks and a recent window. Scores are tracked per key value
    head, which is the granularity the byte accounting assumes.
    """

    sink_tokens: int = 4
    recent_window: int = 32
    name: str = "attention_score"
    needs_attention_scores: bool = True

    def describe(self) -> dict[str, Any]:
        """Return the policy name and its parameters for result records."""
        return {
            "policy": self.name,
            "sink_tokens": self.sink_tokens,
            "recent_window": self.recent_window,
        }

    def allowed(
        self,
        query_positions: torch.Tensor,
        key_length: int,
        budget: int,
        state: PolicyState | None = None,
    ) -> torch.Tensor:
        """Return a boolean mask of shape (batch, query heads, queries, keys)."""
        if state is None:
            raise ValueError("attention_score needs per-sequence state")
        device = query_positions.device
        keys = torch.arange(key_length, device=device)
        causal = keys.reshape(1, -1) <= query_positions.reshape(-1, 1)

        seen = state.positions_seen
        retained = torch.ones(
            state.batch_size, state.num_key_value_heads, key_length, dtype=torch.bool, device=device
        )
        if state.retained is not None and seen > 0:
            retained[:, :, :seen] = state.retained[:, :, :seen]
        per_head = retained.unsqueeze(2) & causal.reshape(1, 1, *causal.shape)
        return per_head.repeat_interleave(state.num_key_value_groups, dim=1)

    def evict(self, state: PolicyState, budget: int) -> None:
        """Reduce the retained set of every head of every sequence to the budget, in place."""
        if state.scores is None or state.positions_seen <= budget:
            return
        seen = state.positions_seen
        device = state.device
        scores = state.scores[:, :, :seen].clone()

        keep = torch.zeros(
            state.batch_size, state.num_key_value_heads, seen, dtype=torch.bool, device=device
        )
        sink = min(self.sink_tokens, budget)
        keep[:, :, :sink] = True
        recent = min(self.recent_window, budget - sink)
        if recent > 0:
            keep[:, :, seen - recent :] = True

        # The sinks and the recent window are position based, so every head of
        # every sequence has kept the same number of positions at this point.
        remaining = budget - int(keep[0, 0].sum().item())
        if remaining > 0:
            scores = scores.masked_fill(keep, float("-inf"))
            chosen = torch.topk(scores, remaining, dim=-1).indices
            keep.scatter_(2, chosen, True)
        if state.retained is None:
            state.retained = torch.zeros(
                state.batch_size,
                state.num_key_value_heads,
                state.sequence_length,
                dtype=torch.bool,
                device=device,
            )
        state.retained[:, :, :seen] = keep

    def accumulate(self, state: PolicyState, attention_weights: torch.Tensor) -> None:
        """Add the attention mass a chunk placed on each key position to the running scores."""
        batch, _, _, key_length = attention_weights.shape
        grouped = attention_weights.reshape(
            batch, state.num_key_value_heads, state.num_key_value_groups, -1, key_length
        )
        chunk_scores = grouped.float().sum(dim=(2, 3))
        if state.scores is None:
            state.scores = torch.zeros(
                state.batch_size,
                state.num_key_value_heads,
                state.sequence_length,
                dtype=torch.float32,
                device=state.device,
            )
        state.scores[:, :, :key_length] += chunk_scores


POLICIES: dict[str, type[RetentionPolicy]] = {
    RecencySinkPolicy.name: RecencySinkPolicy,
    AttentionScorePolicy.name: AttentionScorePolicy,
}


def build_policy(config: Mapping[str, Any]) -> RetentionPolicy:
    """Build the retention policy named in a config mapping.

    The policy name is pinned in the config because the two policies do not
    produce the same results.
    """
    settings = dict(config)
    name = settings.pop("policy", None)
    if name not in POLICIES:
        raise ValueError(f"unknown retention policy {name!r}; choose one of {sorted(POLICIES)}")
    return POLICIES[name](**settings)


@dataclass(frozen=True)
class EvaluationProtocol:
    """What the retained set is measured relative to while a sequence is scored.

    Masking says which positions a query may attend to. It does not by itself
    say what the retained set is anchored to, and the readings disagree on
    perplexity, so one is pinned here and recorded in every result.

        global    One retained set fixed for the whole sequence. A query at
                  position 2000 is denied position 1500 even though position
                  1500 was recent and still cached when it was decoded. This
                  measures something stricter than eviction.
        sliding   Query t sees the sinks and the window before t, for every t
                  including the first. This is StreamingLLM style sliding
                  window attention. Honest, but a different deployment claim
                  than prefilling and then evicting.
        prefill_evict_decode
                  Prefill a prefix with full attention, apply the policy to
                  that prefix, then score teacher forced loss on the tail only,
                  with each query attending to the retained prefix and its own
                  chunk causally. The visible cache is held at the budget
                  throughout the tail, so the peak matches what memory.py
                  charges for the retention ratio. H2O and SnapKV report this
                  way.

    prefill_evict_decode is the one implemented, because it is the one the byte
    accounting describes. Loss is never scored on prefill positions, where
    nothing has been evicted yet, and that is the confound the other two
    readings introduce.
    """

    name: str = "prefill_evict_decode"
    prefill_tokens: int = 3072
    chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.name != "prefill_evict_decode":
            raise ValueError(
                f"only the prefill_evict_decode protocol is implemented, got {self.name!r}"
            )
        if self.prefill_tokens < 0:
            raise ValueError(f"prefill_tokens must not be negative, got {self.prefill_tokens}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    def describe(self) -> dict[str, Any]:
        """Return the protocol name and its parameters for result records."""
        return {
            "protocol": self.name,
            "prefill_tokens": self.prefill_tokens,
            "protocol_chunk_size": self.chunk_size,
        }

    def scored_tokens(self, sequence_length: int) -> int:
        """Return how many positions of a sequence carry loss."""
        if self.prefill_tokens >= sequence_length:
            raise ValueError(
                f"prefill_tokens {self.prefill_tokens} leaves no tail to score in a "
                f"sequence of length {sequence_length}"
            )
        return sequence_length - self.prefill_tokens


class RetentionController:
    """Applies a per-layer retention ratio to a model by masking the attention.

    Retention ratios are set at runtime with set_retention and take effect on the
    next forward pass, so no layer needs the model reloaded to change tier.
    """

    def __init__(
        self,
        model: nn.Module,
        policy: RetentionPolicy,
        kv: KVSpec,
        protocol: EvaluationProtocol | None = None,
        retention: Mapping[int, float] | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.kv = kv
        self.protocol = protocol if protocol is not None else EvaluationProtocol()
        self.attention_modules = find_attention_modules(model)
        config = model.config
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(getattr(config, "num_key_value_heads", self.num_attention_heads))
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        policy_sinks = getattr(policy, "sink_tokens", None)
        if policy_sinks is not None and policy_sinks != kv.sink_tokens:
            raise ValueError(
                f"the {policy.name} policy pins {policy_sinks} sink tokens but the KV spec "
                f"declares {kv.sink_tokens}; the accounting and the hooks would disagree"
            )
        self._retention: dict[int, float] = {entry.layer: 1.0 for entry in self.attention_modules}
        if retention is not None:
            self.set_retention(retention)
        self._handles: list[Any] = []
        self._state: dict[int, PolicyState] = {}
        self._sequence_length: int | None = None
        self._attention_weights: dict[int, torch.Tensor] = {}
        self._masking_enabled = True
        self._exempt_below: int = 0
        self._mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    @property
    def layers(self) -> tuple[int, ...]:
        """Return the layer indices that carry a retention ratio."""
        return tuple(entry.layer for entry in self.attention_modules)

    def retention(self) -> dict[int, float]:
        """Return the retention ratio currently applied to each layer."""
        return dict(self._retention)

    def set_retention(self, retention: Mapping[int, float]) -> None:
        """Set the retention ratio of one or more layers, taking effect on the next forward pass."""
        unknown = set(retention) - set(self._retention)
        if unknown:
            raise KeyError(f"retention names layers {sorted(unknown)} that hold no attention module")
        for layer, ratio in retention.items():
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"retention ratio for layer {layer} must lie in (0, 1], got {ratio}")
            self._retention[layer] = float(ratio)

    def set_uniform_retention(self, ratio: float, layers: Iterable[int] | None = None) -> None:
        """Set one retention ratio across every layer, or across the named layers."""
        selected = self.layers if layers is None else tuple(layers)
        self.set_retention({layer: ratio for layer in selected})

    def budget_tokens(self, layer: int, sequence_length: int) -> int:
        """Return how many positions layer may attend to at its current ratio."""
        return retained_tokens(self.kv, self._retention[layer], length=sequence_length)

    def is_inactive(self) -> bool:
        """Return whether every layer keeps its whole cache, so no mask is needed."""
        return all(ratio >= 1.0 for ratio in self._retention.values())

    def attach(self) -> "RetentionController":
        """Register the mask hooks on every attention module."""
        if self._handles:
            return self
        for entry in self.attention_modules:
            self._handles.append(
                entry.module.register_forward_pre_hook(
                    self._make_pre_hook(entry.layer), with_kwargs=True
                )
            )
            if self.policy.needs_attention_scores:
                self._handles.append(
                    entry.module.register_forward_hook(
                        self._make_post_hook(entry.layer), with_kwargs=True
                    )
                )
        return self

    def detach(self) -> None:
        """Remove the mask hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def begin_sequence(self, batch_size: int, sequence_length: int, device: torch.device) -> None:
        """Reset per-sequence policy state before processing a new batch of sequences."""
        self._sequence_length = sequence_length
        self._attention_weights = {}
        self._mask_cache = {}
        self._state = {
            layer: PolicyState(
                batch_size=batch_size,
                num_key_value_heads=self.num_key_value_heads,
                num_key_value_groups=self.num_key_value_groups,
                sequence_length=sequence_length,
                device=device,
            )
            for layer in self.layers
        }

    def set_prefill_exemption(self, positions: int) -> None:
        """Let queries below a position attend to everything causally.

        The protocol prefills a prefix with full attention. When the retained
        set depends only on position, the prefix and the tail can share one
        forward pass, and the exemption is what keeps the prefix unmasked
        inside it.
        """
        self._exempt_below = int(positions)
        self._mask_cache = {}

    def set_masking(self, enabled: bool) -> None:
        """Switch the retention mask on or off, leaving score accumulation running.

        The prefill phase runs with full attention but still needs the attention
        weights it produces, which is what the score based policy selects from.
        """
        self._masking_enabled = enabled

    def end_chunk(self, reserve: int = 0, evict: bool = True) -> None:
        """Take in a chunk's attention weights and let the policy evict.

        reserve leaves room for the positions the next chunk will add, so the
        visible cache never rises above the budget at the moment attention is
        computed.
        """
        if not isinstance(self.policy, AttentionScorePolicy):
            self._attention_weights = {}
            return
        if self._sequence_length is None:
            raise RuntimeError("call begin_sequence before processing a chunk")
        for layer, weights in self._attention_weights.items():
            state = self._state[layer]
            self.policy.accumulate(state, weights)
            state.positions_seen = weights.shape[-1]
            if evict:
                budget = self.budget_tokens(layer, self._sequence_length) - reserve
                self.policy.evict(state, max(budget, self.kv.sink_tokens))
        self._attention_weights = {}

    def describe(self) -> dict[str, Any]:
        """Return the policy and retention settings for result records."""
        return {
            **self.policy.describe(),
            **self.protocol.describe(),
            "context_length": self.kv.context_length,
            "sink_tokens_declared": self.kv.sink_tokens,
            "retention_by_layer": {str(k): v for k, v in sorted(self._retention.items())},
        }

    def _make_pre_hook(self, layer: int) -> Callable[..., Any]:
        def pre_hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            ratio = self._retention[layer]
            if ratio >= 1.0 or not self._masking_enabled:
                return None
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_ids = kwargs.get("position_ids")
            if hidden_states is None or position_ids is None:
                raise RuntimeError(
                    "the retention hook needs hidden_states and position_ids in the "
                    "attention call, which this model version does not provide"
                )
            if position_ids.shape[0] > 1 and not bool(
                (position_ids == position_ids[0]).all()
            ):
                raise NotImplementedError(
                    "the retention hooks assume every sequence in a batch sits at the same "
                    "positions; padded or ragged batches are not supported"
                )
            query_positions = position_ids[0].to(torch.long)
            cache = kwargs.get("past_key_values")
            cached = 0
            if cache is not None and len(cache.layers) > layer:
                cached = cache.layers[layer].get_seq_length()
            key_length = cached + hidden_states.shape[1]
            sequence_length = self._sequence_length or key_length
            budget = retained_tokens(self.kv, ratio, length=sequence_length)

            # A policy whose retained set depends only on position gives every
            # layer at the same ratio the same mask, so it is built once.
            cacheable = not self.policy.needs_attention_scores
            cache_key = (
                ratio,
                key_length,
                int(query_positions[0]),
                int(query_positions.numel()),
                self._exempt_below,
            )
            allowed = self._mask_cache.get(cache_key) if cacheable else None
            if allowed is None:
                allowed = self.policy.allowed(
                    query_positions, key_length, budget, self._state.get(layer)
                )
                if self._exempt_below > 0:
                    keys = torch.arange(key_length, device=query_positions.device)
                    causal = keys.reshape(1, -1) <= query_positions.reshape(-1, 1)
                    exempt = query_positions.reshape(-1, 1) < self._exempt_below
                    allowed = allowed | (causal & exempt)
                while allowed.ndim < 4:
                    allowed = allowed.unsqueeze(0)
                if cacheable:
                    self._mask_cache[cache_key] = allowed

            # transformers hands the attention module a boolean mask on some
            # paths and an additive float mask on others, and no mask at all
            # when it can use the built in causal path. The merged mask keeps
            # whichever form arrived, so nothing downstream has to change.
            existing = kwargs.get("attention_mask")
            if torch.is_tensor(existing) and existing.dtype == torch.bool:
                merged = existing[..., :key_length] & allowed
            elif torch.is_tensor(existing):
                merged = torch.where(
                    allowed, existing[..., :key_length], torch.finfo(existing.dtype).min
                )
            else:
                dtype = hidden_states.dtype
                merged = torch.where(
                    allowed,
                    torch.zeros((), dtype=dtype, device=hidden_states.device),
                    torch.full((), torch.finfo(dtype).min, dtype=dtype, device=hidden_states.device),
                )
            kwargs["attention_mask"] = merged
            return args, kwargs

        return pre_hook

    def _make_post_hook(self, layer: int) -> Callable[..., Any]:
        def post_hook(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    f"the {self.policy.name} policy needs attention weights, which the "
                    f"{self.model.config._attn_implementation!r} attention implementation does "
                    "not return; load the model with attn_implementation='eager'"
                )
            self._attention_weights[layer] = weights.detach()

        return post_hook


def forward_with_retention(
    model: nn.Module,
    input_ids: torch.Tensor,
    controller: RetentionController,
    chunk_size: int | None = None,
    collect: Callable[[torch.Tensor, int], None] | None = None,
) -> torch.Tensor | None:
    """Run a sequence through the model with the controller's retention applied.

    A policy whose retained set does not depend on the data is simulated exactly
    in one pass, because the mask is a function of position alone. A policy that
    reads attention weights needs the sequence split into chunks, so that it can
    evict between them the way it would during decoding.

    Returns the logits for the whole sequence, or None when collect is given, in
    which case each chunk's logits are passed to collect with the index of the
    first token in that chunk.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")
    sequence_length = input_ids.shape[1]
    controller.begin_sequence(input_ids.shape[0], sequence_length, input_ids.device)

    needs_chunks = controller.policy.needs_attention_scores and not controller.is_inactive()
    if not needs_chunks and chunk_size is None:
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        if collect is not None:
            collect(logits, 0)
            return None
        return logits

    size = chunk_size or sequence_length
    if needs_chunks and chunk_size is None:
        raise ValueError(
            f"the {controller.policy.name} policy evicts between chunks, so chunk_size is required"
        )

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    pieces: list[torch.Tensor] = []
    for start in range(0, sequence_length, size):
        stop = min(start + size, sequence_length)
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, stop, device=input_ids.device),
            )
        controller.end_chunk()
        if collect is not None:
            collect(output.logits, start)
        else:
            pieces.append(output.logits)
    if collect is not None:
        return None
    return torch.cat(pieces, dim=1)


def score_sequence(
    model: nn.Module,
    input_ids: torch.Tensor,
    controller: RetentionController,
    route: str = "auto",
) -> torch.Tensor:
    """Return the per-token negative log likelihood of the scored tail of each sequence.

    The prefix named by the protocol is prefilled with full attention and
    carries no loss, because nothing has been evicted there in a real run. The
    tail is scored teacher forced, with each query attending to the retained
    prefix and to its own chunk causally. A policy that evicts by attention
    score re-evicts between chunks with room reserved for the positions the next
    chunk adds, so the visible cache never rises above the budget at the moment
    attention is computed.

    A policy whose retained set depends only on position takes one pass over the
    whole sequence, with prefix queries exempted from the mask. Passing
    route="chunked" forces the two phase route instead, which is how the two are
    checked against each other.

    Returns a float32 tensor of shape (batch, sequence_length - prefill_tokens).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")

    from transformers.cache_utils import DynamicCache

    protocol = controller.protocol
    batch, total = input_ids.shape
    prefill = protocol.prefill_tokens
    protocol.scored_tokens(total)
    device = input_ids.device
    needs_scores = controller.policy.needs_attention_scores

    controller.begin_sequence(batch, total, device)

    if route not in ("auto", "single_pass", "chunked"):
        raise ValueError(f"unknown route {route!r}")
    if route == "single_pass" and needs_scores:
        raise ValueError(
            f"the {controller.policy.name} policy evicts between chunks and cannot run in one pass"
        )
    if route != "chunked" and not needs_scores:
        # The retained set depends only on position, so the prefix and the tail
        # go through in one pass: prefix queries are exempted from the mask and
        # tail queries carry it. This is the same computation as the chunked
        # route with one reduction order instead of several, and it avoids
        # growing the cache a chunk at a time.
        controller.set_masking(True)
        controller.set_prefill_exemption(prefill)
        # The first token of a sequence has no prediction to be scored against.
        first_scored = max(prefill, 1)
        keep = total - first_scored + 1
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=keep).logits
            step_logits = logits[:, :-1]
            targets = input_ids[:, first_scored:]
            token_loss = torch.nn.functional.cross_entropy(
                step_logits.reshape(-1, step_logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="none",
            )
            return token_loss.reshape(batch, -1)
        finally:
            controller.set_prefill_exemption(0)

    cache = DynamicCache()
    previous_logit: torch.Tensor | None = None

    # Eager attention materializes a query by key matrix, so a prefix that needs
    # attention weights is walked in chunks. A policy that reads no weights can
    # take the prefix in one pass.
    controller.set_masking(False)
    prefill_step = protocol.chunk_size if needs_scores else max(prefill, 1)
    first_tail = min(protocol.chunk_size, total - prefill)
    start = 0
    while start < prefill:
        stop = min(start + prefill_step, prefill)
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, stop, device=device),
                logits_to_keep=1,
            )
        # The prefix is evicted once, at the boundary, after every prefill chunk
        # has contributed its attention mass.
        controller.end_chunk(reserve=first_tail, evict=stop >= prefill)
        previous_logit = output.logits[:, -1]
        start = stop

    controller.set_masking(True)
    losses: list[torch.Tensor] = []
    start = prefill
    while start < total:
        stop = min(start + protocol.chunk_size, total)
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, stop, device=device),
            )
        logits = output.logits
        if previous_logit is None:
            # Nothing was prefilled, so the first token of the sequence has no
            # prediction to be scored against.
            step_logits = logits[:, :-1]
            targets = input_ids[:, start + 1 : stop]
        else:
            step_logits = torch.cat([previous_logit.unsqueeze(1), logits[:, :-1]], dim=1)
            targets = input_ids[:, start:stop]
        if targets.numel():
            token_loss = torch.nn.functional.cross_entropy(
                step_logits.reshape(-1, step_logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="none",
            )
            losses.append(token_loss.reshape(batch, -1))
        previous_logit = logits[:, -1]
        next_start = stop
        controller.end_chunk(reserve=min(protocol.chunk_size, max(total - next_start, 0)))
        start = stop

    return torch.cat(losses, dim=1)
