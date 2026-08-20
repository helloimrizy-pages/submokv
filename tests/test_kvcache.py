"""Tests for per-layer KV cache retention."""

from __future__ import annotations

import pytest
import torch

from submokv.kvcache import (
    AttentionScorePolicy,
    EvaluationProtocol,
    RecencySinkPolicy,
    RetentionController,
    build_policy,
    find_attention_modules,
    forward_with_retention,
    score_sequence,
)
from submokv.memory import KVSpec

# The chunked route reduces over a different set of shapes than a single pass,
# so the two agree in exact arithmetic but not bit for bit.
CHUNK_TOLERANCE = 1e-5


def test_find_attention_modules_returns_one_module_per_layer(tiny_model) -> None:
    assert [entry.layer for entry in find_attention_modules(tiny_model)] == [0, 1, 2]


def test_recency_mask_keeps_the_sinks_and_the_most_recent_positions() -> None:
    policy = RecencySinkPolicy(sink_tokens=2)
    allowed = policy.allowed(torch.arange(16), key_length=16, budget=4, state=None)
    assert allowed[10].nonzero().flatten().tolist() == [0, 1, 9, 10]
    assert allowed[3].nonzero().flatten().tolist() == [0, 1, 2, 3]


def test_recency_mask_never_exceeds_the_budget() -> None:
    policy = RecencySinkPolicy(sink_tokens=2)
    allowed = policy.allowed(torch.arange(16), key_length=16, budget=4, state=None)
    assert allowed.sum(-1).tolist() == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def test_recency_mask_admits_no_future_key() -> None:
    policy = RecencySinkPolicy(sink_tokens=2)
    allowed = policy.allowed(torch.arange(16), key_length=16, budget=8, state=None)
    keys = torch.arange(16).reshape(1, -1)
    assert not bool((allowed & (keys > torch.arange(16).reshape(-1, 1))).any())


def test_full_retention_reproduces_the_unmodified_logits_exactly(
    tiny_model, tiny_ids, tiny_kv
) -> None:
    """Every layer at retention 1.00 must leave the model bit for bit unchanged."""
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    after = forward_with_retention(tiny_model, tiny_ids, controller)
    assert torch.equal(before, after)


def test_retention_below_one_changes_the_logits(tiny_model, tiny_ids, tiny_kv) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.25)
    after = forward_with_retention(tiny_model, tiny_ids, controller)
    assert not torch.equal(before, after)


def test_retention_can_be_toggled_back_without_reloading(tiny_model, tiny_ids, tiny_kv) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.25)
    forward_with_retention(tiny_model, tiny_ids, controller)
    controller.set_uniform_retention(1.0)
    assert torch.equal(before, forward_with_retention(tiny_model, tiny_ids, controller))


def test_retention_is_set_per_layer(tiny_model, tiny_kv) -> None:
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv)
    controller.set_retention({0: 0.25, 2: 0.75})
    assert controller.retention() == {0: 0.25, 1: 1.0, 2: 0.75}
    assert {layer: controller.budget_tokens(layer, 16) for layer in controller.layers} == {
        0: 4,
        1: 16,
        2: 12,
    }


def test_a_masked_query_attends_to_no_more_keys_than_its_budget(
    tiny_model_eager, tiny_ids, tiny_kv
) -> None:
    """The mask must actually bite, not merely be built."""
    recorded: dict[str, torch.Tensor] = {}

    def record(module, args, kwargs, output):
        recorded["weights"] = output[1].detach()

    handle = tiny_model_eager.model.layers[0].self_attn.register_forward_hook(
        record, with_kwargs=True
    )
    controller = RetentionController(
        tiny_model_eager, RecencySinkPolicy(sink_tokens=2), tiny_kv
    ).attach()
    controller.set_uniform_retention(0.25)
    forward_with_retention(tiny_model_eager, tiny_ids, controller)
    handle.remove()

    attended = (recorded["weights"][0, 0] > 1e-12).sum(-1)
    assert int(attended.max().item()) <= controller.budget_tokens(0, 16)
    assert attended.tolist() == [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def test_a_position_independent_policy_gives_the_same_result_chunked_or_not(
    tiny_model, tiny_ids, tiny_kv
) -> None:
    """The recency mask is a function of position alone, so one pass simulates it."""
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.5)
    single = forward_with_retention(tiny_model, tiny_ids, controller)
    chunked = forward_with_retention(tiny_model, tiny_ids, controller, chunk_size=4)
    assert torch.allclose(single, chunked, atol=CHUNK_TOLERANCE)


def test_attention_score_policy_reduces_each_head_to_its_budget(
    tiny_model_eager, tiny_ids, tiny_kv
) -> None:
    controller = RetentionController(
        tiny_model_eager, AttentionScorePolicy(sink_tokens=2, recent_window=2), tiny_kv
    ).attach()
    controller.set_uniform_retention(0.5)
    forward_with_retention(tiny_model_eager, tiny_ids, controller, chunk_size=4)
    state = controller._state[0]
    budget = controller.budget_tokens(0, 16)
    assert state.retained is not None
    assert state.retained.sum(-1).flatten().tolist() == [budget] * 4


def test_attention_score_policy_keeps_a_separate_set_for_each_sequence(
    tiny_model_eager, tiny_kv
) -> None:
    """Retained positions depend on the data, so two sequences must not share a set."""
    generator = torch.Generator().manual_seed(7)
    batch = torch.randint(0, 64, (3, 16), generator=generator)
    controller = RetentionController(
        tiny_model_eager, AttentionScorePolicy(sink_tokens=2, recent_window=2), tiny_kv
    ).attach()
    controller.set_uniform_retention(0.5)
    logits = forward_with_retention(tiny_model_eager, batch, controller, chunk_size=4)
    assert logits.shape[:2] == (3, 16)
    retained = controller._state[0].retained
    assert not bool((retained[0] == retained[1]).all())


def test_attention_score_policy_always_keeps_the_sinks(tiny_model_eager, tiny_ids, tiny_kv) -> None:
    controller = RetentionController(
        tiny_model_eager, AttentionScorePolicy(sink_tokens=2, recent_window=2), tiny_kv
    ).attach()
    controller.set_uniform_retention(0.5)
    forward_with_retention(tiny_model_eager, tiny_ids, controller, chunk_size=4)
    retained = controller._state[0].retained
    assert bool(retained[..., :2].all())


def test_attention_score_policy_needs_chunks(tiny_model_eager, tiny_ids, tiny_kv) -> None:
    controller = RetentionController(
        tiny_model_eager, AttentionScorePolicy(sink_tokens=2), tiny_kv
    ).attach()
    controller.set_uniform_retention(0.5)
    with pytest.raises(ValueError, match="chunk_size is required"):
        forward_with_retention(tiny_model_eager, tiny_ids, controller)


def test_attention_score_policy_reports_a_missing_attention_implementation(
    tiny_model, tiny_ids, tiny_kv
) -> None:
    """sdpa returns no attention weights, so the policy must say so rather than fail quietly."""
    controller = RetentionController(tiny_model, AttentionScorePolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.5)
    with pytest.raises(RuntimeError, match="attn_implementation='eager'"):
        forward_with_retention(tiny_model, tiny_ids, controller, chunk_size=4)


def test_detach_removes_the_hooks(tiny_model, tiny_ids, tiny_kv) -> None:
    with torch.no_grad():
        before = tiny_model(input_ids=tiny_ids).logits
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.25)
    controller.detach()
    with torch.no_grad():
        after = tiny_model(input_ids=tiny_ids).logits
    assert torch.equal(before, after)


def test_controller_rejects_an_unknown_layer(tiny_model, tiny_kv) -> None:
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv)
    with pytest.raises(KeyError):
        controller.set_retention({9: 0.5})


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_controller_rejects_a_ratio_outside_its_range(tiny_model, tiny_kv, ratio: float) -> None:
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv)
    with pytest.raises(ValueError):
        controller.set_retention({0: ratio})


def test_ragged_batch_positions_are_refused(tiny_model, tiny_kv) -> None:
    """A padded batch would put different sequences at different positions."""
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
    controller.set_uniform_retention(0.5)
    ids = torch.zeros(2, 8, dtype=torch.long)
    positions = torch.stack([torch.arange(8), torch.arange(8) + 3])
    with pytest.raises(NotImplementedError, match="same positions"):
        with torch.no_grad():
            tiny_model(input_ids=ids, position_ids=positions)


def test_build_policy_reads_the_pinned_name() -> None:
    policy = build_policy({"policy": "recency_sink", "sink_tokens": 8})
    assert isinstance(policy, RecencySinkPolicy)
    assert policy.sink_tokens == 8
    assert build_policy({"policy": "attention_score"}).name == "attention_score"
    with pytest.raises(ValueError, match="unknown retention policy"):
        build_policy({"policy": "made_up"})


def test_describe_records_the_policy_and_the_retention(tiny_model, tiny_kv) -> None:
    controller = RetentionController(tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv)
    controller.set_retention({0: 0.25})
    summary = controller.describe()
    assert summary["policy"] == "recency_sink"
    assert summary["sink_tokens"] == 2
    assert summary["retention_by_layer"] == {"0": 0.25, "1": 1.0, "2": 1.0}


def test_budget_tokens_uses_the_sequence_length_not_the_declared_context(tiny_model) -> None:
    """A ratio must mean the same share of the cache in the hooks as in the accounting."""
    controller = RetentionController(tiny_model, RecencySinkPolicy(), KVSpec(context_length=4096))
    controller.set_uniform_retention(0.25)
    assert controller.budget_tokens(0, 4096) == 1024
    assert controller.budget_tokens(0, 512) == 128


def test_every_mask_form_transformers_produces_is_handled(tiny_model, tiny_model_eager, tiny_ids, tiny_kv) -> None:
    """The mask arrives as None, as a boolean tensor, or as an additive float tensor."""
    forms: set[str] = set()

    def watch(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        forms.add("none" if mask is None else str(mask.dtype))
        return None

    for model, chunk in ((tiny_model, None), (tiny_model, 4), (tiny_model_eager, None)):
        handle = model.model.layers[0].self_attn.register_forward_pre_hook(watch, with_kwargs=True)
        controller = RetentionController(model, RecencySinkPolicy(sink_tokens=2), tiny_kv).attach()
        controller.set_uniform_retention(0.5)
        forward_with_retention(model, tiny_ids, controller, chunk_size=chunk)
        controller.detach()
        handle.remove()

    assert forms == {"none", "torch.bool", "torch.float32"}


def test_policy_and_accounting_must_agree_on_the_sinks(tiny_model) -> None:
    with pytest.raises(ValueError, match="sink tokens"):
        RetentionController(
            tiny_model, RecencySinkPolicy(sink_tokens=8), KVSpec(context_length=16, sink_tokens=2)
        )


def test_shipped_config_pins_one_policy() -> None:
    from pathlib import Path

    from submokv.cli import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "olmoe.yaml")
    policy = build_policy(config["retention"])
    assert policy.name == "recency_sink"
    assert policy.sink_tokens == config["kv"]["sink_tokens"]


def _plain_tail_nll(model, ids, prefill: int) -> torch.Tensor:
    """Return the per-token tail loss of an unmodified full attention pass."""
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    step_logits = logits[:, prefill - 1 : -1]
    targets = ids[:, prefill:]
    loss = torch.nn.functional.cross_entropy(
        step_logits.reshape(-1, step_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    )
    return loss.reshape(ids.shape[0], -1)


def test_the_protocol_scores_the_tail_only(tiny_model, tiny_ids, tiny_kv) -> None:
    protocol = EvaluationProtocol(prefill_tokens=8, chunk_size=4)
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv, protocol
    ).attach()
    losses = score_sequence(tiny_model, tiny_ids, controller)
    assert losses.shape == (1, 8)
    assert protocol.scored_tokens(16) == 8


def test_full_retention_under_the_protocol_matches_a_plain_pass(
    tiny_model, tiny_ids, tiny_kv
) -> None:
    """With nothing evicted, the two phase route must agree with one full attention pass."""
    protocol = EvaluationProtocol(prefill_tokens=8, chunk_size=4)
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv, protocol
    ).attach()
    losses = score_sequence(tiny_model, tiny_ids, controller)
    assert torch.allclose(losses, _plain_tail_nll(tiny_model, tiny_ids, 8), atol=CHUNK_TOLERANCE)


def test_lower_retention_raises_the_tail_loss(tiny_model, tiny_ids, tiny_kv) -> None:
    protocol = EvaluationProtocol(prefill_tokens=8, chunk_size=4)
    controller = RetentionController(
        tiny_model, RecencySinkPolicy(sink_tokens=2), tiny_kv, protocol
    ).attach()
    full = score_sequence(tiny_model, tiny_ids, controller).mean().item()
    controller.set_uniform_retention(0.25)
    reduced = score_sequence(tiny_model, tiny_ids, controller).mean().item()
    assert reduced != full


def test_the_visible_cache_never_exceeds_the_budget(tiny_model_eager, tiny_ids, tiny_kv) -> None:
    """The peak visible cache is what memory.py charges for, so it is checked directly."""
    seen: list[int] = []

    def record(module, args, kwargs, output):
        weights = output[1]
        if weights is not None:
            seen.append(int((weights[0, 0] > 1e-12).sum(-1).max().item()))

    protocol = EvaluationProtocol(prefill_tokens=8, chunk_size=4)
    controller = RetentionController(
        tiny_model_eager, AttentionScorePolicy(sink_tokens=2, recent_window=2), tiny_kv, protocol
    ).attach()
    controller.set_uniform_retention(0.5)
    handle = tiny_model_eager.model.layers[0].self_attn.register_forward_hook(
        record, with_kwargs=True
    )
    score_sequence(tiny_model_eager, tiny_ids, controller)
    handle.remove()

    budget = controller.budget_tokens(0, 16)
    tail_peaks = seen[protocol.prefill_tokens // protocol.chunk_size :]
    assert max(tail_peaks) <= budget


def test_describe_records_the_protocol_next_to_the_policy(tiny_model, tiny_kv) -> None:
    controller = RetentionController(
        tiny_model,
        RecencySinkPolicy(sink_tokens=2),
        tiny_kv,
        EvaluationProtocol(prefill_tokens=3072, chunk_size=256),
    )
    summary = controller.describe()
    assert summary["policy"] == "recency_sink"
    assert summary["protocol"] == "prefill_evict_decode"
    assert summary["prefill_tokens"] == 3072


def test_an_unimplemented_protocol_is_refused() -> None:
    with pytest.raises(ValueError, match="only the prefill_evict_decode protocol"):
        EvaluationProtocol(name="sliding")


def test_a_prefix_that_swallows_the_sequence_is_refused() -> None:
    with pytest.raises(ValueError, match="leaves no tail"):
        EvaluationProtocol(prefill_tokens=16).scored_tokens(16)
