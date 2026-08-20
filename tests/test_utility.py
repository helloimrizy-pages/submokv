"""Tests for the deterministic perplexity utility."""

from __future__ import annotations

import pytest
import torch

from submokv.utility import CHEAP, FULL, CalibrationSpec, SequenceStore, UtilityFunction


def make_spec(**overrides) -> CalibrationSpec:
    """Return a calibration spec sized for the tiny test model."""
    settings = dict(sequence_length=16, cheap_sequences=2, full_sequences=4, seed=0)
    settings.update(overrides)
    return CalibrationSpec(**settings)


def make_store(spec: CalibrationSpec | None = None) -> SequenceStore:
    """Return a store over fixed random windows, with the two pools disjoint."""
    spec = spec or make_spec()
    generator = torch.Generator().manual_seed(11)
    pool = torch.randint(0, 64, (12, spec.sequence_length), generator=generator)
    return SequenceStore.from_windows(pool[:8], pool[8:], spec)


def test_subsample_blocks_do_not_overlap() -> None:
    store = make_store()
    first = store.subsample(CHEAP, 0)
    second = store.subsample(CHEAP, 1)
    assert first.shape == (2, 16)
    assert not bool((first == second).all())
    assert store.max_subsamples(CHEAP) == 4


def test_subsample_is_the_same_every_call() -> None:
    """F must not sample anything, so the same request returns the same rows."""
    store = make_store()
    assert torch.equal(store.subsample(CHEAP, 1), store.subsample(CHEAP, 1))


def test_a_subsample_beyond_the_pool_is_refused() -> None:
    store = make_store()
    with pytest.raises(ValueError, match="not enough"):
        store.subsample(CHEAP, 4)


def test_calibration_and_evaluation_pools_must_be_disjoint() -> None:
    """Picking an allocation on the data it is scored on would make the result worthless."""
    spec = make_spec()
    generator = torch.Generator().manual_seed(3)
    pool = torch.randint(0, 64, (8, spec.sequence_length), generator=generator)
    with pytest.raises(ValueError, match="appear in both"):
        SequenceStore.from_windows(pool, pool[:4], spec)


def test_a_shared_split_name_is_refused() -> None:
    generator = torch.Generator().manual_seed(3)
    pool = torch.randint(0, 64, (8, 16), generator=generator)
    spec = make_spec(calibration_split="train", evaluation_split="train")
    with pytest.raises(ValueError, match="both read split"):
        SequenceStore.from_windows(pool[:4], pool[4:], spec)


def test_fidelity_sizes(tmp_path) -> None:
    spec = make_spec()
    assert spec.size(CHEAP) == 2
    assert spec.size(FULL) == 4
    with pytest.raises(ValueError, match="unknown fidelity"):
        spec.size("guess")
