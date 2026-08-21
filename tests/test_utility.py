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


def test_summing_from_the_accelerator_widens_on_the_host() -> None:
    """A one step device and dtype change silently corrupts float64 sums from MPS.

    torch.Tensor.to("cpu", torch.float64) attempts the widening on the source
    device, which has no float64, and returns reinterpreted bits rather than
    raising. Every perplexity in the project flows through this sum.
    """
    from submokv.utility import total_in_float64

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    values = torch.rand(4, 1024, device=device) + 1.0
    expected = values.float().sum().item()
    assert total_in_float64(values) == pytest.approx(expected, rel=1e-6)


def test_the_one_step_conversion_is_the_thing_being_avoided() -> None:
    """Pin the behaviour that motivated the two step form, so a fix upstream is noticed."""
    if not torch.backends.mps.is_available():
        pytest.skip("the corruption only appears on MPS")
    values = torch.rand(4, 1024, device="mps") + 1.0
    one_step = values.detach().to("cpu", torch.float64).sum().item()
    two_step = values.detach().cpu().to(torch.float64).sum().item()
    assert two_step == pytest.approx(values.float().sum().item(), rel=1e-6)
    if one_step == pytest.approx(two_step, rel=1e-6):
        pytest.skip("torch no longer corrupts the one step conversion on MPS")


def test_the_device_is_chosen_when_the_config_names_none() -> None:
    """The same config has to run on whatever machine holds the weights."""
    from submokv.utility import default_device

    chosen = default_device()
    assert chosen in {"cuda", "mps", "cpu"}
    if torch.cuda.is_available():
        assert chosen == "cuda"
    elif torch.backends.mps.is_available():
        assert chosen == "mps"


def test_the_shipped_config_pins_no_device() -> None:
    from pathlib import Path

    from submokv.cli import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "olmoe.yaml")
    assert config["runtime"]["device"] is None


def test_pretokenized_store_records_its_tokenizer_provenance() -> None:
    store = make_store()
    assert store.describe()["tokenizer_source"] == "pretokenized_windows"
