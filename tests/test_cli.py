"""Tests for command-line path resolution that do not load model weights."""

from __future__ import annotations

from pathlib import Path

from submokv.cli import resolve_model_path


def test_filtered_cached_snapshot_is_accepted_when_hub_calls_it_incomplete(
    tmp_path, monkeypatch
) -> None:
    import huggingface_hub
    from huggingface_hub import constants

    model_name = "owner/model"
    snapshot = tmp_path / "models--owner--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    def incomplete(*args, **kwargs):
        raise RuntimeError("README and logo are missing")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", incomplete)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    resolved = resolve_model_path({"model": {"name": model_name}}, None)
    assert resolved == str(snapshot)


def test_explicit_model_path_does_not_contact_the_hub(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def unexpected(*args, **kwargs):
        raise AssertionError("the hub should not be contacted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    assert resolve_model_path({"model": {"name": "owner/model"}}, snapshot) == str(snapshot)
