"""Tests for result records."""

from __future__ import annotations

import json

import pytest

from submokv.records import ResultRecord, load_records, record


def test_a_record_carries_the_config_commit_seed_and_wall_clock(tmp_path) -> None:
    with record("demo", {"budget": 0.35}, seed=7, results_dir=tmp_path) as entry:
        entry.payload["value"] = 1.25
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    loaded = json.loads(written[0].read_text())
    assert loaded["name"] == "demo"
    assert loaded["seed"] == 7
    assert loaded["config"] == {"budget": 0.35}
    assert loaded["payload"]["value"] == 1.25
    assert loaded["git_commit"]
    assert loaded["wall_clock_seconds"] >= 0.0
    assert loaded["environment"]["torch"]


def test_a_failed_run_still_leaves_a_record(tmp_path) -> None:
    """A run that raises must leave evidence rather than nothing."""
    with pytest.raises(RuntimeError):
        with record("broken", {}, seed=0, results_dir=tmp_path):
            raise RuntimeError("evaluation died")
    loaded = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "evaluation died" in loaded["payload"]["error"]


def test_load_records_filters_by_name(tmp_path) -> None:
    ResultRecord(name="a", config={}, seed=0, started_at="20260820T000000Z").write(tmp_path)
    ResultRecord(name="b", config={}, seed=0, started_at="20260820T000001Z").write(tmp_path)
    assert len(load_records(tmp_path)) == 2
    assert [r["name"] for r in load_records(tmp_path, name="b")] == ["b"]
    assert load_records(tmp_path / "missing") == []
