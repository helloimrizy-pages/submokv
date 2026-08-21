"""Tests for the four-state Sub-MoKV submodularity diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from submokv.submodularity import (
    KV,
    KV_GIVEN_WEIGHT,
    KV_TO_KV,
    WEIGHT,
    WEIGHT_GIVEN_KV,
    WEIGHT_TO_WEIGHT,
    InteractionSpec,
    TierUpgrade,
    build_interaction_matrix,
    classify_difference,
    format_diagnostic_report,
    four_state_plans,
    merge_submodularity_reports,
    parse_upgrades,
    run_submodularity_diagnostic,
    summarize_interactions,
    validate_test_layers,
)


def test_the_representative_four_layer_matrix_has_32_interactions() -> None:
    matrix = build_interaction_matrix((2, 10, 18, 26))
    counts = {
        modality: sum(spec.modality == modality for spec in matrix)
        for modality in (WEIGHT_TO_WEIGHT, KV_TO_KV, WEIGHT_GIVEN_KV, KV_GIVEN_WEIGHT)
    }
    assert len(matrix) == 32
    assert counts == {
        WEIGHT_TO_WEIGHT: 12,
        KV_TO_KV: 12,
        WEIGHT_GIVEN_KV: 4,
        KV_GIVEN_WEIGHT: 4,
    }
    assert len({spec.interaction_id for spec in matrix}) == len(matrix)


def test_extra_target_upgrades_expand_the_matrix_without_changing_the_layer_grid() -> None:
    weights = parse_upgrades("2:4,3:8", WEIGHT)
    kv = parse_upgrades("0.25:0.5,0.5:1", KV)
    matrix = build_interaction_matrix((0, 1, 2, 3), weights, kv)
    assert len(matrix) == 64
    assert {spec.target_layer for spec in matrix} == {0, 1, 2, 3}


def test_the_four_states_form_the_required_lattice_square() -> None:
    spec = InteractionSpec(
        target_layer=1,
        target=TierUpgrade(WEIGHT, 2, 4),
        conditioning_layer=1,
        conditioning=TierUpgrade(KV, 0.25, 1.0),
    )
    states = four_state_plans(spec, num_layers=3)
    assert states.s_a.weight_bits == (16, 2, 16)
    assert states.s_a.kv_retention == (1.0, 0.25, 1.0)
    assert states.s_a_union_j.weight_bits == (16, 4, 16)
    assert states.s_b.weight_bits == (16, 2, 16)
    assert states.s_b.kv_retention == (1.0, 1.0, 1.0)
    assert states.s_a.is_no_greater_than(states.s_b)
    assert states.s_a_union_j.is_no_greater_than(states.s_b_union_j)


def test_intra_component_states_use_distinct_target_and_conditioning_layers() -> None:
    with pytest.raises(ValueError, match="distinct layers"):
        InteractionSpec(
            1,
            TierUpgrade(WEIGHT, 2, 4),
            1,
            TierUpgrade(WEIGHT, 2, 16),
        )


def test_transition_parser_checks_supported_tiers_and_direction() -> None:
    assert parse_upgrades("2:4, 3:8", WEIGHT) == (
        TierUpgrade(WEIGHT, 2, 4),
        TierUpgrade(WEIGHT, 3, 8),
    )
    with pytest.raises(ValueError, match="outside supported tiers"):
        parse_upgrades("5:8", WEIGHT)
    with pytest.raises(ValueError, match="must increase"):
        parse_upgrades("0.75:0.5", KV)


def test_invalid_model_layer_is_rejected_instead_of_clipped() -> None:
    assert validate_test_layers((2, 6, 10, 14), 16) == (2, 6, 10, 14)
    with pytest.raises(ValueError, match="do not exist in a 16-layer model"):
        validate_test_layers((2, 10, 18, 26), 16)


def test_epsilon_controls_the_boundary_classification() -> None:
    assert classify_difference(-0.005, epsilon=0.01) == "submodular"
    assert classify_difference(-0.02, epsilon=0.01) == "supermodular"
    assert classify_difference(0.0, epsilon=0.0) == "submodular"


def _summary_row(diff: float, marginal_a: float = 1.0, marginal_b: float = 1.0):
    return {
        "modality": WEIGHT_GIVEN_KV,
        "classification": classify_difference(diff, 0.01),
        "second_order_difference": diff,
        "marginal_gain_s_a": marginal_a,
        "marginal_gain_s_b": marginal_b,
    }


def test_summary_applies_the_seventy_thirty_decision_rule() -> None:
    mostly_submodular = [_summary_row(0.1)] * 8 + [_summary_row(-0.1)] * 2
    summary = summarize_interactions(mostly_submodular, epsilon=0.01)
    assert summary["submodular_rate"] == pytest.approx(0.8)
    assert summary["verdict"] == "NEAR-SUBMODULAR / WEAKLY SUBMODULAR"

    synergy = [_summary_row(0.1)] * 6 + [_summary_row(-0.1)] * 4
    assert summarize_interactions(synergy, 0.01)["verdict"] == (
        "SUPERMODULAR / SYNERGY-DOMINATED"
    )


def test_summary_distinguishes_strict_negatives_inside_epsilon() -> None:
    summary = summarize_interactions([_summary_row(-0.005)], epsilon=0.01)
    assert summary["submodular_pairs"] == 1
    assert summary["strict_supermodular_pairs"] == 1
    assert summary["epsilon_band_pairs"] == 1


def test_core_weight_given_kv_synergy_cannot_be_diluted_by_intra_component_rows() -> None:
    intra = [
        {**_summary_row(0.1), "modality": WEIGHT_TO_WEIGHT}
        for _ in range(28)
    ]
    core_cross = [_summary_row(-0.1) for _ in range(4)]
    summary = summarize_interactions(intra + core_cross, epsilon=0.01)
    assert summary["submodular_rate"] == pytest.approx(28 / 32)
    assert summary["by_modality"][WEIGHT_GIVEN_KV]["strict_supermodular_rate"] == 1.0
    assert summary["verdict"] == "SUPERMODULAR / SYNERGY-DOMINATED"


def test_runner_logs_all_four_utilities_and_restores_the_tiny_model(tiny_utility) -> None:
    # One layer has no intra-layer pairs, leaving the two cross-component directions.
    matrix = build_interaction_matrix((1,))
    report = run_submodularity_diagnostic(tiny_utility, matrix, epsilon=0.0)
    assert len(report["interactions"]) == 2
    assert report["execution"]["unique_allocation_states"] >= 4
    for row in report["interactions"]:
        assert set(row["states"]) == {"s_a", "s_a_union_j", "s_b", "s_b_union_j"}
        a = row["states"]["s_a"]["utility"]
        aj = row["states"]["s_a_union_j"]["utility"]
        b = row["states"]["s_b"]["utility"]
        bj = row["states"]["s_b_union_j"]["utility"]
        assert row["marginal_gain_s_a"] == pytest.approx(aj - a)
        assert row["marginal_gain_s_b"] == pytest.approx(bj - b)
        assert row["second_order_difference"] == pytest.approx((aj - a) - (bj - b))
    assert set(tiny_utility.controller.retention().values()) == {1.0}
    assert {
        bits
        for layer in tiny_utility.quantizer.current_plan().values()
        for bits in layer.values()
    } == {16}
    json.dumps(report)


def test_formatted_report_contains_the_verdict_and_units(tiny_utility) -> None:
    report = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,)), epsilon=0.01
    )
    rendered = format_diagnostic_report(report)
    assert "SUB-MoKV SUBMODULARITY DIAGNOSTIC REPORT" in rendered
    assert "Delta(j|S_A)" in rendered
    assert "DECISION VERDICT:" in rendered
    assert "Sequences: 2" in rendered


def test_standalone_script_dry_run_needs_no_model_or_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "submodularity_diagnostic.py"),
            "--config",
            str(root / "configs" / "olmoe.yaml"),
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["test_layers"] == [2, 6, 10, 14]
    assert payload["calibration"]["split"] == "validation"
    assert payload["calibration"]["sequence_length"] == 2048
    assert payload["num_interactions"] == 32

    shard = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "submodularity_diagnostic.py"),
            "--config",
            str(root / "configs" / "olmoe.yaml"),
            "--dry-run",
            "--shard",
            "1",
            "--num-shards",
            "2",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    shard_payload = json.loads(shard.stdout)
    assert shard_payload["shard"] == 1
    assert shard_payload["num_shard_interactions"] == 16


def _split_report(report, num_shards: int):
    order = [row["interaction_id"] for row in report["interactions"]]
    shards = []
    for shard in range(num_shards):
        partial = deepcopy(report)
        partial["interactions"] = report["interactions"][shard::num_shards]
        partial["execution"].update(
            {
                "shard": shard,
                "num_shards": num_shards,
                "full_matrix_interactions": len(order),
                "full_matrix_order": order,
            }
        )
        shards.append(partial)
    return shards


def test_two_shard_reports_merge_back_in_full_matrix_order(tiny_utility, tmp_path) -> None:
    full = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0, 1)), epsilon=0.01
    )
    shards = _split_report(full, 2)
    merged = merge_submodularity_reports(shards)
    assert [row["interaction_id"] for row in merged["interactions"]] == [
        row["interaction_id"] for row in full["interactions"]
    ]
    assert merged["execution"]["merged_from_shards"] == [0, 1]
    assert merged["summary"]["total_interactions"] == len(full["interactions"])

    root = Path(__file__).resolve().parents[1]
    shard_paths = []
    for index, shard in enumerate(shards):
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        shard_paths.append(path)
    output = tmp_path / "merged.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "submodularity_diagnostic.py"),
            "--merge-shards",
            *(str(path) for path in shard_paths),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["execution"]["merged_from_shards"] == [0, 1]


def test_shard_merge_refuses_a_reference_mismatch(tiny_utility) -> None:
    full = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,)), epsilon=0.01
    )
    shards = _split_report(full, 2)
    shards[1]["reference"]["perplexity"] += 0.5
    with pytest.raises(ValueError, match="disagree on the full-precision reference"):
        merge_submodularity_reports(shards)


def test_shard_merge_refuses_an_incomplete_worker_set(tiny_utility) -> None:
    full = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,)), epsilon=0.01
    )
    first = _split_report(full, 2)[0]
    with pytest.raises(ValueError, match="incomplete shard set"):
        merge_submodularity_reports([first])
