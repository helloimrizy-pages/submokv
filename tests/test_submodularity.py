"""Tests for the four-state Sub-MoKV submodularity diagnostic."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from submokv.ground_set import GroundSet
from submokv.submodularity import (
    KV,
    KV_GIVEN_WEIGHT,
    KV_TO_KV,
    WEIGHT,
    WEIGHT_GIVEN_KV,
    WEIGHT_TO_WEIGHT,
    EpsilonPolicy,
    InteractionSpec,
    OutsideGroundSetError,
    TierLadders,
    TierUpgrade,
    build_interaction_matrix,
    default_floor_specs,
    classify_difference,
    format_diagnostic_report,
    four_state_plans,
    merge_submodularity_reports,
    parse_upgrades,
    run_submodularity_diagnostic,
    summarize_interactions,
    validate_test_layers,
)

# The ladders the shipped OLMoE config declares, which is what the allocator
# may buy. 2 bits is deliberately absent.
LADDERS = TierLadders(weight_tiers=(3, 4, 8, 16), kv_tiers=(0.25, 0.50, 0.75, 1.00))


def test_the_representative_four_layer_matrix_has_32_interactions() -> None:
    matrix = build_interaction_matrix((2, 10, 18, 26), LADDERS)
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
    weights = parse_upgrades("3:4,4:8", WEIGHT)
    kv = parse_upgrades("0.25:0.5,0.5:1", KV)
    matrix = build_interaction_matrix((0, 1, 2, 3), LADDERS, weights, kv)
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
    with pytest.raises(ValueError, match="outside expressible tiers"):
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
        "epsilon_used": 0.01,
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
    assert summary["strict_negative_pairs"] == 1
    assert summary["epsilon_band_pairs"] == 1
    # Inside epsilon is evidence of nothing, so it must not be counted as one.
    assert summary["resolved_pairs"] == 0


def test_core_weight_given_kv_synergy_cannot_be_diluted_by_intra_component_rows() -> None:
    intra = [
        {**_summary_row(0.1), "modality": WEIGHT_TO_WEIGHT}
        for _ in range(28)
    ]
    core_cross = [_summary_row(-0.1) for _ in range(4)]
    summary = summarize_interactions(intra + core_cross, epsilon=0.01)
    assert summary["submodular_rate"] == pytest.approx(28 / 32)
    assert summary["by_modality"][WEIGHT_GIVEN_KV]["supermodular_rate"] == 1.0
    assert summary["verdict"] == "SUPERMODULAR / SYNERGY-DOMINATED"


def test_runner_logs_all_four_utilities_and_restores_the_tiny_model(tiny_utility) -> None:
    # One layer has no intra-layer pairs, leaving the two cross-component directions.
    matrix = build_interaction_matrix((1,), LADDERS)
    report = run_submodularity_diagnostic(tiny_utility, matrix, epsilon=0.0)
    assert len(report["interactions"]) == 2
    assert report["execution"]["unique_allocation_states"] >= 4
    for row in report["interactions"]:
        assert [entry["subsample"] for entry in row["per_subsample"]] == row["subsamples"]
        for entry in row["per_subsample"]:
            assert set(entry) >= {"ppl_s_a", "ppl_s_a_union_j", "ppl_s_b", "ppl_s_b_union_j"}
            assert entry["marginal_gain_s_a"] == pytest.approx(
                entry["ppl_s_a"] - entry["ppl_s_a_union_j"]
            )
            assert entry["marginal_gain_s_b"] == pytest.approx(
                entry["ppl_s_b"] - entry["ppl_s_b_union_j"]
            )
            assert entry["second_order_difference"] == pytest.approx(
                entry["marginal_gain_s_a"] - entry["marginal_gain_s_b"]
            )
        # The cell is the mean over its draws, never a single number.
        assert row["second_order_difference"] == pytest.approx(
            statistics.fmean(e["second_order_difference"] for e in row["per_subsample"])
        )
    assert set(tiny_utility.controller.retention().values()) == {1.0}
    assert {
        bits
        for layer in tiny_utility.quantizer.current_plan().values()
        for bits in layer.values()
    } == {16}
    json.dumps(report)


def test_formatted_report_contains_the_verdict_and_units(tiny_utility) -> None:
    report = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
    )
    rendered = format_diagnostic_report(report)
    assert "SUB-MoKV SUBMODULARITY DIAGNOSTIC REPORT" in rendered
    assert "mean D" in rendered
    assert "DECISION VERDICT:" in rendered
    assert "2 sequences" in rendered
    # The asymmetry between target and conditioning moves must be stated, because
    # a reader will otherwise assume both sides are adjacent.
    assert "TARGET moves are ADJACENT" in rendered
    assert "CONDITIONING moves are LARGE" in rendered


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
    # The calibration settings come from the config, not from a flag default;
    # a floor and the run it calibrates must read the same windows.
    assert payload["calibration"]["split"] == "train"
    assert payload["calibration"]["sequence_length"] == 4096
    assert payload["calibration"]["sequences"] == 64
    # Two adjacent target steps per ladder over four layers.
    assert payload["num_interactions"] == 64
    assert payload["tier_ladders"]["weight_tiers"] == [3, 4, 8, 16]
    targets = {f"{r['target_upgrade']['from']}->{r['target_upgrade']['to']}"
               for r in payload["interactions"]}
    context = {f"{r['conditioning_upgrade']['from']}->{r['conditioning_upgrade']['to']}"
               for r in payload["interactions"]}
    assert targets <= {"3->4", "4->8", "0.25->0.5", "0.5->0.75"}
    # Conditioning is deliberately NOT adjacent.
    assert context <= {"3->16", "0.25->1"}

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
    assert shard_payload["num_shard_interactions"] == 32


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
        tiny_utility, build_interaction_matrix((0, 1), LADDERS), epsilon=0.01
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
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
    )
    shards = _split_report(full, 2)
    shards[1]["reference"]["perplexity"] += 0.5
    with pytest.raises(ValueError, match="disagree on the full-precision reference"):
        merge_submodularity_reports(shards)


def test_shard_merge_refuses_an_incomplete_worker_set(tiny_utility) -> None:
    full = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
    )
    first = _split_report(full, 2)[0]
    with pytest.raises(ValueError, match="incomplete shard set"):
        merge_submodularity_reports([first])


def _olmoe_config() -> dict:
    """Return the shipped OLMoE config, which is what a run actually reads."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs" / "olmoe.yaml").read_text(encoding="utf-8"))


def test_a_config_level_tier_change_is_visible_in_the_interaction_matrix() -> None:
    """The ground set, not a module constant, decides which tiers a probe may move along.

    A run once reported every weight row as a 2->4 or 2->16 move on a tier the
    config had removed from the search space, because the matrix builder held
    its own hardcoded ladder. Changing ground_set.weight_tiers must change the
    matrix.
    """
    config = _olmoe_config()
    assert config["ground_set"]["weight_tiers"] == [3, 4, 8, 16]

    shipped = TierLadders.from_ground_set(GroundSet.from_config(config))
    shipped_matrix = build_interaction_matrix((0, 1), shipped)
    shipped_moves = {
        move.label
        for spec in shipped_matrix
        for move in (spec.target, spec.conditioning)
        if move.kind == WEIGHT
    }
    assert shipped_moves == {"W:3->4", "W:3->16"}
    assert all(spec.in_ground_set(shipped) for spec in shipped_matrix)

    narrowed = dict(config)
    narrowed["ground_set"] = {**config["ground_set"], "weight_tiers": [4, 8]}
    changed = TierLadders.from_ground_set(GroundSet.from_config(narrowed))
    changed_matrix = build_interaction_matrix((0, 1), changed)
    changed_moves = {
        move.label
        for spec in changed_matrix
        for move in (spec.target, spec.conditioning)
        if move.kind == WEIGHT
    }
    assert changed_moves == {"W:4->8"}
    assert changed_moves != shipped_moves

    # And the shipped ladder's moves are now off-ladder under the narrowed one.
    assert not all(spec.in_ground_set(changed) for spec in shipped_matrix)


def test_a_kv_tier_change_in_the_config_moves_the_kv_rows_too() -> None:
    config = _olmoe_config()
    narrowed = dict(config)
    narrowed["ground_set"] = {**config["ground_set"], "kv_tiers": [0.50, 1.00]}
    ladders = TierLadders.from_ground_set(GroundSet.from_config(narrowed))
    matrix = build_interaction_matrix((0, 1), ladders)
    kv_moves = {
        move.label
        for spec in matrix
        for move in (spec.target, spec.conditioning)
        if move.kind == KV
    }
    assert kv_moves == {"KV:0.50->1"}


def test_the_ladders_report_which_tiers_the_allocator_can_buy() -> None:
    assert LADDERS.contains(TierUpgrade(WEIGHT, 3, 4))
    assert not LADDERS.contains(TierUpgrade(WEIGHT, 2, 4))
    assert LADDERS.adjacent(WEIGHT) == (
        TierUpgrade(WEIGHT, 3, 4),
        TierUpgrade(WEIGHT, 4, 8),
        TierUpgrade(WEIGHT, 8, 16),
    )
    assert LADDERS.full_move(KV) == TierUpgrade(KV, 0.25, 1.00)
    assert LADDERS.reference_in_ground_set()
    assert not TierLadders(weight_tiers=(3, 4), kv_tiers=(0.25, 1.0)).reference_in_ground_set()


def test_a_probe_outside_the_ground_set_is_refused_by_default(tiny_utility) -> None:
    off_ladder = (
        InteractionSpec(0, TierUpgrade(WEIGHT, 2, 4), 0, TierUpgrade(KV, 0.25, 1.0)),
    )
    with pytest.raises(OutsideGroundSetError, match="does not contain"):
        run_submodularity_diagnostic(tiny_utility, off_ladder, epsilon=0.01)


def test_an_allowed_outside_probe_is_labelled_rather_than_silently_reported(
    tiny_utility,
) -> None:
    matrix = (
        InteractionSpec(0, TierUpgrade(WEIGHT, 2, 4), 0, TierUpgrade(KV, 0.25, 1.0)),
        InteractionSpec(0, TierUpgrade(WEIGHT, 3, 4), 0, TierUpgrade(KV, 0.25, 1.0)),
    )
    report = run_submodularity_diagnostic(
        tiny_utility, matrix, epsilon=0.01, allow_outside_ground_set=True
    )
    flags = [row["in_ground_set"] for row in report["interactions"]]
    assert flags == [False, True]
    assert report["interactions"][0]["target_upgrade"]["in_ground_set"] is False
    assert report["ground_set_scope"]["interactions_outside_ground_set"] == 1
    assert "OUTSIDE the ground set" in format_diagnostic_report(report)


def test_the_headline_counter_and_the_classification_column_use_one_test() -> None:
    """Regression: the summary once printed 100% submodular above 15.62% supermodular.

    The column showed the epsilon test while the counter showed the strict sign
    test, so the two lines contradicted each other on the same rows.
    """
    rows = [_summary_row(0.1)] * 3 + [_summary_row(-0.005)] * 5
    summary = summarize_interactions(rows, epsilon=0.01)

    column_submodular = sum(row["classification"] == "submodular" for row in rows)
    assert summary["headline_epsilon_test"]["submodular_pairs"] == column_submodular
    assert summary["headline_epsilon_test"]["supermodular_pairs"] == 8 - column_submodular
    assert (
        summary["headline_epsilon_test"]["submodular_pairs"]
        + summary["headline_epsilon_test"]["supermodular_pairs"]
        == summary["total_interactions"]
    )
    # The strict count is still reported, but as a labelled secondary line.
    assert summary["secondary_strict_sign_test"]["strict_negative_pairs"] == 5
    assert summary["secondary_strict_sign_test"]["inside_epsilon_band_pairs"] == 5
    assert summary["tests"]["headline"].startswith("epsilon test")
    assert summary["tests"]["secondary"].startswith("strict sign test")


def test_cells_inside_epsilon_are_not_counted_as_submodular_evidence() -> None:
    inside = [_summary_row(0.001)] * 10
    summary = summarize_interactions(inside, epsilon=0.01)
    assert summary["submodular_pairs"] == 10
    assert summary["resolved_pairs"] == 0
    assert summary["verdict"] == "UNRESOLVED / BELOW THE DECISION THRESHOLD"


def test_a_rendered_report_names_the_test_behind_every_count(tiny_utility) -> None:
    report = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
    )
    rendered = format_diagnostic_report(report)
    assert "HEADLINE - epsilon test (this is the Classification column):" in rendered
    assert "SECONDARY - strict sign test, epsilon ignored (not the classification):" in rendered
    assert "GATE 1, STATISTICAL - can the difference be told from zero:" in rendered
    assert "Classification (epsilon test)" in rendered
    assert "Ground-set ladders: weight [3, 4, 8, 16]" in rendered


def test_every_classification_carries_the_epsilon_it_was_compared_against(
    tiny_utility,
) -> None:
    tolerance = EpsilonPolicy.from_mapping(
        {
            WEIGHT_TO_WEIGHT: 0.03,
            KV_TO_KV: 0.004,
            WEIGHT_GIVEN_KV: 0.03,
            KV_GIVEN_WEIGHT: 0.004,
        },
        "measured second-order floor",
    )
    report = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=tolerance
    )
    for row in report["interactions"]:
        assert row["epsilon_used"] == tolerance.for_modality(row["modality"])
        assert row["epsilon_source"] == "measured second-order floor"
        assert row["classification"] == classify_difference(
            row["second_order_difference"], row["epsilon_used"]
        )
    assert report["epsilon_ppl"]["uniform"] is False
    assert report["epsilon_ppl"]["by_modality"][KV_TO_KV] == 0.004


def test_conditioning_moves_can_be_narrowed_to_an_adjacent_step() -> None:
    matrix = build_interaction_matrix(
        (0, 1),
        LADDERS,
        weight_upgrades=(TierUpgrade(WEIGHT, 3, 4),),
        kv_upgrades=(TierUpgrade(KV, 0.25, 0.50),),
        weight_conditioning=TierUpgrade(WEIGHT, 3, 4),
        kv_conditioning=TierUpgrade(KV, 0.25, 0.50),
    )
    labels = {
        move.label for spec in matrix for move in (spec.target, spec.conditioning)
    }
    assert labels == {"W:3->4", "KV:0.25->0.50"}
    assert all(spec.in_ground_set(LADDERS) for spec in matrix)


def test_the_floor_measures_the_spread_of_the_difference_not_of_a_delta(
    tiny_utility_wide,
) -> None:
    """The classified quantity is D, so the floor must be the spread of D.

    What is asserted here is the shape of the record and the arithmetic, not
    the value: the tiny model is random weights on random tokens.
    """
    from submokv.submodularity import second_order_noise_floor

    tiny_utility = tiny_utility_wide
    ladders = TierLadders.from_ground_set(tiny_utility.ground_set)
    specs = default_floor_specs(ladders, target_layer=1, conditioning_layer=0)
    assert {spec.modality for spec in specs} == set(
        (WEIGHT_TO_WEIGHT, KV_TO_KV, WEIGHT_GIVEN_KV, KV_GIVEN_WEIGHT)
    )

    floor = second_order_noise_floor(tiny_utility, specs, num_subsamples=5)
    assert set(floor["by_modality"]) == set(
        (WEIGHT_TO_WEIGHT, KV_TO_KV, WEIGHT_GIVEN_KV, KV_GIVEN_WEIGHT)
    )
    for row in floor["specs"]:
        assert len(row["second_order_differences"]) == 5
        assert len(row["corner_perplexities"]) == 5
        for corner, difference in zip(row["corner_perplexities"], row["second_order_differences"]):
            expected = (corner["ppl_s_a"] - corner["ppl_s_a_union_j"]) - (
                corner["ppl_s_b"] - corner["ppl_s_b_union_j"]
            )
            assert difference == pytest.approx(expected)
            # Every corner of one square is read from one draw; a mismatch here
            # would mean the pairing that makes D measurable was broken.
            assert corner["subsample"] in range(5)
        assert row["second_order_stdev"] == pytest.approx(
            __import__("statistics").stdev(row["second_order_differences"])
        )
    json.dumps(floor)


def test_the_floor_refuses_too_few_draws_to_be_a_spread(tiny_utility_wide) -> None:
    from submokv.submodularity import second_order_noise_floor

    tiny_utility = tiny_utility_wide
    ladders = TierLadders.from_ground_set(tiny_utility.ground_set)
    specs = default_floor_specs(ladders, 1, 0)
    with pytest.raises(ValueError, match="not a spread"):
        second_order_noise_floor(tiny_utility, specs, num_subsamples=4)


def test_the_floor_refuses_specs_off_the_ground_set_ladder(tiny_utility_wide) -> None:
    from submokv.submodularity import second_order_noise_floor

    tiny_utility = tiny_utility_wide

    off_ladder = (
        InteractionSpec(1, TierUpgrade(WEIGHT, 2, 4), 0, TierUpgrade(WEIGHT, 3, 4)),
    )
    with pytest.raises(OutsideGroundSetError, match="does not calibrate"):
        second_order_noise_floor(tiny_utility, off_ladder, num_subsamples=5)


def test_epsilon_is_built_per_modality_from_the_measured_floor() -> None:
    from submokv.submodularity import epsilon_from_floor

    floor = {
        "by_modality": {
            WEIGHT_TO_WEIGHT: {"second_order_stdev_pooled": 0.0400},
            KV_TO_KV: {"second_order_stdev_pooled": 0.0050},
            WEIGHT_GIVEN_KV: {"second_order_stdev_pooled": 0.0300},
            KV_GIVEN_WEIGHT: {"second_order_stdev_pooled": 0.0060},
        }
    }
    epsilon = epsilon_from_floor(floor)
    assert epsilon.for_modality(WEIGHT_TO_WEIGHT) == pytest.approx(0.04)
    assert epsilon.for_modality(KV_TO_KV) == pytest.approx(0.005)
    assert not epsilon.is_uniform()
    assert "measured second-order floor" in epsilon.source

    # A cell reported as the mean of k draws has standard error sigma/sqrt(k).
    averaged = epsilon_from_floor(floor, subsamples_per_cell=4)
    assert averaged.for_modality(WEIGHT_TO_WEIGHT) == pytest.approx(0.02)
    assert "sqrt(4)" in averaged.source


def test_epsilon_refuses_a_modality_the_floor_never_measured() -> None:
    from submokv.submodularity import epsilon_from_floor

    with pytest.raises(ValueError, match="must have its own measured floor"):
        epsilon_from_floor({"by_modality": {WEIGHT_TO_WEIGHT: {"second_order_stdev_pooled": 0.01}}})


def test_pooling_uses_within_spec_spread_not_the_spread_of_the_means() -> None:
    from submokv.submodularity import _pooled_stdev

    # Two specs with identical internal spread but very different means. The
    # pooled floor must see the spread, not the offset between them.
    pooled, degrees = _pooled_stdev([[1.0, 2.0, 3.0], [101.0, 102.0, 103.0]])
    assert pooled == pytest.approx(1.0)
    assert degrees == 4


def test_a_per_modality_epsilon_in_the_config_reaches_the_run(tmp_path) -> None:
    """A measured floor is four numbers, so the config path must carry four."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = _olmoe_config()
    config["submodularity"]["epsilon_ppl"] = {
        WEIGHT_TO_WEIGHT: 0.0321,
        KV_TO_KV: 0.0047,
        WEIGHT_GIVEN_KV: 0.0298,
        KV_GIVEN_WEIGHT: 0.0051,
    }
    config["submodularity"]["epsilon_source"] = "measured second-order floor"
    path = tmp_path / "olmoe_measured.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "submodularity_diagnostic.py"),
            "--config",
            str(path),
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["epsilon_ppl"]["uniform"] is False
    assert payload["epsilon_ppl"]["source"] == "measured second-order floor"
    assert payload["epsilon_ppl"]["by_modality"][KV_TO_KV] == 0.0047
    assert payload["epsilon_ppl"]["by_modality"][WEIGHT_TO_WEIGHT] == 0.0321


def test_a_run_with_no_configured_epsilon_is_refused(tmp_path) -> None:
    """Epsilon must come from a measured floor, so there is no silent default."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = _olmoe_config()
    config["submodularity"].pop("epsilon_ppl")
    path = tmp_path / "olmoe_no_epsilon.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "submodularity_diagnostic.py"),
            "--config",
            str(path),
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "measured second-order noise floor, not a guess" in completed.stderr


def test_the_binding_epsilon_is_the_standard_error_of_a_mean_of_k() -> None:
    """DECISION.md Amendment 1: epsilon is sigma2/sqrt(k), not sigma2.

    Task C reports each cell as the mean of k calibration draws, whose standard
    error is sigma2/sqrt(k). Comparing that mean against the single-draw spread
    under-resolves by sqrt(k).
    """
    from submokv.submodularity import epsilon_from_floor

    floor = {
        "by_modality": {
            WEIGHT_TO_WEIGHT: {"second_order_stdev_pooled": 0.0300},
            KV_TO_KV: {"second_order_stdev_pooled": 0.0090},
            WEIGHT_GIVEN_KV: {"second_order_stdev_pooled": 0.0300},
            KV_GIVEN_WEIGHT: {"second_order_stdev_pooled": 0.0090},
        }
    }
    binding = epsilon_from_floor(floor, subsamples_per_cell=3)
    assert binding.for_modality(WEIGHT_TO_WEIGHT) == pytest.approx(0.03 / 3**0.5)
    assert binding.for_modality(KV_TO_KV) == pytest.approx(0.009 / 3**0.5)
    # Strictly tighter than the single-draw floor, and by exactly sqrt(k).
    single = epsilon_from_floor(floor, subsamples_per_cell=1)
    ratio = single.for_modality(KV_TO_KV) / binding.for_modality(KV_TO_KV)
    assert ratio == pytest.approx(3**0.5)
    assert "sqrt(3)" in binding.source


def test_distinct_conditioning_moves_separate_the_two_cross_component_squares() -> None:
    """With conditioning defaulted to the target move, W|KV and KV|W are one square.

    The mixed second difference is symmetric in its two components, so the same
    weight move and KV move on one layer give the identical D whichever is
    called the target. Giving conditioning its own moves separates them.
    """
    def corners(spec):
        p = four_state_plans(spec, 16)
        return frozenset((p.s_a, p.s_a_union_j, p.s_b, p.s_b_union_j))

    def difference(spec, ppl):
        p = four_state_plans(spec, 16)
        return (ppl[p.s_a] - ppl[p.s_a_union_j]) - (ppl[p.s_b] - ppl[p.s_b_union_j])

    cross = {s.modality: s for s in default_floor_specs(LADDERS, 8, 4)}
    a, b = cross[WEIGHT_GIVEN_KV], cross[KV_GIVEN_WEIGHT]
    # The same four corners, with S_A+j and S_B swapped between the two specs.
    assert corners(a) == corners(b)
    # Which makes D literally the same number, whatever the corners evaluate to.
    ppl = {plan: 6.0 + 0.37 * index for index, plan in enumerate(sorted(corners(a), key=str))}
    assert difference(a, ppl) == pytest.approx(difference(b, ppl))

    cross = {
        s.modality: s
        for s in default_floor_specs(
            LADDERS,
            target_layer=8,
            conditioning_layer=4,
            weight_conditioning=TierUpgrade(WEIGHT, 3, 16),
            kv_conditioning=TierUpgrade(KV, 0.25, 1.0),
        )
    }
    assert corners(cross[WEIGHT_GIVEN_KV]) != corners(cross[KV_GIVEN_WEIGHT])


def test_floor_specs_can_be_narrowed_to_one_modality() -> None:
    specs = default_floor_specs(
        LADDERS,
        target_layer=8,
        conditioning_layer=4,
        weight_conditioning=TierUpgrade(WEIGHT, 3, 16),
        modalities=[WEIGHT_TO_WEIGHT],
    )
    assert len(specs) == 1
    assert specs[0].modality == WEIGHT_TO_WEIGHT
    assert specs[0].target.label == "W:3->4"
    assert specs[0].conditioning.label == "W:3->16"
    with pytest.raises(ValueError, match="unknown modality"):
        default_floor_specs(LADDERS, 8, 4, modalities=["nonsense"])


def _floor_stub(modality: str, sigma: float, low: float, high: float) -> dict:
    return {
        "by_modality": {
            modality: {
                "second_order_stdev_pooled": sigma,
                "second_order_stdev_pooled_ci": {"confidence": 0.95, "low": low, "high": high},
                "spec_ids": [f"id.{modality}"],
            }
        }
    }


def test_the_conditioning_check_applies_the_amendment_2c_rule() -> None:
    """Inside the baseline interval the floor stands; outside it Task B is unfinished."""
    from submokv.submodularity import compare_conditioning_floor

    baseline = _floor_stub(WEIGHT_TO_WEIGHT, 0.00455, 0.00284, 0.01117)

    inside = compare_conditioning_floor(baseline, _floor_stub(WEIGHT_TO_WEIGHT, 0.0060, 0, 1))
    assert inside["floor_stands"] is True
    assert inside["comparisons"][0]["inside_baseline_interval"] is True
    assert inside["comparisons"][0]["ratio_to_baseline"] == pytest.approx(0.0060 / 0.00455)
    assert "FLOOR STANDS" in inside["verdict"]

    above = compare_conditioning_floor(baseline, _floor_stub(WEIGHT_TO_WEIGHT, 0.0150, 0, 1))
    assert above["floor_stands"] is False
    assert above["modalities_outside_interval"] == [WEIGHT_TO_WEIGHT]
    assert "re-derive" in above["verdict"]

    below = compare_conditioning_floor(baseline, _floor_stub(WEIGHT_TO_WEIGHT, 0.0020, 0, 1))
    assert below["floor_stands"] is False

    # The interval is closed: a checked value exactly on a bound is inside.
    for bound in (0.00284, 0.01117):
        edge = compare_conditioning_floor(baseline, _floor_stub(WEIGHT_TO_WEIGHT, bound, 0, 1))
        assert edge["floor_stands"] is True


def test_the_conditioning_check_refuses_a_modality_the_baseline_never_measured() -> None:
    from submokv.submodularity import compare_conditioning_floor

    baseline = _floor_stub(WEIGHT_TO_WEIGHT, 0.00455, 0.00284, 0.01117)
    with pytest.raises(ValueError, match="nothing to compare"):
        compare_conditioning_floor(baseline, _floor_stub(KV_TO_KV, 0.002, 0, 1))


def test_the_conditioning_check_renders_the_rule_and_the_outcome() -> None:
    from submokv.submodularity import compare_conditioning_floor, format_conditioning_check

    baseline = _floor_stub(WEIGHT_TO_WEIGHT, 0.00455, 0.00284, 0.01117)
    rendered = format_conditioning_check(
        compare_conditioning_floor(baseline, _floor_stub(WEIGHT_TO_WEIGHT, 0.0060, 0, 1))
    )
    assert "Amendment 2 C" in rendered
    assert "Rule fixed before either number existed" in rendered
    assert "inside" in rendered


def test_a_context_only_comparison_selects_no_verdict() -> None:
    """A comparison no rule was declared for in advance must not decide anything."""
    from submokv.submodularity import compare_conditioning_floor, format_conditioning_check

    baseline = _floor_stub(WEIGHT_TO_WEIGHT, 0.00455, 0.00284, 0.01117)
    check = _floor_stub(WEIGHT_TO_WEIGHT, 0.0150, 0, 1)

    context = compare_conditioning_floor(baseline, check, binding=False, question="layer")
    assert context["binding"] is False
    assert context["floor_stands"] is None
    assert context["consistent"] is False
    assert context["decided_before_measurement"] is False
    assert "CONTEXT" in context["verdict"]
    assert "no branch turns on it" in context["rule"]

    rendered = format_conditioning_check(context)
    assert "CONTEXT ONLY" in rendered
    assert "Amendment 2 C" not in rendered

    # The same numbers, declared binding, do decide.
    decided = compare_conditioning_floor(baseline, check, binding=True)
    assert decided["floor_stands"] is False
    assert "Amendment 2 C" in format_conditioning_check(decided)


REAL_FLOOR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "second_order_floor__20260821T200032Z__601002ea.json"
)


def _real_floor() -> dict:
    return json.loads(REAL_FLOOR.read_text(encoding="utf-8"))["payload"]["second_order_floor"]


def test_the_band_and_the_effect_gate_come_from_the_measured_floor() -> None:
    """Both tolerances are derived from one record, so they cannot drift apart."""
    from submokv.submodularity import EffectFloor, EpsilonBand

    floor = _real_floor()
    band = EpsilonBand.from_floor_record(floor, subsamples_per_cell=3)
    # Matches what the floor run itself printed, to the digit.
    assert band.point.for_modality(WEIGHT_TO_WEIGHT) == pytest.approx(0.0026298, abs=1e-6)
    assert band.point.for_modality(KV_TO_KV) == pytest.approx(0.0013430, abs=1e-6)
    # The bounds bracket the point estimate, and all three are sigma2/sqrt(3).
    for modality in (WEIGHT_TO_WEIGHT, KV_TO_KV, WEIGHT_GIVEN_KV, KV_GIVEN_WEIGHT):
        assert (
            band.low.for_modality(modality)
            < band.point.for_modality(modality)
            < band.high.for_modality(modality)
        )
    assert band.as_dict()["binding_reading"] == "point"

    effect = EffectFloor.from_floor_record(floor, phi=0.10, multiple=3.0)
    assert effect.for_kind(WEIGHT) == pytest.approx(0.03460, abs=1e-4)
    assert effect.for_kind(KV) == pytest.approx(0.00781, abs=1e-4)
    assert effect.phi == 0.10


def test_the_effect_gate_reproduces_the_task_b_check_scores() -> None:
    """The gates must score the two Task B check squares as they were reported."""
    from submokv.submodularity import EffectFloor

    effect = EffectFloor.from_floor_record(_real_floor(), phi=0.10, multiple=3.0)
    # check 1: L8|L4 at matrix conditioning. Resolved, but 4.9% of its gain.
    one = effect.evaluate(WEIGHT, magnitude=0.06176, difference=0.00305)
    assert one["gate_effect_e1"] is True
    assert one["gate_effect_e2"] is False
    assert one["relative_interaction"] == pytest.approx(0.0494, abs=1e-3)
    assert one["effect_failure_reasons"] == ["effect_e2_interaction_immaterial"]
    # check 2: L14|L10. Six of six draws negative, and still immaterial at 7.1%.
    two = effect.evaluate(WEIGHT, magnitude=0.12129, difference=-0.00867)
    assert two["gate_effect_e1"] is True
    assert two["gate_effect_e2"] is False
    assert two["relative_interaction"] == pytest.approx(0.0715, abs=1e-3)
    # A target that buys nothing fails the other clause, and says so distinctly.
    dead = effect.evaluate(WEIGHT, magnitude=0.01, difference=0.005)
    assert dead["gate_effect_e1"] is False
    assert "effect_e1_target_buys_nothing" in dead["effect_failure_reasons"]


def test_a_subsample_count_that_does_not_match_k_fails_loudly(tiny_utility_wide) -> None:
    """DECISION.md Amendment 1: a mean of k draws is only comparable to sigma2/sqrt(k)."""
    matrix = build_interaction_matrix((0,), LADDERS)
    with pytest.raises(ValueError, match="not comparable"):
        run_submodularity_diagnostic(
            tiny_utility_wide,
            matrix,
            epsilon=0.01,
            subsamples=(0, 1),
            expected_subsamples_per_cell=3,
        )


def test_a_cell_is_a_mean_over_its_draws_with_its_spread(tiny_utility_wide) -> None:
    report = run_submodularity_diagnostic(
        tiny_utility_wide, build_interaction_matrix((0,), LADDERS), epsilon=0.01,
        subsamples=(0, 1, 2),
    )
    assert report["calibration"]["subsamples"] == [0, 1, 2]
    assert report["calibration"]["subsamples_per_cell"] == 3
    for row in report["interactions"]:
        assert len(row["per_subsample"]) == 3
        assert "second_order_stdev" in row and "second_order_stderr" in row
        assert 0 <= row["sign_agreement"] <= 3
    json.dumps(report)


def test_the_branch_turns_on_both_gates_not_on_resolution_alone(tiny_utility_wide) -> None:
    """A statistically resolved but immaterial cell must not count toward the branch."""
    from submokv.submodularity import EffectFloor

    # A floor that E1 passes trivially and E2 cannot: phi of 1.0 demands the
    # interaction be as large as the gain it sits on.
    strict = EffectFloor.from_first_order_noise({WEIGHT: 0.0, KV: 0.0}, phi=1.0)
    report = run_submodularity_diagnostic(
        tiny_utility_wide,
        build_interaction_matrix((0, 1), LADDERS),
        epsilon=0.0,
        effect_floor=strict,
        subsamples=(0, 1, 2),
    )
    gates = report["summary"]["gates"]
    assert gates["effect_gate_in_force"] is True
    assert gates["counts_toward_branch"] == 0
    assert gates["counts_toward_branch_rate"] == 0.0
    assert report["summary"]["verdict"] == "UNRESOLVED / BELOW THE DECISION THRESHOLD"
    # Resolution alone would have said otherwise.
    assert report["summary"]["resolution"]["resolved_pairs"] > 0
    assert "effect_e2_interaction_immaterial" in gates["failure_reasons"]


def test_the_band_reports_three_readings_and_flags_a_flip(tiny_utility_wide) -> None:
    from submokv.submodularity import EpsilonBand

    band = EpsilonBand.from_floor_record(_real_floor(), subsamples_per_cell=3)
    report = run_submodularity_diagnostic(
        tiny_utility_wide,
        build_interaction_matrix((0, 1), LADDERS),
        epsilon=band,
        subsamples=(0, 1, 2),
    )
    reported = report["summary"]["band"]
    assert set(reported["readings"]) == {"point", "sigma2_low", "sigma2_high"}
    assert reported["binding_reading"] == "point"
    # A wider epsilon can never resolve more cells than a narrower one.
    assert (
        reported["readings"]["sigma2_high"]["resolved_statistical"]
        <= reported["readings"]["point"]["resolved_statistical"]
        <= reported["readings"]["sigma2_low"]["resolved_statistical"]
    )
    assert isinstance(reported["verdict_stable_across_band"], bool)
    if not reported["verdict_stable_across_band"]:
        assert "FLIPS" in reported["note"]


def test_the_report_states_the_move_asymmetry_and_the_layer_coverage(tiny_utility_wide) -> None:
    from submokv.submodularity import EffectFloor

    effect = EffectFloor.from_first_order_noise({WEIGHT: 0.001, KV: 0.001}, phi=0.10)
    report = run_submodularity_diagnostic(
        tiny_utility_wide,
        build_interaction_matrix(
            (0, 1), LADDERS, weight_conditioning=TierUpgrade(WEIGHT, 3, 16)
        ),
        epsilon=0.01,
        effect_floor=effect,
        subsamples=(0, 1, 2),
    )
    assert report["layer_coverage"]["layers_sampled"] == 2
    assert report["layer_coverage"]["layers_in_model"] == 3
    rendered = format_diagnostic_report(report)
    assert "TARGET moves are ADJACENT" in rendered
    assert "GATE 2, EFFECT - would an allocator act on it:" in rendered
    assert "|D|/m distribution vs phi" in rendered
    assert "Layer coverage: 2 of 3 layers sampled" in rendered


def test_shards_merge_with_the_band_and_the_effect_gate_intact(tiny_utility_wide) -> None:
    """A merged report must be classified against the same two tolerances as its shards."""
    from submokv.submodularity import EffectFloor, EpsilonBand

    band = EpsilonBand.from_floor_record(_real_floor(), subsamples_per_cell=3)
    effect = EffectFloor.from_first_order_noise({WEIGHT: 0.001, KV: 0.001}, phi=0.10)
    full = run_submodularity_diagnostic(
        tiny_utility_wide,
        build_interaction_matrix((0, 1), LADDERS),
        epsilon=band,
        effect_floor=effect,
        subsamples=(0, 1, 2),
    )
    merged = merge_submodularity_reports(_split_report(full, 2))
    assert merged["epsilon_band"]["binding_reading"] == "point"
    assert merged["effect_gate"]["phi"] == 0.10
    assert merged["summary"]["band"] is not None
    assert merged["summary"]["gates"]["effect_gate_in_force"] is True
    # The merged summary reproduces the whole-matrix one exactly.
    for key in ("branch_rate", "resolved_pairs", "verdict"):
        assert merged["summary"][key] == full["summary"][key]


def test_shards_that_disagree_on_a_tolerance_refuse_to_merge(tiny_utility_wide) -> None:
    from submokv.submodularity import EffectFloor

    effect = EffectFloor.from_first_order_noise({WEIGHT: 0.001, KV: 0.001}, phi=0.10)
    full = run_submodularity_diagnostic(
        tiny_utility_wide,
        build_interaction_matrix((0,), LADDERS),
        epsilon=0.01,
        effect_floor=effect,
        subsamples=(0, 1, 2),
    )
    shards = _split_report(full, 2)
    shards[1]["effect_gate"] = {**shards[1]["effect_gate"], "phi": 0.25}
    with pytest.raises(ValueError, match="disagree on effect_gate"):
        merge_submodularity_reports(shards)
