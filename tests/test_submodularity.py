"""Tests for the four-state Sub-MoKV submodularity diagnostic."""

from __future__ import annotations

import json
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
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
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
    # The calibration settings come from the config, not from a flag default;
    # a floor and the run it calibrates must read the same windows.
    assert payload["calibration"]["split"] == "train"
    assert payload["calibration"]["sequence_length"] == 4096
    assert payload["calibration"]["sequences"] == 64
    assert payload["num_interactions"] == 32
    assert payload["tier_ladders"]["weight_tiers"] == [3, 4, 8, 16]

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
    assert summary["verdict"] == "UNRESOLVED / INSIDE THE NOISE FLOOR"


def test_a_rendered_report_names_the_test_behind_every_count(tiny_utility) -> None:
    report = run_submodularity_diagnostic(
        tiny_utility, build_interaction_matrix((0,), LADDERS), epsilon=0.01
    )
    rendered = format_diagnostic_report(report)
    assert "HEADLINE - epsilon test (this is the Classification column):" in rendered
    assert "SECONDARY - strict sign test, epsilon ignored (not the classification):" in rendered
    assert "RESOLUTION - how many cells had enough signal to classify at all:" in rendered
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
