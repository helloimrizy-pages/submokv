#!/usr/bin/env python3
"""Run the standalone Sub-MoKV pairwise submodularity diagnostic.

The tier ladders come from the ground set the config declares, so a probe of a
tier the allocator cannot buy is refused rather than silently reported.  Target
and conditioning transitions default to that ladder; ``--all-upgrades`` expands
the layer grid over every adjacent move on it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Deterministic CUDA matmuls require this variable to exist before torch is
# imported.  Respect an explicit setting made by the caller.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Make ``python scripts/submodularity_diagnostic.py`` work without installing
# the package or setting PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPOSITORY_ROOT / ".hf-cache"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from submokv.cli import load_config, resolve_model_path  # noqa: E402
from submokv.ground_set import GroundSet  # noqa: E402
from submokv.records import environment, git_commit  # noqa: E402
from submokv.submodularity import (  # noqa: E402
    KV,
    WEIGHT,
    EffectFloor,
    EpsilonBand,
    EpsilonPolicy,
    OutsideGroundSetError,
    TierLadders,
    adjacent_upgrades,
    build_interaction_matrix,
    format_diagnostic_report,
    matrix_as_dict,
    merge_submodularity_reports,
    parse_upgrades,
    run_submodularity_diagnostic,
    validate_test_layers,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "olmoe.yaml"


def _parse_layers(text: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def _configured_upgrades(config: dict[str, Any], key: str) -> str | None:
    """Return the configured transitions as a FROM:TO string, or None if unset.

    None means "use the ground set's ladder", which is the only default; there
    is deliberately no hardcoded tier list here.
    """
    value = config.get("submodularity", {}).get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"submodularity.{key} must contain [from, to] pairs")
            parts.append(f"{pair[0]}:{pair[1]}")
        return ",".join(parts)
    raise ValueError(f"submodularity.{key} must be a transition string or list of pairs")


def _load_floor(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a measured floor payload and the record that carried it."""
    record = json.loads(path.read_text(encoding="utf-8"))
    floor = record.get("payload", {}).get("second_order_floor", record)
    if "by_modality" not in floor or "specs" not in floor:
        raise ValueError(f"{path} does not look like a second-order floor record")
    return floor, record


def _resolve_tolerances(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[EpsilonBand, EffectFloor, dict[str, Any]]:
    """Derive both tolerances from the measured floor, never from a transcription.

    The band and the effect gate come from one record, so the epsilon a cell is
    compared against and the floor it was measured from cannot drift apart.
    """
    settings = config.get("submodularity", {})
    k = settings.get("epsilon_cell_subsamples")
    if k is None:
        raise ValueError(
            "submodularity.epsilon_cell_subsamples is not set; epsilon is sigma2/sqrt(k) and "
            "the run must declare the k it was built for"
        )
    phi = settings.get("effect_relative_phi")
    if phi is None:
        raise ValueError(
            "submodularity.effect_relative_phi is not set; the effect gate of DECISION.md "
            "Amendment 2 B needs it and it must not be chosen after a classification exists"
        )
    floor, record = _load_floor(args.floor)
    band = EpsilonBand.from_floor_record(floor, int(k))
    effect = EffectFloor.from_floor_record(
        floor, float(phi), float(settings.get("effect_noise_multiple", 3.0))
    )
    provenance = {
        "floor_record": str(args.floor.resolve()),
        "floor_git_commit": record.get("git_commit"),
        "floor_created_at": floor.get("created_at"),
        "floor_calibration": floor.get("calibration"),
        "epsilon_cell_subsamples": int(k),
        "effect_relative_phi": float(phi),
    }
    return band, effect, provenance


def _resolve_epsilon(args: argparse.Namespace, config: dict[str, Any]) -> EpsilonPolicy:
    """Return the classification tolerance, per modality where one is configured.

    ``submodularity.epsilon_ppl`` accepts either a bare number or a mapping from
    modality to tolerance. The mapping is what a measured second-order noise
    floor produces, because the weight and KV scales differ by roughly a factor
    of seven and one constant cannot serve both.
    """
    if args.epsilon is not None:
        return EpsilonPolicy.uniform(float(args.epsilon), "command line --epsilon")
    configured = config.get("submodularity", {}).get("epsilon_ppl")
    if configured is None:
        raise ValueError(
            "no epsilon is configured; set submodularity.epsilon_ppl in the config or pass "
            "--epsilon. It must come from a measured second-order noise floor, not a guess."
        )
    if isinstance(configured, Mapping):
        source = str(
            config.get("submodularity", {}).get("epsilon_source", "config mapping")
        )
        return EpsilonPolicy.from_mapping(configured, source)
    source = str(
        config.get("submodularity", {}).get("epsilon_source", "config constant")
    )
    return EpsilonPolicy.uniform(float(configured), source)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    calibration = updated.setdefault("calibration", {})
    kv = updated.setdefault("kv", {})
    protocol = updated.setdefault("protocol", {})
    runtime = updated.setdefault("runtime", {})
    retention = updated.setdefault("retention", {})

    # Every calibration setting defaults to the config. A run whose noise floor
    # was measured at one sequence length and split cannot be classified against
    # a floor measured at another, and the previous defaults silently moved a
    # 4096-token train-split experiment onto 2048-token validation windows.
    old_length = int(calibration.get("sequence_length", kv.get("context_length", 2048)))
    old_prefill = int(protocol.get("prefill_tokens", max(1, 3 * old_length // 4)))
    new_length = old_length if args.sequence_length is None else int(args.sequence_length)
    calibration["sequence_length"] = new_length
    if args.sequences is not None:
        calibration["cheap_sequences"] = int(args.sequences)
    if args.calibration_split is not None:
        calibration["calibration_split"] = args.calibration_split
    kv["context_length"] = new_length

    if args.batch_size is not None:
        kv["batch_size"] = int(args.batch_size)
    if args.prefill_tokens is None:
        scaled = round(old_prefill * new_length / old_length)
        protocol["prefill_tokens"] = min(max(0, scaled), new_length - 1)
    else:
        protocol["prefill_tokens"] = int(args.prefill_tokens)
    if args.chunk_size is not None:
        protocol["chunk_size"] = int(args.chunk_size)
    if args.policy is not None:
        retention["policy"] = args.policy
    if args.master_store is not None:
        runtime["master_store"] = args.master_store
    if args.dtype is not None:
        runtime["dtype"] = args.dtype

    if not 0 <= protocol["prefill_tokens"] < new_length:
        raise ValueError(
            f"prefill_tokens must be in [0, {new_length - 1}], got {protocol['prefill_tokens']}"
        )
    batch_size = int(kv.get("batch_size", 1))
    sequences = int(calibration["cheap_sequences"])
    if sequences < batch_size:
        raise ValueError(
            f"sequences ({sequences}) must be at least the KV batch size ({batch_size})"
        )
    if sequences % batch_size:
        raise ValueError(
            f"sequences ({sequences}) must be divisible by the KV batch size "
            f"({batch_size}); the evaluator intentionally refuses a differently sized final batch"
        )
    return updated


def _resolve_snapshot(
    config: dict[str, Any],
    model_path: Path | None,
    local_files_only: bool,
) -> str:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        return str(model_path)
    if local_files_only:
        # snapshot_download regards a deliberately filtered snapshot as
        # incomplete when non-runtime repository files (README, logo, git
        # attributes) were omitted. resolve_model_path falls back to the
        # concrete cached revision, which from_pretrained and the checkpoint
        # master store can load directly. verify_device exercises that exact
        # path before an experiment is allowed to run.
        return resolve_model_path(config, None)
    from huggingface_hub import snapshot_download

    return snapshot_download(config["model"]["name"])


def _default_output_path(results_dir: Path, suffix: str = "") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"_{suffix}" if suffix else ""
    return results_dir / f"submodularity{label}__{stamp}.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--master-store", choices=("checkpoint", "memory"), default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="fail instead of downloading a missing Hugging Face snapshot",
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="override calibration.cheap_sequences; defaults to the config",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=(2048, 4096),
        default=None,
        help="override calibration.sequence_length; defaults to the config",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default=None,
        help="override calibration.calibration_split; defaults to the config",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefill-tokens", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--policy", choices=("recency_sink", "attention_score"), default=None)

    parser.add_argument(
        "--layers",
        type=_parse_layers,
        default=None,
        help="zero-based representative layer indices, comma separated",
    )
    parser.add_argument(
        "--weight-upgrades",
        type=str,
        default=None,
        help="comma-separated FROM:TO bit-width probes, e.g. 2:4,3:8",
    )
    parser.add_argument(
        "--kv-upgrades",
        type=str,
        default=None,
        help="comma-separated FROM:TO retention probes, e.g. 0.25:0.5,0.5:1",
    )
    parser.add_argument(
        "--weight-conditioning",
        type=str,
        default=None,
        help="single FROM:TO weight move used as the conditioning component, e.g. 3:4",
    )
    parser.add_argument(
        "--kv-conditioning",
        type=str,
        default=None,
        help="single FROM:TO retention move used as the conditioning component",
    )
    parser.add_argument(
        "--all-upgrades",
        action="store_true",
        help="test every adjacent move on the ground set's weight and KV ladders",
    )
    parser.add_argument(
        "--allow-outside-ground-set",
        action="store_true",
        help=(
            "record probes of tiers the allocator cannot buy instead of refusing them; "
            "affected rows are labelled in_ground_set: false"
        ),
    )
    parser.add_argument("--epsilon", type=float, default=None, help="PPL tolerance")
    parser.add_argument(
        "--floor",
        type=Path,
        default=None,
        help=(
            "a measured second-order floor record; both tolerances are derived from it, "
            "which is what a classifying run requires"
        ),
    )
    parser.add_argument(
        "--subsamples",
        type=str,
        default=None,
        help=(
            "comma-separated calibration subsample indices, one cell per set; the count must "
            "equal submodularity.epsilon_cell_subsamples"
        ),
    )
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0, help="zero-based interaction shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-eval-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY_ROOT / "cache")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--merge-shards",
        type=Path,
        nargs="+",
        default=None,
        metavar="JSON",
        help="merge completed shard JSON files without loading a model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the interaction matrix without loading a model or dataset",
    )
    parser.add_argument("--quiet", action="store_true", help="hide per-interaction progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merge_shards:
        if args.dry_run:
            parser.error("--merge-shards cannot be combined with --dry-run")
        try:
            shard_reports = [
                json.loads(path.read_text(encoding="utf-8")) for path in args.merge_shards
            ]
            report = merge_submodularity_reports(shard_reports)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        report["provenance"] = {
            "git_commit": git_commit(REPOSITORY_ROOT),
            "environment": environment(),
            "merged_shard_paths": [str(path.resolve()) for path in args.merge_shards],
            "shard_provenance": [item.get("provenance", {}) for item in shard_reports],
            "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
        }
        output = args.output or _default_output_path(args.results_dir, "merged")
        _write_json(output, report)
        print(format_diagnostic_report(report))
        print(f"JSON log: {output.resolve()}")
        return 0

    if args.sequences is not None and args.sequences <= 0:
        parser.error("--sequences must be positive")
    if args.subsample < 0:
        parser.error("--subsample must be non-negative")
    if args.epsilon is not None and args.epsilon < 0:
        parser.error("--epsilon must be non-negative")
    if args.all_upgrades and (args.weight_upgrades or args.kv_upgrades):
        parser.error("--all-upgrades cannot be combined with explicit upgrade lists")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards - 1}]")

    try:
        original_config = load_config(args.config)
        config = _apply_overrides(original_config, args)
        ground = GroundSet.from_config(config)

        configured_layers = config.get("submodularity", {}).get("test_layers")
        if args.layers is not None:
            requested_layers = args.layers
        elif configured_layers is not None:
            requested_layers = tuple(int(layer) for layer in configured_layers)
        else:
            requested_layers = (2, 10, 18, 26)
        layers = validate_test_layers(requested_layers, ground.model.num_hidden_layers)

        # The ground set is the single source of truth for which tiers exist.
        ladders = TierLadders.from_ground_set(ground)

        if args.all_upgrades:
            weight_upgrades = adjacent_upgrades(WEIGHT, ladders)
            kv_upgrades = adjacent_upgrades(KV, ladders)
        else:
            weight_text = args.weight_upgrades or _configured_upgrades(
                config, "weight_upgrades"
            )
            kv_text = args.kv_upgrades or _configured_upgrades(config, "kv_upgrades")
            weight_upgrades = (
                parse_upgrades(weight_text, WEIGHT) if weight_text else None
            )
            kv_upgrades = parse_upgrades(kv_text, KV) if kv_text else None

        conditioning_text = args.weight_conditioning or _configured_upgrades(
            config, "weight_conditioning"
        )
        kv_conditioning_text = args.kv_conditioning or _configured_upgrades(
            config, "kv_conditioning"
        )
        weight_conditioning = (
            parse_upgrades(conditioning_text, WEIGHT)[0] if conditioning_text else None
        )
        kv_conditioning = (
            parse_upgrades(kv_conditioning_text, KV)[0] if kv_conditioning_text else None
        )

        interactions = build_interaction_matrix(
            layers,
            ladders,
            weight_upgrades,
            kv_upgrades,
            weight_conditioning=weight_conditioning,
            kv_conditioning=kv_conditioning,
        )
        offenders = sorted(
            {
                move.label
                for spec in interactions
                for move in (spec.target, spec.conditioning)
                if not ladders.contains(move)
            }
        )
        if offenders and not args.allow_outside_ground_set:
            raise OutsideGroundSetError(
                f"requested move(s) {', '.join(offenders)} are outside the ground set's "
                f"ladders weight={list(ladders.weight_tiers)} kv={list(ladders.kv_tiers)}. "
                "Fix configs/*.yaml or the --weight-upgrades/--kv-upgrades flags, or pass "
                "--allow-outside-ground-set to record them labelled in_ground_set: false."
            )
        shard_interactions = tuple(
            spec for index, spec in enumerate(interactions) if index % args.num_shards == args.shard
        )
        if args.subsamples is not None:
            draws = tuple(int(part) for part in args.subsamples.split(",") if part.strip())
        else:
            draws = (int(args.subsample),)
        band: EpsilonBand | None = None
        effect: EffectFloor | None = None
        tolerance_provenance: dict[str, Any] = {}
        if args.floor is not None:
            band, effect, tolerance_provenance = _resolve_tolerances(args, config)
            epsilon = band
            expected_k = tolerance_provenance["epsilon_cell_subsamples"]
            # Checked here rather than inside the runner so a config mismatch
            # costs a second instead of a 13 GiB model load.
            if len(draws) != expected_k:
                raise ValueError(
                    f"the tolerance was built for k={expected_k} subsamples per cell but this "
                    f"run requests {len(draws)} ({list(draws)}). Epsilon is sigma2/sqrt(k), so "
                    "the two are not comparable. DECISION.md Amendment 1 requires this to "
                    "fail rather than classify against the wrong tolerance."
                )
        else:
            epsilon = _resolve_epsilon(args, config)
            expected_k = None
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": ground.model.name,
                    "num_hidden_layers": ground.model.num_hidden_layers,
                    "test_layers": list(layers),
                    "tier_ladders": ladders.describe(),
                    "subsamples": list(draws),
                    "epsilon_ppl": (
                        band.as_dict() if band is not None else epsilon.as_dict()
                    ),
                    "effect_gate": effect.as_dict() if effect is not None else None,
                    "calibration": {
                        "dataset": config["calibration"]["dataset"],
                        "subset": config["calibration"]["subset"],
                        "split": config["calibration"]["calibration_split"],
                        "sequences": config["calibration"]["cheap_sequences"],
                        "sequence_length": config["calibration"]["sequence_length"],
                        "batch_size": config["kv"]["batch_size"],
                        "prefill_tokens": config["protocol"]["prefill_tokens"],
                    },
                    "num_interactions": len(interactions),
                    "shard": args.shard,
                    "num_shards": args.num_shards,
                    "num_shard_interactions": len(shard_interactions),
                    "interactions": matrix_as_dict(shard_interactions, ladders),
                },
                indent=2,
            )
        )
        return 0

    try:
        snapshot = _resolve_snapshot(
            config, args.model_path, args.local_files_only
        )
        from submokv.utility import build_utility

        _, utility = build_utility(
            config,
            model_path=snapshot,
            device=args.device,
            cache_dir=args.cache_dir,
            shard=args.shard,
            num_shards=args.num_shards,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    def progress(index: int, total: int, plan: Any, subsample: int) -> None:
        if not args.quiet:
            state = plan.compact_dict()
            print(
                f"[state {index:>3}/{total}] subsample {subsample} | "
                f"w={state['weight_bits']} kv={state['kv_retention']}",
                file=sys.stderr,
                flush=True,
            )

    try:
        report = run_submodularity_diagnostic(
            utility,
            shard_interactions,
            epsilon=epsilon,
            effect_floor=effect,
            subsamples=draws,
            expected_subsamples_per_cell=expected_k,
            use_cache=not args.no_eval_cache,
            allow_outside_ground_set=args.allow_outside_ground_set,
            progress=progress,
        )
    except (OutsideGroundSetError, ValueError) as error:
        parser.error(str(error))
    report["provenance"] = {
        "git_commit": git_commit(REPOSITORY_ROOT),
        "environment": environment(),
        "config_path": str(args.config.resolve()),
        "effective_config": config,
        "model_snapshot": snapshot,
        "tolerances": tolerance_provenance,
        "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
    }
    report["test_layers"] = list(layers)
    report["execution"].update(
        {
            "shard": args.shard,
            "num_shards": args.num_shards,
            "full_matrix_interactions": len(interactions),
            "full_matrix_order": [spec.interaction_id for spec in interactions],
        }
    )

    suffix = f"s{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    output = args.output or _default_output_path(args.results_dir, suffix)
    _write_json(output, report)
    print(format_diagnostic_report(report))
    print(f"JSON log: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
