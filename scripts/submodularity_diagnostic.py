#!/usr/bin/env python3
"""Run the standalone Sub-MoKV pairwise submodularity diagnostic.

The default OLMoE run evaluates 32 interactions over four representative
layers, 64 fixed WikiText validation windows of 2,048 tokens, 2->4 bit target
upgrades, and 25%->50% KV target upgrades.  Use ``--all-upgrades`` to expand
the same layer grid over every adjacent tier transition.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Deterministic CUDA matmuls require this variable to exist before torch is
# imported.  Respect an explicit setting made by the caller.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Make ``python scripts/submodularity_diagnostic.py`` work without installing
# the package or setting PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPOSITORY_ROOT / ".hf-cache"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from submokv.cli import load_config  # noqa: E402
from submokv.ground_set import GroundSet  # noqa: E402
from submokv.records import environment, git_commit  # noqa: E402
from submokv.submodularity import (  # noqa: E402
    KV,
    WEIGHT,
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


def _configured_upgrades(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get("submodularity", {}).get(key, fallback)
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


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    calibration = updated.setdefault("calibration", {})
    kv = updated.setdefault("kv", {})
    protocol = updated.setdefault("protocol", {})
    runtime = updated.setdefault("runtime", {})
    retention = updated.setdefault("retention", {})

    old_length = int(calibration.get("sequence_length", kv.get("context_length", 2048)))
    old_prefill = int(protocol.get("prefill_tokens", max(1, 3 * old_length // 4)))
    new_length = int(args.sequence_length)
    calibration["sequence_length"] = new_length
    calibration["cheap_sequences"] = int(args.sequences)
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
    if args.sequences < batch_size:
        raise ValueError(
            f"sequences ({args.sequences}) must be at least the KV batch size ({batch_size})"
        )
    if args.sequences % batch_size:
        raise ValueError(
            f"sequences ({args.sequences}) must be divisible by the KV batch size "
            f"({batch_size}); the evaluator intentionally refuses a differently sized final batch"
        )
    return updated


def _resolve_snapshot(
    model_name: str,
    model_path: Path | None,
    local_files_only: bool,
) -> str:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        return str(model_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, local_files_only=local_files_only)


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

    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--calibration-split", type=str, default="validation")
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
        "--all-upgrades",
        action="store_true",
        help="test every adjacent weight and KV ladder move (112 rows for four layers)",
    )
    parser.add_argument("--epsilon", type=float, default=None, help="PPL tolerance")
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

    if args.sequences <= 0:
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

        if args.all_upgrades:
            weight_upgrades = adjacent_upgrades(WEIGHT)
            kv_upgrades = adjacent_upgrades(KV)
        else:
            weight_text = args.weight_upgrades or _configured_upgrades(
                config, "weight_upgrades", "2:4"
            )
            kv_text = args.kv_upgrades or _configured_upgrades(
                config, "kv_upgrades", "0.25:0.50"
            )
            weight_upgrades = parse_upgrades(weight_text, WEIGHT)
            kv_upgrades = parse_upgrades(kv_text, KV)
        interactions = build_interaction_matrix(layers, weight_upgrades, kv_upgrades)
        shard_interactions = tuple(
            spec for index, spec in enumerate(interactions) if index % args.num_shards == args.shard
        )
        epsilon = float(
            args.epsilon
            if args.epsilon is not None
            else config.get("submodularity", {}).get("epsilon_ppl", 0.01)
        )
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": ground.model.name,
                    "num_hidden_layers": ground.model.num_hidden_layers,
                    "test_layers": list(layers),
                    "epsilon_ppl": epsilon,
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
                    "interactions": matrix_as_dict(shard_interactions),
                },
                indent=2,
            )
        )
        return 0

    try:
        snapshot = _resolve_snapshot(
            ground.model.name, args.model_path, args.local_files_only
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

    def progress(index: int, total: int, spec: Any) -> None:
        if not args.quiet:
            print(f"[{index:>3}/{total}] {spec.label}", file=sys.stderr, flush=True)

    report = run_submodularity_diagnostic(
        utility,
        shard_interactions,
        epsilon=epsilon,
        subsample=args.subsample,
        use_cache=not args.no_eval_cache,
        progress=progress,
    )
    report["provenance"] = {
        "git_commit": git_commit(REPOSITORY_ROOT),
        "environment": environment(),
        "config_path": str(args.config.resolve()),
        "effective_config": config,
        "model_snapshot": snapshot,
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
