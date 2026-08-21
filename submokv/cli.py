"""Command line entry points for Sub-MoKV."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .ground_set import BudgetInfeasibleError, GroundSet, UnitKind
from .memory import format_bytes, reference_footprint, total_params
from .records import record

DEFAULT_CONFIG = Path("configs/olmoe.yaml")


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML configuration file and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(config).__name__}")
    return config


def _print_ground_set(ground_set: GroundSet) -> None:
    summary = ground_set.describe()
    print(f"ground set signature   {summary['signature']}")
    print(f"model                  {summary['model_name']}")
    print(f"units                  {summary['num_units']} "
          f"({summary['num_weight_units']} weight, {summary['num_kv_units']} KV)")
    print(f"increments             {summary['num_increments']} "
          f"({summary['num_weight_increments']} weight, {summary['num_kv_increments']} KV)")
    print(f"weight tiers           {summary['weight_tiers']}")
    print(f"KV tiers               {summary['kv_tiers']}")
    print(f"context length         {summary['context_length']} tokens, batch {summary['batch_size']}")
    print()
    print("ladder step costs, one representative unit of each kind")
    print(f"  {'increment':<16} {'from':>6} {'to':>6} {'cost':>14}")
    for kind in (UnitKind.WEIGHT, UnitKind.KV):
        first = next(u for u in ground_set.units if u.kind is kind)
        for increment in ground_set.increments:
            if increment.unit_id != first.unit_id:
                continue
            print(
                f"  {increment.increment_id:<16} {increment.from_tier:>6} "
                f"{increment.to_tier:>6} {format_bytes(increment.cost_bytes):>14}"
            )


def _print_budgets(ground_set: GroundSet, fractions: Sequence[float]) -> None:
    model = ground_set.model
    reference = reference_footprint(model, ground_set.kv, ground_set.quant)
    base = ground_set.footprint(ground_set.base_allocation())
    full = ground_set.footprint(ground_set.full_allocation())

    print(f"total parameters       {total_params(model):,}")
    print()
    print(f"{'footprint':<22} {'fixed weights':>15} {'experts':>15} {'KV cache':>15} {'total':>15}")
    for label, footprint in (("reference (16 bit)", reference), ("base state", base), ("all tiers at top", full)):
        print(
            f"{label:<22} {format_bytes(footprint.fixed_weight_bytes):>15} "
            f"{format_bytes(footprint.expert_weight_bytes):>15} "
            f"{format_bytes(footprint.kv_bytes):>15} "
            f"{format_bytes(footprint.total_bytes):>15}"
        )
    print()
    base_fraction = base.total_bytes / reference.total_bytes
    print(f"reachable range        {base_fraction:.4f} to "
          f"{full.total_bytes / reference.total_bytes:.4f} of the reference footprint")
    weight_costs = [i.cost_bytes for i in ground_set.increments if i.kind is UnitKind.WEIGHT]
    kv_costs = [i.cost_bytes for i in ground_set.increments if i.kind is UnitKind.KV]
    cheapest_weight, cheapest_kv = min(weight_costs), min(kv_costs)
    print(f"cheapest increment     weight {format_bytes(cheapest_weight)}, "
          f"KV {format_bytes(cheapest_kv)} "
          f"(ratio {cheapest_weight / cheapest_kv:.1f} to 1)")
    print(f"total increments       {len(ground_set.increments)} "
          f"({len(weight_costs)} weight, {len(kv_costs)} KV)")
    print()
    print(f"{'fraction':>9} {'budget':>14} {'base':>14} {'slack':>14}  status")
    for fraction in fractions:
        try:
            plan = ground_set.plan_budget(fraction)
        except BudgetInfeasibleError:
            shortfall = base.total_bytes - int(fraction * reference.total_bytes)
            print(
                f"{fraction:>9.2f} {format_bytes(int(fraction * reference.total_bytes)):>14} "
                f"{format_bytes(base.total_bytes):>14} {'-' + format_bytes(shortfall):>14}  "
                f"INFEASIBLE, base state exceeds budget"
            )
            continue
        weight_steps = min(plan.slack_bytes // cheapest_weight, len(weight_costs))
        kv_steps = min(plan.slack_bytes // cheapest_kv, len(kv_costs))
        if plan.slack_bytes < min(cheapest_weight, cheapest_kv):
            status = "feasible but empty, slack buys no increment"
        else:
            status = (f"slack buys at most {weight_steps} of {len(weight_costs)} weight "
                      f"or {kv_steps} of {len(kv_costs)} KV steps")
        print(
            f"{fraction:>9.2f} {format_bytes(plan.budget_bytes):>14} "
            f"{format_bytes(plan.base_bytes):>14} {format_bytes(plan.slack_bytes):>14}  {status}"
        )


def resolve_model_path(config: dict[str, Any], given: Path | None) -> str:
    """Return a local snapshot path for the model, asking the hub where its cache is.

    The cache is not always under the home directory. Rented machines commonly
    point HF_HOME at whatever volume holds the disk, so the location is read
    from huggingface_hub rather than assumed.
    """
    if given is not None:
        return str(given)
    name = config["model"]["name"]
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception as error:
        from huggingface_hub import constants

        pattern = (
            Path(constants.HF_HUB_CACHE)
            / ("models--" + name.replace("/", "--"))
            / "snapshots"
            / "*"
        )
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(
                f"no local snapshot of {name} found under {constants.HF_HUB_CACHE}; "
                f"pass --model-path or download it first ({type(error).__name__}: {error})"
            ) from error
        return matches[-1]


def _run_diagnostic(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Load the model, run the requested diagnostic, and write a result record."""
    from .diagnostics import (
        diagnostic_0_sensitivity,
        diagnostic_1_headroom,
        diagnostic_2_interaction,
        noise_floor_sweep,
    )
    from .utility import build_utility

    model_path = resolve_model_path(config, args.model_path)
    seed = int(config.get("seed", 0))
    if getattr(args, "sequences", None):
        config = {**config, "calibration": {**config["calibration"], "cheap_sequences": args.sequences}}
    if getattr(args, "master_store", None):
        config = {**config, "runtime": {**config["runtime"], "master_store": args.master_store}}
    shard = getattr(args, "shard", 0)
    num_shards = getattr(args, "num_shards", 1)
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is not in range for {num_shards} shards")
    _, utility = build_utility(
        config, model_path=model_path, device=args.device, shard=shard, num_shards=num_shards
    )

    full_config = {
        **config,
        "model_path": model_path,
        "command": args.command,
        "shard": shard,
        "num_shards": num_shards,
    }
    name = args.command.replace("-", "_")
    if args.command == "second-order-floor" and getattr(args, "compare_to", None) is not None:
        # A conditioning check is not a floor: it measures one square and emits
        # a verdict rather than an epsilon block. Naming it apart keeps the two
        # out of each other's way in results/.
        name = "second_order_floor_conditioning_check"
    if num_shards > 1:
        name = f"{name}_s{shard}of{num_shards}"
    with record(name, full_config, seed) as entry:
        if args.command == "noise-floor" and args.paired:
            from .diagnostics import paired_noise_floor, top_retention, top_weight_plan

            ground = utility.ground_set
            middle = ground.model.num_hidden_layers // 2
            weight_probe = top_weight_plan(ground)
            weight_probe[middle] = {e: 3 for e in range(ground.model.num_experts)}
            kv_probe = top_retention(ground)
            kv_probe[middle] = 0.25
            probes = {
                f"d0.w.l{middle:02d}.b3": (weight_probe, top_retention(ground)),
                f"d0.kv.l{middle:02d}.r0.25": (top_weight_plan(ground), kv_probe),
            }
            entry.payload["paired"] = paired_noise_floor(utility, probes, args.subsamples)
        elif args.command == "noise-floor":
            sizes = [int(part) for part in args.sizes.split(",")]
            entry.payload["sweep"] = noise_floor_sweep(utility, sizes, args.subsamples)
            entry.payload["determinism"] = utility.verify_determinism(
                utility.ground_set.full_allocation()
            )
        elif args.command == "diagnostic-0":
            entry.payload["diagnostic_0"] = diagnostic_0_sensitivity(
                utility,
                per_expert_layers=[int(p) for p in args.expert_layers.split(",")],
                per_expert_sample=args.expert_sample,
                shard=shard,
                num_shards=num_shards,
            )
        elif args.command == "diagnostic-1":
            entry.payload["diagnostic_1"] = diagnostic_1_headroom(
                utility, args.budget, args.samples, seed, shard=shard, num_shards=num_shards
            )
        elif args.command == "diagnostic-2":
            entry.payload["diagnostic_2"] = diagnostic_2_interaction(utility, args.budget)
        elif args.command == "second-order-floor":
            entry.payload["second_order_floor"] = _second_order_floor(args, utility)
        entry.payload["setup"] = utility.describe()
        entry.payload["ground_set"] = utility.ground_set.describe()
        entry.payload["evaluation_count"] = utility.evaluation_count


def _second_order_floor(args: argparse.Namespace, utility: Any) -> dict[str, Any]:
    """Measure the spread of the second-order difference and print the epsilon it implies."""
    import sys

    from .submodularity import (
        MODALITY_LABELS,
        TierLadders,
        compare_conditioning_floor,
        default_floor_specs,
        epsilon_from_floor,
        format_conditioning_check,
        format_second_order_floor,
        parse_upgrades,
        second_order_noise_floor,
    )
    from .submodularity import KV as KV_KIND
    from .submodularity import WEIGHT as WEIGHT_KIND

    target, conditioning = (int(part) for part in args.layers.split(","))
    ladders = TierLadders.from_ground_set(utility.ground_set)
    weight_move = parse_upgrades(args.weight_move, WEIGHT_KIND)[0] if args.weight_move else None
    kv_move = parse_upgrades(args.kv_move, KV_KIND)[0] if args.kv_move else None
    weight_context = (
        parse_upgrades(args.weight_conditioning, WEIGHT_KIND)[0]
        if args.weight_conditioning
        else None
    )
    kv_context = (
        parse_upgrades(args.kv_conditioning, KV_KIND)[0] if args.kv_conditioning else None
    )
    modalities = (
        [part.strip() for part in args.modalities.split(",") if part.strip()]
        if args.modalities
        else None
    )
    specs = default_floor_specs(
        ladders,
        target,
        conditioning,
        weight_move,
        kv_move,
        weight_conditioning=weight_context,
        kv_conditioning=kv_context,
        modalities=modalities,
    )

    def progress(index: int, total: int, plan: Any, subsample: int) -> None:
        state = plan.compact_dict()
        print(
            f"[state {index:>2}/{total}] subsample {subsample} | "
            f"w={state['weight_bits']} kv={state['kv_retention']}",
            file=sys.stderr,
            flush=True,
        )

    floor = second_order_noise_floor(
        utility, specs, num_subsamples=args.subsamples, progress=progress
    )
    # The tolerance a run classifies against is recorded next to the floor it
    # came from, in both readings, so the choice is visible rather than implied.
    if args.compare_to is not None:
        import json as _json

        baseline_record = _json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        baseline = baseline_record.get("payload", {}).get(
            "second_order_floor", baseline_record
        )
        comparison = compare_conditioning_floor(baseline, floor)
        comparison["baseline_record"] = str(Path(args.compare_to).resolve())
        comparison["baseline_git_commit"] = baseline_record.get("git_commit")
        floor["conditioning_check"] = comparison
        print(format_second_order_floor(floor))
        print()
        print(format_conditioning_check(comparison))
        return floor

    single = epsilon_from_floor(floor, subsamples_per_cell=1)
    binding = epsilon_from_floor(floor, subsamples_per_cell=args.cell_subsamples)
    floor["epsilon"] = {
        "binding": "mean_of_k",
        "rule": (
            "DECISION.md Amendment 1 (2026-08-21): epsilon(modality) = sigma2(modality) / "
            "sqrt(k), where k is the number of calibration subsamples averaged into one "
            "reported cell by the run being classified"
        ),
        "cell_subsamples": args.cell_subsamples,
        "mean_of_k": binding.as_dict(),
        "single_draw": {
            **single.as_dict(),
            "note": (
                "the raw measured sigma2, kept for reference; superseded as the tolerance "
                "by Amendment 1"
            ),
        },
        "config_yaml": {
            "epsilon_ppl": dict(binding.by_modality),
            "epsilon_source": binding.source,
            "epsilon_cell_subsamples": args.cell_subsamples,
        },
    }
    print(format_second_order_floor(floor))
    print()
    print(
        f"Binding tolerance (DECISION.md Amendment 1): sigma2 / sqrt({args.cell_subsamples})"
    )
    print("Paste into configs/*.yaml under submodularity:")
    print("  epsilon_ppl:")
    for modality, value in sorted(binding.by_modality):
        print(
            f"    {modality}: {value:.6f}   # {MODALITY_LABELS[modality]}, "
            f"sigma2 {single.for_modality(modality):.6f}"
        )
    print(f'  epsilon_source: "{binding.source}"')
    print(f"  epsilon_cell_subsamples: {args.cell_subsamples}")
    print()
    print(
        f"The matrix run MUST then use exactly {args.cell_subsamples} subsamples per cell; "
        "a different count invalidates this tolerance."
    )
    return floor


def _merge_shards(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Combine shard records for one diagnostic into the whole experiment's result."""
    from .diagnostics import merge_diagnostic_0, merge_diagnostic_1
    from .records import load_records

    mergers = {"diagnostic_0": merge_diagnostic_0, "diagnostic_1": merge_diagnostic_1}
    if args.name not in mergers:
        raise ValueError(f"no merge rule for {args.name!r}; choose one of {sorted(mergers)}")

    found = [r for r in load_records(args.results_dir) if r["name"].startswith(f"{args.name}_s")]
    if not found:
        raise FileNotFoundError(f"no shard records for {args.name!r} in {args.results_dir}")
    commits = {r["git_commit"] for r in found}
    if len(commits) > 1:
        raise ValueError(f"shard records span more than one commit: {sorted(commits)}")

    payloads = [r["payload"][args.name] for r in found]
    merged = mergers[args.name](payloads)
    # The merged record carries the configuration the shards actually ran, not
    # whatever the config file says now. A run overridden on the command line
    # would otherwise be written down with the wrong sequence count.
    shard_config = {**found[0]["config"], "merged_from": [r["name"] for r in found]}
    with record(
        f"{args.name}__merged",
        shard_config,
        int(shard_config.get("seed", 0)),
        args.results_dir,
    ) as entry:
        entry.payload[args.name] = merged
        entry.payload["setup"] = found[0]["payload"].get("setup", {})
        entry.payload["evaluation_count"] = sum(
            r["payload"].get("evaluation_count", 0) for r in found
        )
    status = "complete" if merged.get("complete", True) else "INCOMPLETE"
    print(f"merged {len(found)} shard(s) of {args.name}: {status}")
    for key in ("num_probes_measured", "num_probes_total", "num_samples_measured"):
        if key in merged:
            print(f"  {key}: {merged[key]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested command."""
    parser = argparse.ArgumentParser(prog="submokv", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to a YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ground-set", help="print the units, tiers, and increment costs")
    subparsers.add_parser("budget", help="print the footprint breakdown and budget feasibility")
    describe = subparsers.add_parser("describe", help="print the ground set summary as JSON")
    describe.add_argument("--indent", type=int, default=2)

    for name, help_text in (
        ("noise-floor", "measure how far perplexity moves with the calibration draw"),
        ("diagnostic-0", "sensitivity spread, per layer and per expert"),
        ("diagnostic-1", "achievable headroom across random feasible allocations"),
        ("diagnostic-2", "interaction between the weight and KV axes"),
        (
            "second-order-floor",
            "spread of the second-order difference, which is what epsilon is set from",
        ),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--model-path", type=Path, default=None)
        sub.add_argument("--device", type=str, default=None)
        sub.add_argument(
            "--sequences",
            type=int,
            default=None,
            help="override calibration.cheap_sequences for this run",
        )
        sub.add_argument(
            "--master-store",
            type=str,
            default=None,
            choices=["memory", "checkpoint"],
            help="where the unmodified expert weights are read from",
        )
        sub.add_argument("--shard", type=int, default=0, help="index of this worker")
        sub.add_argument(
            "--num-shards",
            type=int,
            default=1,
            help="split the work across this many workers, one per accelerator",
        )
        if name == "noise-floor":
            sub.add_argument("--sizes", type=str, default="16,32,64")
            sub.add_argument("--subsamples", type=int, default=6)
            sub.add_argument(
                "--paired",
                action="store_true",
                help="measure the spread of a drop-from-top difference instead of the sweep",
            )
        if name == "diagnostic-0":
            sub.add_argument("--expert-layers", type=str, default="0,8")
            sub.add_argument("--expert-sample", type=int, default=8)
        if name in ("diagnostic-1", "diagnostic-2"):
            sub.add_argument("--budget", type=float, default=0.35)
        if name == "diagnostic-1":
            sub.add_argument("--samples", type=int, default=30)
        if name == "second-order-floor":
            sub.add_argument(
                "--subsamples",
                type=int,
                default=6,
                help="non-overlapping calibration draws per square; at least five",
            )
            sub.add_argument(
                "--layers",
                type=str,
                default="8,4",
                help="TARGET,CONDITIONING layer indices for the two intra-component squares",
            )
            sub.add_argument(
                "--weight-move",
                type=str,
                default=None,
                help="FROM:TO weight step to measure the floor on (default: ladder bottom step)",
            )
            sub.add_argument(
                "--kv-move",
                type=str,
                default=None,
                help="FROM:TO retention step to measure the floor on (default: ladder bottom step)",
            )
            sub.add_argument(
                "--weight-conditioning",
                type=str,
                default=None,
                help="FROM:TO weight move to condition on (default: the same as --weight-move)",
            )
            sub.add_argument(
                "--kv-conditioning",
                type=str,
                default=None,
                help="FROM:TO retention move to condition on (default: the same as --kv-move)",
            )
            sub.add_argument(
                "--modalities",
                type=str,
                default=None,
                help="comma-separated modalities to measure (default: all four)",
            )
            sub.add_argument(
                "--compare-to",
                type=Path,
                default=None,
                help=(
                    "a prior floor record; apply the DECISION.md Amendment 2 C conditioning "
                    "rule instead of emitting an epsilon block"
                ),
            )
            sub.add_argument(
                "--cell-subsamples",
                type=int,
                default=3,
                help=(
                    "k, the number of draws the calibrated run averages into one cell; "
                    "epsilon is sigma2/sqrt(k) per DECISION.md Amendment 1, and the "
                    "calibrated run must use exactly this many"
                ),
            )

    merge = subparsers.add_parser("merge", help="combine shard records into one result")
    merge.add_argument("--name", type=str, required=True, help="diagnostic_0 or diagnostic_1")
    merge.add_argument("--results-dir", type=Path, default=Path("results"))

    args = parser.parse_args(argv)
    config = load_config(args.config)
    ground_set = GroundSet.from_config(config)

    if args.command == "ground-set":
        _print_ground_set(ground_set)
    elif args.command == "budget":
        _print_budgets(ground_set, config.get("budgets", {}).get("fractions", []))
    elif args.command == "describe":
        print(json.dumps(ground_set.describe(), indent=args.indent))
    elif args.command == "merge":
        _merge_shards(args, config)
    else:
        _run_diagnostic(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
