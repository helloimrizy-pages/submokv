"""Command line entry points for Sub-MoKV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from .ground_set import BudgetInfeasibleError, GroundSet, UnitKind
from .memory import format_bytes, reference_footprint, total_params

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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested command."""
    parser = argparse.ArgumentParser(prog="submokv", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to a YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ground-set", help="print the units, tiers, and increment costs")
    subparsers.add_parser("budget", help="print the footprint breakdown and budget feasibility")
    describe = subparsers.add_parser("describe", help="print the ground set summary as JSON")
    describe.add_argument("--indent", type=int, default=2)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    ground_set = GroundSet.from_config(config)

    if args.command == "ground-set":
        _print_ground_set(ground_set)
    elif args.command == "budget":
        _print_budgets(ground_set, config.get("budgets", {}).get("fractions", []))
    elif args.command == "describe":
        print(json.dumps(ground_set.describe(), indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
