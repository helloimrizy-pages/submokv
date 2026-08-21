"""Four-state tests for diminishing returns in joint weight/KV allocation.

Each interaction holds every unmentioned layer at the full 16-bit/100% state,
compresses a target and a conditioning component, and evaluates the four
corners of the resulting lattice square::

    S_A             S_A + j
    S_B             S_B + j

``S_B`` differs from ``S_A`` only by upgrading the conditioning component and
``j`` upgrades the target component.  Consequently ``S_A`` is a subset of
``S_B`` in the allocation lattice.  With utility defined as negative
perplexity degradation, the diminishing-returns difference is

    [F(S_A + j) - F(S_A)] - [F(S_B + j) - F(S_B)].

Positive values are submodular; negative values are synergy spikes.  The
implementation drives the model with raw per-layer plans rather than with
allocator increments, because a diagnostic square moves one layer at a time.
The *tier ladders* it may move along, however, come from the ground set and
from nowhere else: a probe of a tier the allocator cannot buy is either refused
or carried through the record labelled ``in_ground_set: false``.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ground_set import GroundSet
from .utility import CHEAP, UtilityFunction

WEIGHT = "weight"
KV = "kv"

# The state every unmentioned layer is held at.  This is the full-precision
# reference the diagnostic measures degradation against, not a ladder top: it
# stays 16 bit / 100% cache whatever the allocator's ground set contains.
FULL_WEIGHT_BITS = 16
FULL_KV_RETENTION = 1.00

# Every tier the quantizer and the retention controller can physically express.
# Membership here only rejects nonsense such as a 5-bit width or a retention of
# 1.5.  It says nothing about whether the allocator may buy the tier; that
# question is answered by the ground set, through TierLadders below.
EXPRESSIBLE_WEIGHT_TIERS: tuple[int, ...] = (2, 3, 4, 8, 16)
EXPRESSIBLE_KV_TIERS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

WEIGHT_TO_WEIGHT = "weight_to_weight"
KV_TO_KV = "kv_to_kv"
WEIGHT_GIVEN_KV = "weight_given_kv"
KV_GIVEN_WEIGHT = "kv_given_weight"

MODALITY_LABELS = {
    WEIGHT_TO_WEIGHT: "W|W",
    KV_TO_KV: "KV|KV",
    WEIGHT_GIVEN_KV: "W|KV",
    KV_GIVEN_WEIGHT: "KV|W",
}


def _number(value: int | float) -> int | float:
    """Return integral values as ints so identifiers and JSON stay readable."""
    return int(value) if float(value).is_integer() else float(value)


def _tier_text(value: int | float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}"


def _tier_slug(value: int | float) -> str:
    return _tier_text(value).replace(".", "p")


class OutsideGroundSetError(ValueError):
    """Raised when a probe moves along a tier the allocator cannot buy."""


@dataclass(frozen=True)
class TierUpgrade:
    """One target or conditioning move on a weight or KV tier ladder."""

    kind: str
    before: int | float
    after: int | float

    def __post_init__(self) -> None:
        if self.kind not in (WEIGHT, KV):
            raise ValueError(f"unknown component kind {self.kind!r}")
        allowed: tuple[int | float, ...] = (
            EXPRESSIBLE_WEIGHT_TIERS if self.kind == WEIGHT else EXPRESSIBLE_KV_TIERS
        )
        before = _number(self.before)
        after = _number(self.after)
        if before not in allowed or after not in allowed:
            raise ValueError(
                f"{self.kind} upgrade {before}->{after} is outside expressible tiers {allowed}"
            )
        if before >= after:
            raise ValueError(f"an upgrade must increase its tier, got {before}->{after}")
        if self.kind == WEIGHT and (not isinstance(before, int) or not isinstance(after, int)):
            raise ValueError("weight bit widths must be integers")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    @property
    def short_kind(self) -> str:
        return "W" if self.kind == WEIGHT else "KV"

    @property
    def label(self) -> str:
        return f"{self.short_kind}:{_tier_text(self.before)}->{_tier_text(self.after)}"

    def as_dict(self, ladders: "TierLadders | None" = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "from": self.before, "to": self.after}
        if ladders is not None:
            payload["in_ground_set"] = ladders.contains(self)
        return payload


@dataclass(frozen=True)
class TierLadders:
    """The weight and KV ladders the allocator may actually buy.

    The ground set is the single source of truth for which tiers exist.  This
    holds a copy of its two ladders so that every part of the diagnostic asks
    the same question of the same numbers, and so that a probe of a tier the
    allocator cannot reach is visible rather than silently ordinary.
    """

    weight_tiers: tuple[int, ...]
    kv_tiers: tuple[float, ...]

    @classmethod
    def from_ground_set(cls, ground_set: GroundSet) -> "TierLadders":
        """Return the ladders declared by a ground set, which the config sets."""
        return cls(
            weight_tiers=tuple(int(tier) for tier in ground_set.weight_tiers),
            kv_tiers=tuple(float(tier) for tier in ground_set.kv_tiers),
        )

    def __post_init__(self) -> None:
        for kind, tiers in ((WEIGHT, self.weight_tiers), (KV, self.kv_tiers)):
            if len(tiers) < 2:
                raise ValueError(f"the {kind} ladder needs at least two tiers, got {tiers}")
            if list(tiers) != sorted(tiers) or len(set(tiers)) != len(tiers):
                raise ValueError(f"the {kind} ladder must be ordered and distinct, got {tiers}")
            expressible = EXPRESSIBLE_WEIGHT_TIERS if kind == WEIGHT else EXPRESSIBLE_KV_TIERS
            outside = [tier for tier in tiers if _number(tier) not in expressible]
            if outside:
                raise ValueError(
                    f"the {kind} ladder names tier(s) {outside} that the model path cannot "
                    f"express; expressible tiers are {expressible}"
                )

    def tiers(self, kind: str) -> tuple[int | float, ...]:
        """Return one ladder, low to high."""
        if kind == WEIGHT:
            return self.weight_tiers
        if kind == KV:
            return self.kv_tiers
        raise ValueError(f"unknown component kind {kind!r}")

    def contains(self, upgrade: TierUpgrade) -> bool:
        """Return whether both ends of a move sit on the ground set's ladder."""
        tiers = tuple(_number(tier) for tier in self.tiers(upgrade.kind))
        return upgrade.before in tiers and upgrade.after in tiers

    def adjacent(self, kind: str) -> tuple[TierUpgrade, ...]:
        """Return every single-step move on one ladder."""
        tiers = self.tiers(kind)
        return tuple(
            TierUpgrade(kind, before, after) for before, after in zip(tiers, tiers[1:])
        )

    def full_move(self, kind: str) -> TierUpgrade:
        """Return the bottom-to-top move on one ladder, used as a conditioning default."""
        tiers = self.tiers(kind)
        return TierUpgrade(kind, tiers[0], tiers[-1])

    def outside(self, upgrades: Iterable[TierUpgrade]) -> tuple[TierUpgrade, ...]:
        """Return the moves that leave the ground set, in the order given."""
        return tuple(upgrade for upgrade in upgrades if not self.contains(upgrade))

    def reference_in_ground_set(self) -> bool:
        """Return whether the full-precision reference state is itself buyable.

        The diagnostic holds every unmentioned layer at 16 bit / 100% cache. If
        the ground set's ladders stop below that, the reference corner of every
        square is a state the allocator could never occupy, and the record has
        to say so.
        """
        return (
            FULL_WEIGHT_BITS in self.weight_tiers
            and _number(FULL_KV_RETENTION) in tuple(_number(t) for t in self.kv_tiers)
        )

    def describe(self) -> dict[str, Any]:
        """Return the ladders for result records."""
        return {
            "weight_tiers": list(self.weight_tiers),
            "kv_tiers": list(self.kv_tiers),
            "source": "ground_set",
            "reference_in_ground_set": self.reference_in_ground_set(),
        }


@dataclass(frozen=True)
class InteractionSpec:
    """A target upgrade measured with one conditioning component low and high."""

    target_layer: int
    target: TierUpgrade
    conditioning_layer: int
    conditioning: TierUpgrade

    def __post_init__(self) -> None:
        if self.target_layer < 0 or self.conditioning_layer < 0:
            raise ValueError("layer indices must be non-negative")
        if (
            self.target.kind == self.conditioning.kind
            and self.target_layer == self.conditioning_layer
        ):
            raise ValueError("an intra-component interaction needs two distinct layers")

    @property
    def modality(self) -> str:
        pair = (self.target.kind, self.conditioning.kind)
        return {
            (WEIGHT, WEIGHT): WEIGHT_TO_WEIGHT,
            (KV, KV): KV_TO_KV,
            (WEIGHT, KV): WEIGHT_GIVEN_KV,
            (KV, WEIGHT): KV_GIVEN_WEIGHT,
        }[pair]

    @property
    def interaction_id(self) -> str:
        target = (
            f"{self.target.short_kind.lower()}{_tier_slug(self.target.before)}"
            f"to{_tier_slug(self.target.after)}"
        )
        context = (
            f"{self.conditioning.short_kind.lower()}{_tier_slug(self.conditioning.before)}"
            f"to{_tier_slug(self.conditioning.after)}"
        )
        return (
            f"{self.modality}.t{self.target_layer:02d}.{target}."
            f"c{self.conditioning_layer:02d}.{context}"
        )

    @property
    def label(self) -> str:
        return (
            f"L{self.target_layer} ({self.target.label}) | "
            f"L{self.conditioning_layer} ({self.conditioning.label})"
        )

    def in_ground_set(self, ladders: "TierLadders") -> bool:
        """Return whether both moves in this square sit on the allocator's ladders."""
        return ladders.contains(self.target) and ladders.contains(self.conditioning)

    def as_dict(self, ladders: "TierLadders | None" = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "interaction_id": self.interaction_id,
            "modality": self.modality,
            "target_layer": self.target_layer,
            "target_upgrade": self.target.as_dict(ladders),
            "conditioning_layer": self.conditioning_layer,
            "conditioning_upgrade": self.conditioning.as_dict(ladders),
            "label": self.label,
        }
        if ladders is not None:
            payload["in_ground_set"] = self.in_ground_set(ladders)
        return payload


@dataclass(frozen=True)
class DiagnosticPlan:
    """A uniform expert bit width and KV ratio for every decoder layer."""

    weight_bits: tuple[int, ...]
    kv_retention: tuple[float, ...]

    @classmethod
    def full(cls, num_layers: int) -> "DiagnosticPlan":
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        return cls((16,) * num_layers, (1.0,) * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.weight_bits)

    def with_value(self, layer: int, kind: str, value: int | float) -> "DiagnosticPlan":
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} is outside [0, {self.num_layers - 1}]")
        if kind == WEIGHT:
            bits = list(self.weight_bits)
            bits[layer] = int(value)
            return DiagnosticPlan(tuple(bits), self.kv_retention)
        if kind == KV:
            retention = list(self.kv_retention)
            retention[layer] = float(value)
            return DiagnosticPlan(self.weight_bits, tuple(retention))
        raise ValueError(f"unknown component kind {kind!r}")

    def expanded_weight_plan(self, num_experts: int) -> dict[int, dict[int, int]]:
        return {
            layer: {expert: bits for expert in range(num_experts)}
            for layer, bits in enumerate(self.weight_bits)
        }

    def expanded_retention_plan(self) -> dict[int, float]:
        return {layer: ratio for layer, ratio in enumerate(self.kv_retention)}

    def compact_dict(self) -> dict[str, dict[str, int | float]]:
        """Return only layers that differ from the full-precision reference."""
        return {
            "weight_bits": {
                str(layer): bits
                for layer, bits in enumerate(self.weight_bits)
                if bits != FULL_WEIGHT_BITS
            },
            "kv_retention": {
                str(layer): ratio
                for layer, ratio in enumerate(self.kv_retention)
                if ratio != FULL_KV_RETENTION
            },
        }

    def is_no_greater_than(self, other: "DiagnosticPlan") -> bool:
        if self.num_layers != other.num_layers:
            return False
        return all(a <= b for a, b in zip(self.weight_bits, other.weight_bits)) and all(
            a <= b for a, b in zip(self.kv_retention, other.kv_retention)
        )


@dataclass(frozen=True)
class FourStatePlans:
    """The four allocations needed for one second-order marginal difference."""

    s_a: DiagnosticPlan
    s_a_union_j: DiagnosticPlan
    s_b: DiagnosticPlan
    s_b_union_j: DiagnosticPlan

    def __post_init__(self) -> None:
        if not self.s_a.is_no_greater_than(self.s_b):
            raise ValueError("S_A must be no greater than S_B")
        if not self.s_a_union_j.is_no_greater_than(self.s_b_union_j):
            raise ValueError("S_A+j must be no greater than S_B+j")


def four_state_plans(spec: InteractionSpec, num_layers: int) -> FourStatePlans:
    """Construct the lattice square for an interaction around the full model."""
    base = DiagnosticPlan.full(num_layers)
    s_a = base.with_value(
        spec.target_layer, spec.target.kind, spec.target.before
    ).with_value(
        spec.conditioning_layer,
        spec.conditioning.kind,
        spec.conditioning.before,
    )
    s_a_union_j = s_a.with_value(spec.target_layer, spec.target.kind, spec.target.after)
    s_b = s_a.with_value(
        spec.conditioning_layer,
        spec.conditioning.kind,
        spec.conditioning.after,
    )
    s_b_union_j = s_b.with_value(spec.target_layer, spec.target.kind, spec.target.after)
    return FourStatePlans(s_a, s_a_union_j, s_b, s_b_union_j)


def adjacent_upgrades(kind: str, ladders: TierLadders) -> tuple[TierUpgrade, ...]:
    """Return every adjacent move on one of the ground set's tier ladders."""
    return ladders.adjacent(kind)


def parse_upgrades(text: str, kind: str) -> tuple[TierUpgrade, ...]:
    """Parse comma-separated ``from:to`` transitions for a CLI."""
    parsed: list[TierUpgrade] = []
    for part in text.split(","):
        values = part.strip().split(":")
        if len(values) != 2:
            raise ValueError(
                f"invalid {kind} transition {part!r}; expected comma-separated FROM:TO pairs"
            )
        try:
            before, after = (float(value) for value in values)
        except ValueError as error:
            raise ValueError(f"invalid numeric transition {part!r}") from error
        parsed.append(TierUpgrade(kind, before, after))
    if not parsed:
        raise ValueError(f"at least one {kind} transition is required")
    return tuple(parsed)


def build_interaction_matrix(
    layers: Sequence[int],
    ladders: TierLadders,
    weight_upgrades: Sequence[TierUpgrade] | None = None,
    kv_upgrades: Sequence[TierUpgrade] | None = None,
    weight_conditioning: TierUpgrade | None = None,
    kv_conditioning: TierUpgrade | None = None,
) -> tuple[InteractionSpec, ...]:
    """Return cross- and intra-component pairwise tests in deterministic order.

    For four layers and one target transition on each ladder this produces 32
    interactions: 12 directed W|W pairs, 12 directed KV|KV pairs, and both
    cross-component directions at each of the four layers.

    Every default is read from ``ladders``, which comes from the ground set, so
    a config-level tier change moves the matrix.  The conditioning moves default
    to the bottom-to-top move on each ladder and can be narrowed to an adjacent
    step when the ladder's extremes are the part being avoided.
    """
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("at least one test layer is required")
    if len(set(selected)) != len(selected):
        raise ValueError(f"test layers must be unique, got {selected}")
    if any(layer < 0 for layer in selected):
        raise ValueError(f"test layers must be non-negative, got {selected}")

    weight_moves = tuple(weight_upgrades or (ladders.adjacent(WEIGHT)[0],))
    kv_moves = tuple(kv_upgrades or (ladders.adjacent(KV)[0],))
    if any(move.kind != WEIGHT for move in weight_moves):
        raise ValueError("weight_upgrades contains a non-weight transition")
    if any(move.kind != KV for move in kv_moves):
        raise ValueError("kv_upgrades contains a non-KV transition")

    full_weight_move = weight_conditioning or ladders.full_move(WEIGHT)
    full_kv_move = kv_conditioning or ladders.full_move(KV)
    if full_weight_move.kind != WEIGHT or full_kv_move.kind != KV:
        raise ValueError("a conditioning move must be on the ladder it conditions")
    interactions: list[InteractionSpec] = []

    # Cross-component tests first: these are the central Sub-MoKV hypothesis.
    for layer in selected:
        interactions.extend(
            InteractionSpec(layer, move, layer, full_kv_move) for move in weight_moves
        )
        interactions.extend(
            InteractionSpec(layer, move, layer, full_weight_move) for move in kv_moves
        )

    # Directed pairs matter: layer A conditioned on B need not equal B conditioned on A.
    for target_layer in selected:
        for conditioning_layer in selected:
            if target_layer == conditioning_layer:
                continue
            interactions.extend(
                InteractionSpec(target_layer, move, conditioning_layer, full_weight_move)
                for move in weight_moves
            )
    for target_layer in selected:
        for conditioning_layer in selected:
            if target_layer == conditioning_layer:
                continue
            interactions.extend(
                InteractionSpec(target_layer, move, conditioning_layer, full_kv_move)
                for move in kv_moves
            )
    return tuple(interactions)


def validate_test_layers(layers: Sequence[int], num_layers: int) -> tuple[int, ...]:
    """Validate user/config layer indices against the loaded architecture."""
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("at least one test layer is required")
    if len(set(selected)) != len(selected):
        raise ValueError(f"test layers must be unique, got {selected}")
    invalid = [layer for layer in selected if not 0 <= layer < num_layers]
    if invalid:
        raise ValueError(
            f"layers {invalid} do not exist in a {num_layers}-layer model; valid indices are "
            f"0 through {num_layers - 1}"
        )
    return selected


@dataclass(frozen=True)
class EpsilonPolicy:
    """The tolerance a second-order difference is classified against.

    Weight-side and KV-side differences do not share a scale, so a single global
    constant either forgives real KV structure or reports weight noise as
    signal.  This carries one tolerance per modality together with a note of
    where the numbers came from, and the runner writes the value each row used
    into that row.  A verdict therefore cannot be read without seeing what it
    was compared against.
    """

    by_modality: tuple[tuple[str, float], ...]
    source: str

    def __post_init__(self) -> None:
        known = set(MODALITY_LABELS)
        for modality, value in self.by_modality:
            if modality not in known:
                raise ValueError(f"unknown modality {modality!r}; expected one of {sorted(known)}")
            if value < 0:
                raise ValueError(f"epsilon must be non-negative, got {value} for {modality}")
        if len(dict(self.by_modality)) != len(self.by_modality):
            raise ValueError(f"a modality appears twice in {self.by_modality}")
        if set(dict(self.by_modality)) != known:
            missing = sorted(known - set(dict(self.by_modality)))
            raise ValueError(f"no epsilon given for modality/modalities {missing}")

    @classmethod
    def uniform(cls, value: float, source: str = "unmeasured constant") -> "EpsilonPolicy":
        """Return one tolerance shared by every modality."""
        return cls(tuple((modality, float(value)) for modality in sorted(MODALITY_LABELS)), source)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, float], source: str = "measured"
    ) -> "EpsilonPolicy":
        """Return a per-modality tolerance, which every modality must appear in."""
        return cls(tuple(sorted((str(k), float(v)) for k, v in values.items())), source)

    @classmethod
    def coerce(cls, value: "float | Mapping[str, float] | EpsilonPolicy") -> "EpsilonPolicy":
        """Accept a policy, a per-modality mapping, or a bare number."""
        if isinstance(value, EpsilonPolicy):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        return cls.uniform(float(value))

    def for_modality(self, modality: str) -> float:
        """Return the tolerance one modality is classified against."""
        try:
            return dict(self.by_modality)[modality]
        except KeyError as error:
            raise ValueError(f"no epsilon for modality {modality!r}") from error

    def is_uniform(self) -> bool:
        """Return whether every modality shares one tolerance."""
        return len({value for _, value in self.by_modality}) == 1

    def default(self) -> float:
        """Return the tolerance to use when a row does not name its modality.

        Rows produced by this module always carry their own value, so this is
        only reached by hand-assembled rows.  The widest tolerance is returned,
        which is the conservative choice: it resolves fewer cells and calls
        fewer of them supermodular.
        """
        return max(value for _, value in self.by_modality)

    def as_dict(self) -> dict[str, Any]:
        """Return the tolerances for result records."""
        return {
            "source": self.source,
            "uniform": self.is_uniform(),
            "by_modality": dict(self.by_modality),
        }


@dataclass(frozen=True)
class EffectFloor:
    """The allocation-relevance gate of DECISION.md Amendment 2 B.

    Statistical resolution says a difference can be told apart from zero.  It
    does not say an allocator would act on it, and with epsilon near 0.04% of a
    perplexity the two have come apart.  This carries both clauses:

      E1  the upgrade is worth buying at all:  m > multiple * sigma1(kind)
      E2  the interaction is material:         |D| >= phi * m

    where ``m`` is the larger of the target's two marginal gains.  ``sigma1`` is
    the *single-draw* first-order noise and is deliberately not divided by the
    number of subsamples averaged: the statistical gate should get cheaper with
    more data, this one must not.
    """

    by_kind: tuple[tuple[str, float], ...]
    phi: float
    multiple: float
    source: str

    def __post_init__(self) -> None:
        if set(dict(self.by_kind)) != {WEIGHT, KV}:
            raise ValueError(
                f"an effect floor is needed for both {WEIGHT!r} and {KV!r}, got "
                f"{sorted(dict(self.by_kind))}"
            )
        for kind, value in self.by_kind:
            if value < 0:
                raise ValueError(f"the {kind} effect floor must be non-negative, got {value}")
        if self.phi < 0:
            raise ValueError(f"phi must be non-negative, got {self.phi}")

    @classmethod
    def from_first_order_noise(
        cls,
        sigma1: Mapping[str, float],
        phi: float,
        multiple: float = 3.0,
        source: str = "measured first-order noise",
    ) -> "EffectFloor":
        """Build E1 thresholds as ``multiple * sigma1`` for each component kind."""
        return cls(
            by_kind=tuple(sorted((str(k), multiple * float(v)) for k, v in sigma1.items())),
            phi=float(phi),
            multiple=float(multiple),
            source=source,
        )

    @classmethod
    def from_floor_record(
        cls, floor: Mapping[str, Any], phi: float, multiple: float = 3.0
    ) -> "EffectFloor":
        """Derive the E1 thresholds from a measured second-order floor record.

        ``sigma1`` is the largest ``stdev m(A)`` across the squares whose target
        is of that kind, which is the conservative choice.
        """
        sigma1: dict[str, float] = {}
        for row in floor["specs"]:
            kind = str(row["target_upgrade"]["kind"])
            sigma1[kind] = max(sigma1.get(kind, 0.0), float(row["marginal_gain_s_a_stdev"]))
        missing = sorted({WEIGHT, KV} - set(sigma1))
        if missing:
            raise ValueError(
                f"the floor record measured no square with a {missing} target, so no effect "
                "floor can be derived for it"
            )
        return cls.from_first_order_noise(
            sigma1,
            phi,
            multiple,
            source=(
                f"{multiple:g} x single-draw stdev of the target marginal gain, from the "
                "measured second-order floor"
            ),
        )

    def for_kind(self, kind: str) -> float:
        """Return the E1 threshold for a weight or KV target."""
        try:
            return dict(self.by_kind)[kind]
        except KeyError as error:
            raise ValueError(f"no effect floor for component kind {kind!r}") from error

    def evaluate(self, kind: str, magnitude: float, difference: float) -> dict[str, Any]:
        """Return both clauses for one cell, and why it failed if it did."""
        threshold = self.for_kind(kind)
        e1 = magnitude > threshold
        e2 = abs(difference) >= self.phi * magnitude
        reasons = []
        if not e1:
            reasons.append("effect_e1_target_buys_nothing")
        if not e2:
            reasons.append("effect_e2_interaction_immaterial")
        return {
            "effect_floor_ppl": threshold,
            "effect_relative_phi": self.phi,
            "effect_magnitude_m": magnitude,
            "relative_interaction": abs(difference) / magnitude if magnitude else float("inf"),
            "gate_effect_e1": e1,
            "gate_effect_e2": e2,
            "gate_effect": e1 and e2,
            "effect_failure_reasons": reasons,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the effect gate for result records."""
        return {
            "rule": (
                "DECISION.md Amendment 2 B. E1: m > multiple * sigma1(kind). "
                "E2: |D| >= phi * m. m = max(|mean m(S_A)|, |mean m(S_B)|)."
            ),
            "source": self.source,
            "multiple": self.multiple,
            "phi": self.phi,
            "by_kind": dict(self.by_kind),
            "sigma1_not_divided_by_sqrt_k": True,
        }


@dataclass(frozen=True)
class EpsilonBand:
    """The tolerance at the sigma2 point estimate and at both interval bounds.

    DECISION.md Amendment 2 A: sigma2 is estimated from six values and its 95%
    interval spans roughly a factor of four, so the resolved rate is reported
    three times.  The branch is called on ``point`` and only on ``point``; the
    bounds are a robustness band and never select a verdict.
    """

    point: EpsilonPolicy
    low: EpsilonPolicy
    high: EpsilonPolicy
    subsamples_per_cell: int
    confidence: float

    @classmethod
    def from_floor_record(
        cls, floor: Mapping[str, Any], subsamples_per_cell: int
    ) -> "EpsilonBand":
        """Derive the point tolerance and both bounds from a measured floor."""
        scale = math.sqrt(subsamples_per_cell)
        point: dict[str, float] = {}
        low: dict[str, float] = {}
        high: dict[str, float] = {}
        confidence = 0.95
        for modality, entry in floor["by_modality"].items():
            interval = entry["second_order_stdev_pooled_ci"]
            if interval is None:
                raise ValueError(
                    f"the floor's {modality!r} square carries no confidence interval, so no "
                    "band can be reported for it"
                )
            confidence = float(interval.get("confidence", confidence))
            point[modality] = float(entry["second_order_stdev_pooled"]) / scale
            low[modality] = float(interval["low"]) / scale
            high[modality] = float(interval["high"]) / scale
        tag = f"measured second-order floor, divided by sqrt({subsamples_per_cell})"
        return cls(
            point=EpsilonPolicy.from_mapping(point, tag),
            low=EpsilonPolicy.from_mapping(low, f"{tag}, sigma2 at its lower bound"),
            high=EpsilonPolicy.from_mapping(high, f"{tag}, sigma2 at its upper bound"),
            subsamples_per_cell=int(subsamples_per_cell),
            confidence=confidence,
        )

    def readings(self) -> tuple[tuple[str, EpsilonPolicy], ...]:
        """Return the three tolerances, the binding one first."""
        return (("point", self.point), ("sigma2_low", self.low), ("sigma2_high", self.high))

    def as_dict(self) -> dict[str, Any]:
        """Return the band for result records."""
        return {
            "rule": (
                "DECISION.md Amendment 2 A: the branch is called on the point estimate only; "
                "the bounds are a robustness band and never select a verdict"
            ),
            "binding_reading": "point",
            "subsamples_per_cell": self.subsamples_per_cell,
            "confidence": self.confidence,
            "point": self.point.as_dict(),
            "sigma2_low": self.low.as_dict(),
            "sigma2_high": self.high.as_dict(),
        }


def classify_difference(difference: float, epsilon: float) -> str:
    """Classify one second-order difference using the requested tolerance."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    return "submodular" if difference >= -epsilon else "supermodular"


def _pairwise_gamma(rows: Sequence[Mapping[str, Any]], epsilon: float) -> float | None:
    """Return a conservative empirical pairwise diminishing-returns ratio.

    This is deliberately called *pairwise* gamma: it is not the global
    submodularity ratio over arbitrary sets.  Only cases with a positive
    enhanced-state marginal constrain the ratio.  A non-positive base-state
    marginal against a positive enhanced marginal yields zero.
    """
    ratios: list[float] = []
    for row in rows:
        marginal_a = float(row["marginal_gain_s_a"])
        marginal_b = float(row["marginal_gain_s_b"])
        if marginal_b > epsilon:
            ratios.append(max(0.0, min(1.0, marginal_a / marginal_b)))
    return min(ratios) if ratios else None


HEADLINE_TEST = "epsilon test: submodular iff second_order_difference >= -epsilon"
SECONDARY_TEST = "strict sign test: negative iff second_order_difference < 0, epsilon ignored"
RESOLUTION_TEST = "resolution test: a cell carries information iff |second_order_difference| > epsilon"


def _epsilon_for(row: Mapping[str, Any], epsilon: float) -> float:
    """Return the tolerance a row was classified against.

    Rows written by this module carry their own tolerance, so a record cannot
    be read without seeing what its verdict was compared to.  The argument is
    the fallback for rows assembled by hand in a test.
    """
    return float(row.get("epsilon_used", epsilon))


def _tally(rows: Sequence[Mapping[str, Any]], epsilon: float) -> dict[str, Any]:
    """Return the three tests over one group of rows, each labelled by its test."""
    total = len(rows)
    submodular = sum(row["classification"] == "submodular" for row in rows)
    # Classification and this count now come from the same test.  They used to
    # not: the column showed the epsilon test while the headline counter showed
    # the strict sign test, so one run reported 100% submodular directly above
    # 15.62% supermodular.
    supermodular = total - submodular
    strict_negative = sum(float(row["second_order_difference"]) < 0.0 for row in rows)
    # Every epsilon-test violation is also strictly negative, so the difference
    # is exactly the strict negatives the tolerance forgives.
    inside_epsilon_band = strict_negative - supermodular
    resolved = sum(
        abs(float(row["second_order_difference"])) > _epsilon_for(row, epsilon) for row in rows
    )
    resolved_submodular = sum(
        float(row["second_order_difference"]) > _epsilon_for(row, epsilon) for row in rows
    )
    return {
        "total": total,
        # Headline, the epsilon test.  This is the Classification column.
        "submodular": submodular,
        "supermodular": supermodular,
        "submodular_rate": submodular / total if total else 0.0,
        "supermodular_rate": supermodular / total if total else 0.0,
        # Secondary, the strict sign test.  Reported for continuity with the
        # pre-epsilon literature; it is not the classification.
        "strict_negative": strict_negative,
        "strict_negative_rate": strict_negative / total if total else 0.0,
        "inside_epsilon_band": inside_epsilon_band,
        # Resolution.  A cell inside epsilon is evidence of nothing, and in
        # particular is not evidence of submodularity.
        "resolved": resolved,
        "resolved_rate": resolved / total if total else 0.0,
        "unresolved": total - resolved,
        "resolved_submodular": resolved_submodular,
        "resolved_supermodular": resolved - resolved_submodular,
    }


def summarize_interactions(
    rows: Sequence[Mapping[str, Any]],
    epsilon: float | Mapping[str, float] | EpsilonPolicy,
    effect_floor: "EffectFloor | None" = None,
    band: "EpsilonBand | None" = None,
) -> dict[str, Any]:
    """Aggregate the gates, the classifications, and the robustness band.

    Four distinct quantities are reported side by side, each labelled with the
    test that produced it: the epsilon test is the headline and matches the
    Classification column, the strict sign test is secondary, the resolution
    test says how many cells had enough signal to classify at all, and the
    effect gate says how many of those an allocator would act on.
    """
    tolerance = EpsilonPolicy.coerce(epsilon)
    default_epsilon = tolerance.default()
    overall = _tally(rows, default_epsilon)
    total = overall["total"]

    negative_marginals = sum(
        float(row[key]) < -_epsilon_for(row, default_epsilon)
        for row in rows
        for key in ("marginal_gain_s_a", "marginal_gain_s_b")
    )
    denominator = 2 * total

    by_modality: dict[str, Any] = {}
    for modality in MODALITY_LABELS:
        subset = [row for row in rows if row["modality"] == modality]
        if not subset:
            continue
        modality_epsilon = tolerance.for_modality(modality)
        by_modality[modality] = {
            **_tally(subset, modality_epsilon),
            "epsilon_ppl": modality_epsilon,
            "pairwise_gamma": _pairwise_gamma(subset, modality_epsilon),
            **_gate_tally(subset),
        }

    gates = _gate_tally(rows)
    gate_by_modality = {modality: _gate_tally(
        [row for row in rows if row["modality"] == modality]
    ) for modality in by_modality}

    # DECISION.md Amendment 2 A. The branch is called on "point" and only on
    # "point"; the other two readings are a robustness band.
    band_report: dict[str, Any] | None = None
    if band is not None and rows and "band" in rows[0]:
        band_report = {"binding_reading": "point", "readings": {}}
        for name, _ in band.readings():
            statistical = [bool(row["band"][name]["gate_statistical"]) for row in rows]
            both = [
                s and bool(row.get("gate_effect", True)) for s, row in zip(statistical, rows)
            ]
            band_report["readings"][name] = {
                "resolved_statistical": sum(statistical),
                "resolved_statistical_rate": sum(statistical) / total if total else 0.0,
                "resolved_both_gates": sum(both),
                "resolved_both_gates_rate": sum(both) / total if total else 0.0,
                "by_modality": {
                    modality: {
                        "total": sum(1 for row in rows if row["modality"] == modality),
                        "resolved_statistical": sum(
                            bool(row["band"][name]["gate_statistical"])
                            for row in rows
                            if row["modality"] == modality
                        ),
                        "resolved_both_gates": sum(
                            bool(row["band"][name]["gate_statistical"])
                            and bool(row.get("gate_effect", True))
                            for row in rows
                            if row["modality"] == modality
                        ),
                    }
                    for modality in by_modality
                },
            }
        readings = band_report["readings"]
        verdicts = {
            name: reading["resolved_both_gates_rate"] >= 0.50 for name, reading in readings.items()
        }
        band_report["dead_branch_fires"] = {
            name: not fires for name, fires in verdicts.items()
        }
        band_report["verdict_stable_across_band"] = len(set(verdicts.values())) == 1
        band_report["note"] = (
            "the verdict is the same at the point estimate and at both bounds, which is "
            "stronger than the point estimate alone"
            if band_report["verdict_stable_across_band"]
            else "the verdict FLIPS across the band; this is a finding about the precision "
            "of the floor and must be reported as one, not resolved by picking a bound"
        )

    core_cross = by_modality.get(WEIGHT_GIVEN_KV)
    core_submodular_rate = (
        float(core_cross["submodular_rate"]) if core_cross else overall["submodular_rate"]
    )
    core_supermodular_rate = (
        float(core_cross["supermodular_rate"]) if core_cross else overall["supermodular_rate"]
    )
    decision_violation_rate = max(overall["supermodular_rate"], core_supermodular_rate)
    decision_submodular_rate = min(overall["submodular_rate"], core_submodular_rate)
    # The branch turns on cells that pass BOTH gates when an effect floor is in
    # force, not on statistical resolution alone.
    branch_rate = gates["counts_toward_branch_rate"] if effect_floor is not None else overall["resolved_rate"]

    if not total:
        verdict = "INCONCLUSIVE / NO INTERACTIONS"
        action = "No paper decision: run at least one interaction."
    elif branch_rate < 0.50:
        verdict = "UNRESOLVED / BELOW THE DECISION THRESHOLD"
        action = (
            f"{gates['counts_toward_branch'] if effect_floor is not None else overall['resolved']}"
            f" of {total} cells pass every gate the branch turns on. Under half; see the Dead "
            "branch of DECISION.md."
        )
    elif decision_violation_rate > 0.30:
        verdict = "SUPERMODULAR / SYNERGY-DOMINATED"
        action = (
            "Pivot to an empirical interaction paper; the >30% supermodularity criterion is "
            "met on the epsilon test."
        )
    elif decision_submodular_rate > 0.70:
        verdict = "NEAR-SUBMODULAR / WEAKLY SUBMODULAR"
        action = (
            "Proceed with the method-paper hypothesis, then estimate a formal submodularity "
            "ratio before claiming greedy bounds."
        )
    else:
        verdict = "BORDERLINE / INCONCLUSIVE"
        action = "Increase the sample size before choosing a paper direction; epsilon is fixed."

    return {
        "total_interactions": total,
        "tests": {
            "headline": HEADLINE_TEST,
            "secondary": SECONDARY_TEST,
            "resolution": RESOLUTION_TEST,
            "effect": (
                effect_floor.as_dict()["rule"] if effect_floor is not None else "not in force"
            ),
        },
        "epsilon_ppl": tolerance.as_dict(),
        "headline_epsilon_test": {
            "submodular_pairs": overall["submodular"],
            "submodular_rate": overall["submodular_rate"],
            "supermodular_pairs": overall["supermodular"],
            "supermodular_rate": overall["supermodular_rate"],
        },
        "secondary_strict_sign_test": {
            "strict_negative_pairs": overall["strict_negative"],
            "strict_negative_rate": overall["strict_negative_rate"],
            "inside_epsilon_band_pairs": overall["inside_epsilon_band"],
        },
        "resolution": {
            "resolved_pairs": overall["resolved"],
            "resolved_rate": overall["resolved_rate"],
            "unresolved_pairs": overall["unresolved"],
            "resolved_submodular_pairs": overall["resolved_submodular"],
            "resolved_supermodular_pairs": overall["resolved_supermodular"],
        },
        "gates": gates,
        "gates_by_modality": gate_by_modality,
        "band": band_report,
        "submodular_pairs": overall["submodular"],
        "submodular_rate": overall["submodular_rate"],
        "supermodular_pairs": overall["supermodular"],
        "supermodular_rate": overall["supermodular_rate"],
        "strict_negative_pairs": overall["strict_negative"],
        "strict_negative_rate": overall["strict_negative_rate"],
        "epsilon_band_pairs": overall["inside_epsilon_band"],
        "resolved_pairs": overall["resolved"],
        "resolved_rate": overall["resolved_rate"],
        "branch_rate": branch_rate,
        "negative_marginals": negative_marginals,
        "total_marginals": denominator,
        "monotonicity_violation_rate": negative_marginals / denominator if denominator else 0.0,
        "pairwise_gamma": _pairwise_gamma(rows, default_epsilon),
        "by_modality": by_modality,
        "decision_scope": {
            "rule": (
                "the branch turns on cells passing every gate in force; sub/supermodular "
                "rates are the conservative maximum over overall and W|KV on the epsilon test"
            ),
            "submodular_rate": decision_submodular_rate,
            "supermodular_rate": decision_violation_rate,
            "resolved_rate": overall["resolved_rate"],
            "branch_rate": branch_rate,
        },
        "verdict": verdict,
        "action": action,
    }


def _gate_tally(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return both gates and the reasons cells failed, never pooled into one count."""
    total = len(rows)
    if not total:
        return {"total": 0}
    statistical = sum(bool(row.get("gate_statistical", row.get("resolved", False))) for row in rows)
    has_effect = any("gate_effect" in row for row in rows)
    e1 = sum(bool(row.get("gate_effect_e1", False)) for row in rows)
    e2 = sum(bool(row.get("gate_effect_e2", False)) for row in rows)
    effect = sum(bool(row.get("gate_effect", False)) for row in rows)
    counts = sum(bool(row.get("counts_toward_branch", False)) for row in rows)
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in row.get("failure_reasons", []):
            reasons[reason] = reasons.get(reason, 0) + 1
    relative = [
        float(row["relative_interaction"])
        for row in rows
        if math.isfinite(float(row.get("relative_interaction", float("inf"))))
    ]
    return {
        "total": total,
        "gate_statistical": statistical,
        "gate_statistical_rate": statistical / total,
        "effect_gate_in_force": has_effect,
        "gate_effect_e1": e1 if has_effect else None,
        "gate_effect_e2": e2 if has_effect else None,
        "gate_effect": effect if has_effect else None,
        "gate_effect_rate": effect / total if has_effect else None,
        "counts_toward_branch": counts,
        "counts_toward_branch_rate": counts / total,
        "failure_reasons": reasons,
        # How close the matrix sits to the phi boundary, so a reader can see it
        # rather than infer it from a pass/fail count.
        "relative_interaction": _spread_summary(relative),
    }


def _spread_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return a compact distribution summary for a list of ratios."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _cache_namespace(utility: UtilityFunction) -> str:
    """Fingerprint settings omitted from UtilityFunction's raw-plan cache key."""
    payload = {
        "schema": "submokv.submodularity.v1",
        "ground_set": utility.ground_set.signature,
        "model": utility.ground_set.model.name,
        "model_source": str(getattr(utility.model.config, "_name_or_path", "unknown")),
        "calibration": utility.store.spec.describe(),
        "tokenizer_source": utility.store.tokenizer_source,
        "policy": utility.controller.policy.describe(),
        "protocol": utility.controller.protocol.describe(),
        "master_store": utility.quantizer.master.describe(),
        "batch_size": utility.batch_size,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _plan_cache_key(namespace: str, plan: DiagnosticPlan) -> str:
    payload = json.dumps(
        {
            "namespace": namespace,
            "weight_bits": plan.weight_bits,
            "kv_retention": plan.kv_retention,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "submod.v1." + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_submodularity_diagnostic(
    utility: UtilityFunction,
    interactions: Sequence[InteractionSpec],
    *,
    epsilon: float | Mapping[str, float] | EpsilonPolicy | EpsilonBand = 0.01,
    effect_floor: EffectFloor | None = None,
    fidelity: str = CHEAP,
    subsamples: Sequence[int] = (0,),
    expected_subsamples_per_cell: int | None = None,
    use_cache: bool = True,
    allow_outside_ground_set: bool = False,
    progress: Callable[[int, int, DiagnosticPlan, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate an interaction matrix and return a JSON-serializable report.

    Every cell is measured on ``subsamples`` calibration draws and reported as a
    mean and a spread, never as a single number.  The tier ladders come from
    ``utility.ground_set`` and from nowhere else, so a probe of a tier the
    allocator cannot buy raises ``OutsideGroundSetError`` unless
    ``allow_outside_ground_set`` is set.

    ``expected_subsamples_per_cell`` is the ``k`` the tolerance was built with.
    A mismatch raises, because epsilon is ``sigma2/sqrt(k)`` and comparing a
    mean of a different number of draws against it is comparing two different
    quantities.  DECISION.md Amendment 1 requires that to fail loudly.
    """
    draws = tuple(int(index) for index in subsamples)
    if not draws:
        raise ValueError("at least one calibration subsample is required")
    if len(set(draws)) != len(draws):
        raise ValueError(f"calibration subsamples must be distinct, got {draws}")
    if expected_subsamples_per_cell is not None and len(draws) != expected_subsamples_per_cell:
        raise ValueError(
            f"the tolerance was built for k={expected_subsamples_per_cell} subsamples per cell "
            f"but this run uses {len(draws)} ({list(draws)}). Epsilon is sigma2/sqrt(k), so the "
            "two are not comparable. Re-derive the tolerance or change the subsample count."
        )
    available = utility.store.max_subsamples(fidelity)
    beyond = [index for index in draws if not 0 <= index < available]
    if beyond:
        raise ValueError(
            f"the calibration pool holds {available} non-overlapping subsample(s) at fidelity "
            f"{fidelity!r}; subsample(s) {beyond} do not exist"
        )

    band = epsilon if isinstance(epsilon, EpsilonBand) else None
    tolerance = band.point if band is not None else EpsilonPolicy.coerce(epsilon)
    ground = utility.ground_set
    ladders = TierLadders.from_ground_set(ground)
    num_layers = ground.model.num_hidden_layers
    num_experts = ground.model.num_experts
    matrix = tuple(interactions)
    for spec in matrix:
        validate_test_layers(
            tuple(dict.fromkeys((spec.target_layer, spec.conditioning_layer))), num_layers
        )

    outside = tuple(spec for spec in matrix if not spec.in_ground_set(ladders))
    if outside and not allow_outside_ground_set:
        moves = sorted({
            move.label
            for spec in outside
            for move in (spec.target, spec.conditioning)
            if not ladders.contains(move)
        })
        raise OutsideGroundSetError(
            f"{len(outside)} of {len(matrix)} interactions move along tier(s) the ground set "
            f"does not contain: {', '.join(moves)}. The ground set's ladders are "
            f"weight={list(ladders.weight_tiers)} and kv={list(ladders.kv_tiers)}. Either fix "
            "the requested upgrades or pass allow_outside_ground_set=True to record the rows "
            "labelled in_ground_set: false."
        )

    namespace = _cache_namespace(utility)
    evaluations_before = utility.evaluation_count
    reference_plan = DiagnosticPlan.full(num_layers)

    squares = {spec.interaction_id: four_state_plans(spec, num_layers) for spec in matrix}
    # Plan-major, subsample-minor, exactly as the floor measures: installing an
    # allocation costs a checkpoint read and a requantization of every expert in
    # each changed layer, while moving to the next subsample costs nothing.
    # Order cannot touch the numbers, because F is memoized on (state, draw) and
    # each square is still assembled from four corners read on one shared draw.
    ordered: list[DiagnosticPlan] = [reference_plan]
    for square in squares.values():
        for plan in (square.s_a, square.s_a_union_j, square.s_b, square.s_b_union_j):
            if plan not in ordered:
                ordered.append(plan)

    measured: dict[tuple[DiagnosticPlan, int], Any] = {}
    try:
        for position, plan in enumerate(ordered, start=1):
            for draw in draws:
                if progress is not None:
                    progress(position, len(ordered), plan, draw)
                measured[(plan, draw)] = utility.evaluate_plan(
                    _plan_cache_key(namespace, plan),
                    plan.expanded_weight_plan(num_experts),
                    plan.expanded_retention_plan(),
                    fidelity=fidelity,
                    subsample=draw,
                    use_cache=use_cache,
                )
    finally:
        # Leave the shared model unmodified even if an evaluation fails.
        utility.quantizer.restore()
        utility.controller.set_uniform_retention(1.0)

    rows: list[dict[str, Any]] = []
    for spec in matrix:
        plans = squares[spec.interaction_id]
        per_draw: list[dict[str, Any]] = []
        for draw in draws:
            ppl_a = measured[(plans.s_a, draw)].perplexity
            ppl_aj = measured[(plans.s_a_union_j, draw)].perplexity
            ppl_b = measured[(plans.s_b, draw)].perplexity
            ppl_bj = measured[(plans.s_b_union_j, draw)].perplexity
            marginal_a = ppl_a - ppl_aj
            marginal_b = ppl_b - ppl_bj
            per_draw.append(
                {
                    "subsample": draw,
                    "reference_perplexity": measured[(reference_plan, draw)].perplexity,
                    "ppl_s_a": ppl_a,
                    "ppl_s_a_union_j": ppl_aj,
                    "ppl_s_b": ppl_b,
                    "ppl_s_b_union_j": ppl_bj,
                    "marginal_gain_s_a": marginal_a,
                    "marginal_gain_s_b": marginal_b,
                    "second_order_difference": marginal_a - marginal_b,
                }
            )

        differences = [entry["second_order_difference"] for entry in per_draw]
        marginals_a = [entry["marginal_gain_s_a"] for entry in per_draw]
        marginals_b = [entry["marginal_gain_s_b"] for entry in per_draw]
        mean_difference = statistics.fmean(differences)
        mean_a, mean_b = statistics.fmean(marginals_a), statistics.fmean(marginals_b)
        spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
        # m is the larger of the two marginals: if conditioning collapses an
        # upgrade's value then m(S_B) is near zero and that is precisely the
        # interaction being hunted, so the minimum would discard it.
        magnitude = max(abs(mean_a), abs(mean_b))

        row_epsilon = tolerance.for_modality(spec.modality)
        statistical = abs(mean_difference) > row_epsilon
        row: dict[str, Any] = {
            **spec.as_dict(ladders),
            "subsamples": list(draws),
            "per_subsample": per_draw,
            "marginal_gain_s_a": mean_a,
            "marginal_gain_s_b": mean_b,
            "marginal_gain_s_a_stdev": (
                statistics.stdev(marginals_a) if len(marginals_a) > 1 else 0.0
            ),
            "marginal_gain_s_b_stdev": (
                statistics.stdev(marginals_b) if len(marginals_b) > 1 else 0.0
            ),
            "second_order_difference": mean_difference,
            "second_order_stdev": spread,
            "second_order_stderr": spread / math.sqrt(len(differences)) if spread else 0.0,
            "sign_agreement": sum(
                1 for value in differences if (value > 0) == (mean_difference > 0)
            ),
            "epsilon_used": row_epsilon,
            "epsilon_source": tolerance.source,
            "gate_statistical": statistical,
            "classification": classify_difference(mean_difference, row_epsilon),
            "classification_test": HEADLINE_TEST,
            "strict_negative": mean_difference < 0.0,
            "within_epsilon": abs(mean_difference) <= row_epsilon,
            "resolved": statistical,
        }
        if effect_floor is not None:
            row.update(effect_floor.evaluate(spec.target.kind, magnitude, mean_difference))
            row["counts_toward_branch"] = statistical and bool(row["gate_effect"])
            reasons = list(row["effect_failure_reasons"])
            if not statistical:
                reasons.insert(0, "statistical_inside_epsilon")
            row["failure_reasons"] = reasons
        else:
            row["effect_magnitude_m"] = magnitude
            row["counts_toward_branch"] = statistical
            row["failure_reasons"] = [] if statistical else ["statistical_inside_epsilon"]
        if band is not None:
            row["band"] = {
                name: {
                    "epsilon": policy.for_modality(spec.modality),
                    "gate_statistical": abs(mean_difference)
                    > policy.for_modality(spec.modality),
                }
                for name, policy in band.readings()
            }
        rows.append(row)

    summary = summarize_interactions(rows, tolerance, effect_floor=effect_floor, band=band)
    reference = measured[(reference_plan, draws[0])]
    return {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": {
            "utility": "F(S) = PPL(full 16-bit/100% KV) - PPL(S) = -DeltaPPL",
            "second_order_difference": (
                "D = [PPL(S_A) - PPL(S_A+j)] - [PPL(S_B) - PPL(S_B+j)], the reference cancels"
            ),
            "cell_value": "the mean of D over the calibration subsamples, with its spread",
            "headline_test": HEADLINE_TEST,
            "secondary_test": SECONDARY_TEST,
            "resolution_test": RESOLUTION_TEST,
            "move_asymmetry": (
                "TARGET moves are adjacent ladder steps; CONDITIONING moves are the large "
                "bottom-to-top move. This is deliberate: adjacent conditioning would put S_A "
                "and S_B so close together that nothing could resolve for reasons unrelated "
                "to the model. Do not read both sides as adjacent."
            ),
        },
        "model": {
            "name": ground.model.name,
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
        },
        "calibration": {
            **utility.store.spec.describe(),
            "fidelity": fidelity,
            "subsamples": list(draws),
            "subsamples_per_cell": len(draws),
            "subsamples_available": available,
        },
        "test_layers": sorted(
            {spec.target_layer for spec in matrix}
            | {spec.conditioning_layer for spec in matrix}
        ),
        "layer_coverage": {
            "layers_sampled": len(
                {spec.target_layer for spec in matrix}
                | {spec.conditioning_layer for spec in matrix}
            ),
            "layers_in_model": num_layers,
            "note": (
                "the matrix samples a subset of the model's layers; results do not "
                "generalize to unsampled layers without an assumption"
            ),
        },
        "ground_set_signature": ground.signature,
        "tier_ladders": ladders.describe(),
        "ground_set_scope": {
            "interactions_in_ground_set": len(matrix) - len(outside),
            "interactions_outside_ground_set": len(outside),
            "allow_outside_ground_set": allow_outside_ground_set,
            "outside_interaction_ids": [spec.interaction_id for spec in outside],
        },
        "epsilon_ppl": tolerance.as_dict(),
        "epsilon_band": band.as_dict() if band is not None else None,
        "effect_gate": effect_floor.as_dict() if effect_floor is not None else None,
        "reference": {
            **reference.as_dict(),
            "utility": 0.0,
            "allocation_delta_from_full": reference_plan.compact_dict(),
        },
        "interactions": rows,
        "summary": summary,
        "execution": {
            "matrix_interactions": len(matrix),
            "unique_allocation_states": len(ordered),
            "state_subsample_pairs": len(measured),
            "new_model_evaluations": utility.evaluation_count - evaluations_before,
            "evaluation_cache_enabled": use_cache,
            "cache_namespace": namespace,
            "setup": utility.describe(),
        },
    }


def _epsilon_policy_from_record(value: Any) -> EpsilonPolicy:
    """Rebuild the tolerance from what a record wrote down.

    Records written before the tolerance became per-modality hold a bare number
    under ``epsilon_ppl``; both shapes are accepted so an old shard can still be
    merged and re-summarized against the tolerance it actually ran with.
    """
    if isinstance(value, Mapping) and "binding_reading" in value:
        # A band record: the binding reading is the one a verdict was called on.
        return _epsilon_policy_from_record(value[value["binding_reading"]])
    if isinstance(value, Mapping) and "by_modality" in value:
        return EpsilonPolicy.from_mapping(
            value["by_modality"], str(value.get("source", "recorded"))
        )
    if isinstance(value, Mapping):
        return EpsilonPolicy.from_mapping(value, "recorded")
    return EpsilonPolicy.uniform(float(value), "recorded scalar")


def default_floor_specs(
    ladders: TierLadders,
    target_layer: int,
    conditioning_layer: int,
    weight_move: TierUpgrade | None = None,
    kv_move: TierUpgrade | None = None,
    weight_conditioning: TierUpgrade | None = None,
    kv_conditioning: TierUpgrade | None = None,
    modalities: Sequence[str] | None = None,
) -> tuple[InteractionSpec, ...]:
    """Return one square per modality, on the ground set's bottom adjacent steps.

    The floor has to be measured on the same kind of square the run it
    calibrates will evaluate, so the target moves default to adjacent ladder
    steps rather than bottom-to-top jumps, and every modality gets its own spec
    because the weight and KV scales differ by roughly a factor of seven.

    The conditioning moves default to the target moves.  They are separable
    because a matrix may condition on a large move while targeting an adjacent
    one, and ``sigma2`` plausibly scales with how much the conditioning move
    perturbs the model.

    Note that with the conditioning moves left at their defaults the two
    cross-component squares are the *same* square: the mixed second difference
    is symmetric in its two components, so W|KV and KV|W then yield the
    identical D.  Giving conditioning its own moves separates them.
    """
    if target_layer == conditioning_layer:
        raise ValueError(
            "the intra-component squares need two distinct layers, got "
            f"{target_layer} for both"
        )
    weight = weight_move or ladders.adjacent(WEIGHT)[0]
    kv = kv_move or ladders.adjacent(KV)[0]
    weight_context = weight_conditioning or weight
    kv_context = kv_conditioning or kv
    if weight_context.kind != WEIGHT or kv_context.kind != KV:
        raise ValueError("a conditioning move must be on the ladder it conditions")
    built = {
        WEIGHT_TO_WEIGHT: InteractionSpec(target_layer, weight, conditioning_layer, weight_context),
        KV_TO_KV: InteractionSpec(target_layer, kv, conditioning_layer, kv_context),
        WEIGHT_GIVEN_KV: InteractionSpec(target_layer, weight, target_layer, kv_context),
        KV_GIVEN_WEIGHT: InteractionSpec(target_layer, kv, target_layer, weight_context),
    }
    if modalities is None:
        selected = tuple(built)
    else:
        selected = tuple(str(name) for name in modalities)
        unknown = [name for name in selected if name not in built]
        if unknown:
            raise ValueError(f"unknown modality/modalities {unknown}; expected {sorted(built)}")
    return tuple(built[name] for name in selected)


def compare_conditioning_floor(
    baseline: Mapping[str, Any],
    check: Mapping[str, Any],
    *,
    binding: bool = True,
    question: str = "conditioning",
) -> dict[str, Any]:
    """Compare one square's sigma2 against another square's confidence interval.

    With ``binding`` set this is DECISION.md Amendment 2 C: does the measured
    floor survive Task C conditioning?

    The floor was measured with adjacent conditioning.  The matrix conditions on
    the large bottom-to-top move.  If ``sigma2`` scales with how much the
    conditioning move perturbs the model, a floor measured at the smaller
    perturbation understates the tolerance the matrix needs.

    The rule was fixed before either number existed: a checked ``sigma2`` inside
    the baseline square's 95% interval leaves the floor standing, and one
    outside it means Task B is not finished.
    """
    rows: list[dict[str, Any]] = []
    baseline_by_modality = baseline["by_modality"]
    for modality, entry in sorted(check["by_modality"].items()):
        reference = baseline_by_modality.get(modality)
        if reference is None:
            raise ValueError(
                f"the baseline floor measured no {modality!r} square, so there is nothing to "
                "compare the conditioning check against"
            )
        interval = reference["second_order_stdev_pooled_ci"]
        if interval is None:
            raise ValueError(
                f"the baseline {modality!r} square carries no confidence interval; the rule in "
                "Amendment 2 C is stated against one"
            )
        measured = float(entry["second_order_stdev_pooled"])
        low, high = float(interval["low"]), float(interval["high"])
        inside = low <= measured <= high
        rows.append(
            {
                "modality": modality,
                "label": MODALITY_LABELS[modality],
                "baseline_spec_ids": list(reference["spec_ids"]),
                "check_spec_ids": list(entry["spec_ids"]),
                "baseline_sigma2": float(reference["second_order_stdev_pooled"]),
                "baseline_ci_low": low,
                "baseline_ci_high": high,
                "baseline_ci_confidence": float(interval.get("confidence", 0.95)),
                "check_sigma2": measured,
                "ratio_to_baseline": (
                    measured / float(reference["second_order_stdev_pooled"])
                    if reference["second_order_stdev_pooled"]
                    else float("nan")
                ),
                "inside_baseline_interval": inside,
            }
        )
    if not rows:
        raise ValueError("the conditioning check measured no squares")

    stands = all(row["inside_baseline_interval"] for row in rows)
    outside = [row["modality"] for row in rows if not row["inside_baseline_interval"]]
    if binding:
        rule = (
            "DECISION.md Amendment 2 C: a checked sigma2 inside the baseline square's 95% "
            "interval leaves the measured floor standing; outside it, Task B is not finished "
            "and the floor is re-derived at the matrix conditioning before anything is "
            "classified"
        )
        verdict = (
            "FLOOR STANDS - Task C proceeds on the measured epsilon"
            if stands
            else "FLOOR DOES NOT STAND - re-derive at the matrix conditioning before Task C"
        )
    else:
        # No decision rule was fixed in advance for this comparison, so it
        # cannot select a verdict. It is reported as context and nothing else.
        rule = (
            "CONTEXT ONLY. No decision rule was declared in advance for this comparison, so "
            "it does not select a verdict and no branch turns on it. It is reported so a "
            "reader can see how far the floor travels."
        )
        verdict = (
            "CONTEXT: consistent with the reference square"
            if stands
            else "CONTEXT: outside the reference square's interval - report as a limitation"
        )
    return {
        "question": question,
        "binding": binding,
        "rule": rule,
        "decided_before_measurement": binding,
        "comparisons": rows,
        "floor_stands": stands if binding else None,
        "consistent": stands,
        "modalities_outside_interval": outside,
        "verdict": verdict,
    }


def format_conditioning_check(comparison: Mapping[str, Any]) -> str:
    """Render the Amendment 2 C conditioning check as a terminal table."""
    width = 104
    binding = bool(comparison.get("binding", True))
    title = (
        "TASK B CONDITIONING CHECK (DECISION.md Amendment 2 C)"
        if binding
        else f"TASK B {str(comparison.get('question', '')).upper()} CHECK - CONTEXT ONLY"
    )
    preamble = (
        [
            "Rule fixed before either number existed:",
            "  sigma2 at the matrix conditioning INSIDE the adjacent-conditioning 95% interval",
            "  -> the measured floor stands and Task C proceeds on it.",
            "  OUTSIDE -> Task B is not finished; re-derive the floor before classifying anything.",
        ]
        if binding
        else [
            "No decision rule was declared in advance for this comparison.",
            "  It selects no verdict and no branch turns on it. Reported as context.",
        ]
    )
    lines = [
        "=" * width,
        title.center(width),
        "=" * width,
        *preamble,
        "-" * width,
        f"{'modality':<10} {'baseline s2':>12} {'95% interval':>22} {'checked s2':>12} "
        f"{'ratio':>7}  outcome",
        "-" * width,
    ]
    for row in comparison["comparisons"]:
        interval = f"[{row['baseline_ci_low']:.5f}, {row['baseline_ci_high']:.5f}]"
        lines.append(
            f"{row['label']:<10} {row['baseline_sigma2']:>12.5f} {interval:>22} "
            f"{row['check_sigma2']:>12.5f} {row['ratio_to_baseline']:>7.2f}x  "
            + ("inside" if row["inside_baseline_interval"] else "OUTSIDE")
        )
    lines.extend(["-" * width, f"VERDICT: {comparison['verdict']}", "=" * width])
    return "\n".join(lines)


def _stdev_confidence_interval(
    stdev: float, count: int, confidence: float = 0.95
) -> dict[str, float] | None:
    """Return a chi-square interval for a stdev estimated from ``count`` draws.

    A floor measured from six numbers is itself uncertain by roughly a factor of
    two, and a classification made against it inherits that. The interval is
    reported so nobody reads the point estimate as exact.
    """
    if count < 2:
        return None
    try:
        from scipy.stats import chi2  # type: ignore
    except ImportError:
        return None
    degrees = count - 1
    tail = (1.0 - confidence) / 2.0
    lower = float(chi2.ppf(1.0 - tail, degrees))
    upper = float(chi2.ppf(tail, degrees))
    return {
        "confidence": confidence,
        "low": stdev * math.sqrt(degrees / lower),
        "high": stdev * math.sqrt(degrees / upper),
    }


def _pooled_stdev(groups: Sequence[Sequence[float]]) -> tuple[float, int]:
    """Return the within-group pooled stdev and its degrees of freedom.

    Specs of one modality may sit at different mean interaction strengths, so
    their values are not pooled directly. Only the spread *within* each spec is
    noise; pooling the variances keeps that distinction.
    """
    numerator = 0.0
    degrees = 0
    for values in groups:
        if len(values) < 2:
            continue
        numerator += (len(values) - 1) * statistics.variance(values)
        degrees += len(values) - 1
    if degrees == 0:
        return 0.0, 0
    return math.sqrt(numerator / degrees), degrees


def second_order_noise_floor(
    utility: UtilityFunction,
    interactions: Sequence[InteractionSpec],
    *,
    num_subsamples: int = 6,
    fidelity: str = CHEAP,
    use_cache: bool = True,
    allow_outside_ground_set: bool = False,
    progress: Callable[[int, int, DiagnosticPlan, int], None] | None = None,
) -> dict[str, Any]:
    """Measure the spread of the second-order difference across calibration draws.

    The quantity the submodularity test classifies is not a single perplexity
    and not a single delta.  It is

        D = [F(S_A + j) - F(S_A)] - [F(S_B + j) - F(S_B)]

    and the floor D must clear is the spread of D itself.  Writing F as
    ``PPL(reference) - PPL(S)`` shows the reference cancelling out of every
    term, so D is measured from the four corner perplexities alone and no
    full-precision evaluation is needed:

        D = [PPL(S_A) - PPL(S_A + j)] - [PPL(S_B) - PPL(S_B + j)]

    All four corners of one square are read from the same calibration
    subsample, so the pairing that makes this measurable is preserved; the
    spread is then taken over independent, non-overlapping subsamples.
    """
    if num_subsamples < 5:
        raise ValueError(
            f"a floor from fewer than five draws is not a spread, got {num_subsamples}"
        )
    available = utility.store.max_subsamples(fidelity)
    if available < num_subsamples:
        raise ValueError(
            f"the calibration pool holds {available} non-overlapping subsample(s) at fidelity "
            f"{fidelity!r}, which is fewer than the {num_subsamples} requested"
        )

    ground = utility.ground_set
    ladders = TierLadders.from_ground_set(ground)
    num_layers = ground.model.num_hidden_layers
    num_experts = ground.model.num_experts
    matrix = tuple(interactions)
    if not matrix:
        raise ValueError("at least one interaction spec is required")
    for spec in matrix:
        validate_test_layers(
            tuple(dict.fromkeys((spec.target_layer, spec.conditioning_layer))), num_layers
        )
    outside = tuple(spec for spec in matrix if not spec.in_ground_set(ladders))
    if outside and not allow_outside_ground_set:
        raise OutsideGroundSetError(
            f"{len(outside)} of {len(matrix)} floor specs move along tiers outside the ground "
            f"set's ladders weight={list(ladders.weight_tiers)} kv={list(ladders.kv_tiers)}. "
            "A floor measured off the ladder does not calibrate a test run on it."
        )

    namespace = _cache_namespace(utility)
    evaluations_before = utility.evaluation_count

    squares = {spec.interaction_id: four_state_plans(spec, num_layers) for spec in matrix}
    # Evaluation order is plan-major, subsample-minor. Every subsample of one
    # allocation is scored before the next allocation is installed, because
    # installing one costs a checkpoint read and a requantization of every
    # expert in each changed layer, while changing the subsample costs nothing.
    # Order does not touch the numbers: F is deterministic and memoized on
    # (state, subsample), and each square is still assembled from four corners
    # read on one shared draw.
    ordered_plans: list[DiagnosticPlan] = []
    for square in squares.values():
        for plan in (square.s_a, square.s_a_union_j, square.s_b, square.s_b_union_j):
            if plan not in ordered_plans:
                ordered_plans.append(plan)

    measured: dict[tuple[DiagnosticPlan, int], float] = {}
    try:
        for position, plan in enumerate(ordered_plans, start=1):
            for subsample in range(num_subsamples):
                if progress is not None:
                    progress(position, len(ordered_plans), plan, subsample)
                measured[(plan, subsample)] = utility.evaluate_plan(
                    _plan_cache_key(namespace, plan),
                    plan.expanded_weight_plan(num_experts),
                    plan.expanded_retention_plan(),
                    fidelity=fidelity,
                    subsample=subsample,
                    use_cache=use_cache,
                ).perplexity
    finally:
        utility.quantizer.restore()
        utility.controller.set_uniform_retention(1.0)

    rows: list[dict[str, Any]] = []
    for spec in matrix:
        plans = squares[spec.interaction_id]
        differences: list[float] = []
        marginals_a: list[float] = []
        marginals_b: list[float] = []
        corners: list[dict[str, float]] = []
        for subsample in range(num_subsamples):
            # One subsample supplies all four corners, so the pairing that
            # cancels the calibration draw is kept intact.
            ppl_a = measured[(plans.s_a, subsample)]
            ppl_aj = measured[(plans.s_a_union_j, subsample)]
            ppl_b = measured[(plans.s_b, subsample)]
            ppl_bj = measured[(plans.s_b_union_j, subsample)]
            marginal_a = ppl_a - ppl_aj
            marginal_b = ppl_b - ppl_bj
            marginals_a.append(marginal_a)
            marginals_b.append(marginal_b)
            differences.append(marginal_a - marginal_b)
            corners.append(
                {
                    "subsample": subsample,
                    "ppl_s_a": ppl_a,
                    "ppl_s_a_union_j": ppl_aj,
                    "ppl_s_b": ppl_b,
                    "ppl_s_b_union_j": ppl_bj,
                }
            )
        stdev = statistics.stdev(differences)
        rows.append(
            {
                **spec.as_dict(ladders),
                "subsamples": list(range(num_subsamples)),
                "corner_perplexities": corners,
                "marginal_gain_s_a": marginals_a,
                "marginal_gain_s_b": marginals_b,
                "second_order_differences": differences,
                "second_order_mean": statistics.fmean(differences),
                "second_order_stdev": stdev,
                "second_order_stdev_ci": _stdev_confidence_interval(stdev, len(differences)),
                # Reported alongside so the size of the cancellation is
                # visible: a first-order spread far above the second-order
                # one is the pairing doing its job.
                "marginal_gain_s_a_stdev": statistics.stdev(marginals_a),
                "marginal_gain_s_b_stdev": statistics.stdev(marginals_b),
                "corner_perplexity_stdev": statistics.stdev(
                    [corner["ppl_s_a"] for corner in corners]
                ),
            }
        )

    by_modality: dict[str, Any] = {}
    for modality in MODALITY_LABELS:
        subset = [row for row in rows if row["modality"] == modality]
        if not subset:
            continue
        groups = [row["second_order_differences"] for row in subset]
        pooled, degrees = _pooled_stdev(groups)
        by_modality[modality] = {
            "label": MODALITY_LABELS[modality],
            "num_specs": len(subset),
            "num_values": sum(len(group) for group in groups),
            "spec_ids": [row["interaction_id"] for row in subset],
            "second_order_stdev_pooled": pooled,
            "second_order_stdev_pooled_ci": _stdev_confidence_interval(pooled, degrees + 1),
            "second_order_stdev_max": max(row["second_order_stdev"] for row in subset),
            "second_order_mean_abs": statistics.fmean(
                abs(value) for group in groups for value in group
            ),
            "first_order_stdev_max": max(
                max(row["marginal_gain_s_a_stdev"], row["marginal_gain_s_b_stdev"])
                for row in subset
            ),
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": {
            "second_order_difference": (
                "D = [PPL(S_A) - PPL(S_A+j)] - [PPL(S_B) - PPL(S_B+j)]; identical to "
                "[F(S_A+j) - F(S_A)] - [F(S_B+j) - F(S_B)] because the full-precision "
                "reference cancels"
            ),
            "floor": "stdev of D over non-overlapping calibration subsamples, per modality",
            "pairing": "all four corners of one square are read from the same subsample",
        },
        "model": {"name": ground.model.name, "num_hidden_layers": num_layers},
        "ground_set_signature": ground.signature,
        "tier_ladders": ladders.describe(),
        "calibration": {
            **utility.store.spec.describe(),
            "fidelity": fidelity,
            "sequences_per_subsample": utility.store.spec.size(fidelity),
            "num_subsamples": num_subsamples,
            "subsamples": list(range(num_subsamples)),
            "subsamples_available": available,
        },
        "specs": rows,
        "by_modality": by_modality,
        "execution": {
            "unique_allocation_states": len(ordered_plans),
            "unique_state_subsample_pairs": len(measured),
            "new_model_evaluations": utility.evaluation_count - evaluations_before,
            "evaluation_cache_enabled": use_cache,
            "cache_namespace": namespace,
            "setup": utility.describe(),
        },
    }


def epsilon_from_floor(
    floor: Mapping[str, Any],
    statistic: str = "second_order_stdev_pooled",
    subsamples_per_cell: int = 1,
) -> EpsilonPolicy:
    """Turn a measured second-order floor into the tolerance a run classifies against.

    ``subsamples_per_cell`` is how many draws the consuming run averages into
    one reported cell.  A mean of k draws has standard error sigma/sqrt(k), so
    comparing such a mean against the single-draw sigma under-resolves by that
    factor.  It defaults to 1, which is the single-draw floor DECISION.md fixed.
    """
    if subsamples_per_cell < 1:
        raise ValueError(f"subsamples_per_cell must be positive, got {subsamples_per_cell}")
    by_modality = floor["by_modality"]
    missing = sorted(set(MODALITY_LABELS) - set(by_modality))
    if missing:
        raise ValueError(
            f"the floor measured no spec for modality/modalities {missing}; every modality a "
            "run classifies must have its own measured floor"
        )
    scale = math.sqrt(subsamples_per_cell)
    values = {
        modality: float(entry[statistic]) / scale for modality, entry in by_modality.items()
    }
    source = (
        f"measured second-order floor ({statistic}"
        + (f", divided by sqrt({subsamples_per_cell})" if subsamples_per_cell > 1 else "")
        + ")"
    )
    return EpsilonPolicy.from_mapping(values, source)


def format_second_order_floor(floor: Mapping[str, Any]) -> str:
    """Render the measured floor as a terminal table."""
    width = 110
    calibration = floor["calibration"]
    lines = [
        "=" * width,
        "SUB-MoKV SECOND-ORDER NOISE FLOOR".center(width),
        "=" * width,
        (
            f"Model: {floor['model']['name']} | {calibration['sequences_per_subsample']} "
            f"sequences x {calibration['sequence_length']} tokens | "
            f"split: {calibration['calibration_split']} | "
            f"subsamples: {calibration['subsamples']}"
        ),
        f"Ladders: weight {floor['tier_ladders']['weight_tiers']} | "
        f"kv {floor['tier_ladders']['kv_tiers']}",
        "-" * width,
        f"{'Spec':<46} | {'mean D':>10} | {'stdev D':>10} | {'stdev m(A)':>10} | {'stdev PPL':>10}",
        "-" * width,
    ]
    for row in floor["specs"]:
        descriptor = f"[{MODALITY_LABELS[row['modality']]}] {row['label']}"
        lines.append(
            f"{descriptor:<46} | {_signed(row['second_order_mean'], 10)} | "
            f"{row['second_order_stdev']:>10.4f} | {row['marginal_gain_s_a_stdev']:>10.4f} | "
            f"{row['corner_perplexity_stdev']:>10.4f}"
        )
    lines.extend(["-" * width, "FLOOR PER MODALITY (this is what epsilon is set from):"])
    for modality, entry in sorted(floor["by_modality"].items()):
        interval = entry["second_order_stdev_pooled_ci"]
        span = (
            f"  95% CI [{interval['low']:.4f}, {interval['high']:.4f}]" if interval else ""
        )
        lines.append(
            f"  {entry['label']:<6} sigma2 = {entry['second_order_stdev_pooled']:.4f} PPL "
            f"from {entry['num_values']} values{span}"
        )
        lines.append(
            f"         first-order stdev {entry['first_order_stdev_max']:.4f} PPL, "
            f"so the pairing cancels a factor of "
            f"{entry['first_order_stdev_max'] / entry['second_order_stdev_pooled']:.1f}"
            if entry["second_order_stdev_pooled"] > 0
            else "         first-order stdev n/a"
        )
    lines.append("=" * width)
    return "\n".join(lines)


def _band_from_record(value: Any, subsamples_per_cell: int) -> "EpsilonBand | None":
    """Rebuild the robustness band a record wrote down."""
    if not isinstance(value, Mapping) or "binding_reading" not in value:
        return None
    return EpsilonBand(
        point=_epsilon_policy_from_record(value["point"]),
        low=_epsilon_policy_from_record(value["sigma2_low"]),
        high=_epsilon_policy_from_record(value["sigma2_high"]),
        subsamples_per_cell=int(value.get("subsamples_per_cell", subsamples_per_cell)),
        confidence=float(value.get("confidence", 0.95)),
    )


def _effect_floor_from_record(value: Any) -> "EffectFloor | None":
    """Rebuild the effect gate a record was classified against."""
    if not isinstance(value, Mapping) or "by_kind" not in value:
        return None
    return EffectFloor(
        by_kind=tuple(sorted((str(k), float(v)) for k, v in value["by_kind"].items())),
        phi=float(value["phi"]),
        multiple=float(value.get("multiple", 3.0)),
        source=str(value.get("source", "recorded")),
    )


def merge_submodularity_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge disjoint interaction shards after checking experiment identity.

    Each accelerator evaluates the same full-precision reference and a
    modulo-indexed slice of the interaction matrix.  The reference check is a
    guard against combining results from different model revisions, datasets,
    numerical environments, or subsamples.
    """
    shards = tuple(reports)
    if not shards:
        raise ValueError("no submodularity shard reports were provided")

    executions = [report.get("execution", {}) for report in shards]
    shard_counts = {int(execution.get("num_shards", 1)) for execution in executions}
    if len(shard_counts) != 1:
        raise ValueError(f"shards disagree on num_shards: {sorted(shard_counts)}")
    num_shards = shard_counts.pop()
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    shard_indices = [int(execution.get("shard", 0)) for execution in executions]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError(f"duplicate shard indices: {sorted(shard_indices)}")
    expected_shards = set(range(num_shards))
    if set(shard_indices) != expected_shards:
        missing = sorted(expected_shards - set(shard_indices))
        extra = sorted(set(shard_indices) - expected_shards)
        raise ValueError(f"incomplete shard set; missing={missing}, unexpected={extra}")

    def canonical(report: Mapping[str, Any], key: str) -> str:
        return json.dumps(report.get(key), sort_keys=True, separators=(",", ":"))

    for key in (
        "definition",
        "model",
        "calibration",
        "test_layers",
        "epsilon_ppl",
        "epsilon_band",
        "effect_gate",
        "tier_ladders",
        "ground_set_signature",
    ):
        values = {canonical(report, key) for report in shards}
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}")

    references = {
        round(float(report["reference"]["perplexity"]), 12) for report in shards
    }
    reference_nlls = {
        round(float(report["reference"]["mean_nll"]), 12) for report in shards
    }
    if len(references) != 1 or len(reference_nlls) != 1:
        raise ValueError(
            "shards disagree on the full-precision reference; do not merge results "
            "from different models, devices, or calibration samples"
        )

    orders = [tuple(execution.get("full_matrix_order", ())) for execution in executions]
    if not orders[0] or any(order != orders[0] for order in orders[1:]):
        raise ValueError("shards disagree on the full interaction matrix order")
    full_order = orders[0]
    expected_total = len(full_order)
    declared_totals = {
        int(execution.get("full_matrix_interactions", -1)) for execution in executions
    }
    if declared_totals != {expected_total}:
        raise ValueError(
            f"shards disagree on the full matrix size: {sorted(declared_totals)} "
            f"versus order length {expected_total}"
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    for report in shards:
        for row in report.get("interactions", []):
            interaction_id = str(row["interaction_id"])
            if interaction_id in by_id:
                raise ValueError(f"interaction {interaction_id!r} appears in more than one shard")
            by_id[interaction_id] = row
    missing_interactions = [interaction_id for interaction_id in full_order if interaction_id not in by_id]
    unexpected_interactions = sorted(set(by_id) - set(full_order))
    if missing_interactions or unexpected_interactions:
        raise ValueError(
            "incomplete interaction union; "
            f"missing={missing_interactions}, unexpected={unexpected_interactions}"
        )
    rows = [deepcopy(by_id[interaction_id]) for interaction_id in full_order]

    first = shards[0]
    epsilon = _epsilon_policy_from_record(first["epsilon_ppl"])
    subsamples_per_cell = int(first.get("calibration", {}).get("subsamples_per_cell", 1))
    band = _band_from_record(first.get("epsilon_band"), subsamples_per_cell)
    effect = _effect_floor_from_record(first.get("effect_gate"))
    merged = {
        "schema_version": first.get("schema_version", 1),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definition": deepcopy(first["definition"]),
        "model": deepcopy(first["model"]),
        "calibration": deepcopy(first["calibration"]),
        "test_layers": deepcopy(first["test_layers"]),
        "ground_set_signature": first.get("ground_set_signature"),
        "tier_ladders": deepcopy(first.get("tier_ladders")),
        "ground_set_scope": {
            "interactions_in_ground_set": sum(
                bool(row.get("in_ground_set", True)) for row in rows
            ),
            "interactions_outside_ground_set": sum(
                not bool(row.get("in_ground_set", True)) for row in rows
            ),
            "outside_interaction_ids": [
                str(row["interaction_id"])
                for row in rows
                if not bool(row.get("in_ground_set", True))
            ],
        },
        "epsilon_ppl": deepcopy(first["epsilon_ppl"]),
        "epsilon_band": deepcopy(first.get("epsilon_band")),
        "effect_gate": deepcopy(first.get("effect_gate")),
        "layer_coverage": deepcopy(first.get("layer_coverage")),
        "reference": deepcopy(first["reference"]),
        "interactions": rows,
        "summary": summarize_interactions(rows, epsilon, effect_floor=effect, band=band),
        "execution": {
            "matrix_interactions": expected_total,
            "full_matrix_interactions": expected_total,
            "merged_from_shards": sorted(shard_indices),
            "num_shards": num_shards,
            "new_model_evaluations": sum(
                int(execution.get("new_model_evaluations", 0)) for execution in executions
            ),
            "shard_unique_allocation_states_sum": sum(
                int(execution.get("unique_allocation_states", 0)) for execution in executions
            ),
            "evaluation_cache_enabled": all(
                bool(execution.get("evaluation_cache_enabled", False))
                for execution in executions
            ),
            "full_matrix_order": list(full_order),
            "shards": [deepcopy(execution) for execution in executions],
        },
    }
    return merged


def _signed(value: float, width: int = 10) -> str:
    if not math.isfinite(value):
        return f"{value:+g}".rjust(width)
    return f"{value:+.4f}".rjust(width)


def format_diagnostic_report(report: Mapping[str, Any]) -> str:
    """Render the requested terminal table from a diagnostic report."""
    width = 132
    model = report["model"]
    calibration = report["calibration"]
    summary = report["summary"]
    lines = [
        "=" * width,
        "SUB-MoKV SUBMODULARITY DIAGNOSTIC REPORT".center(width),
        "=" * width,
    ]
    execution = report.get("execution", {})
    if int(execution.get("num_shards", 1)) > 1 and "merged_from_shards" not in execution:
        lines.append(
            (
                f"PARTIAL SHARD {int(execution.get('shard', 0)) + 1}/"
                f"{execution['num_shards']} — VERDICT IS PRELIMINARY UNTIL MERGED"
            ).center(width)
        )

    epsilon = _epsilon_policy_from_record(report["epsilon_ppl"])
    if epsilon.is_uniform():
        epsilon_text = f"{epsilon.default():.5g} PPL"
    else:
        epsilon_text = ", ".join(
            f"{MODALITY_LABELS[modality]} {value:.5g}"
            for modality, value in sorted(epsilon.by_modality)
        )
    ladders = report.get("tier_ladders")
    scope = report.get("ground_set_scope", {})
    coverage = report.get("layer_coverage", {})
    draws = calibration.get("subsamples", [calibration.get("subsample")])
    lines.extend(
        [
            (
                f"Model: {model['name']} | {calibration.get('cheap_sequences', '?')} sequences "
                f"x {calibration['sequence_length']} tokens | "
                f"split: {calibration.get('calibration_split', '?')} | "
                f"subsamples per cell: {draws}"
            ),
            f"Epsilon ({epsilon.source}): {epsilon_text}",
        ]
    )
    effect = report.get("effect_gate")
    if effect:
        by_kind = ", ".join(f"{k} {v:.5g}" for k, v in sorted(effect["by_kind"].items()))
        lines.append(
            f"Effect gate: E1 m > [{by_kind}] PPL | E2 |D| >= {effect['phi']:.2f} x m "
            f"({effect['source']})"
        )
    if ladders is not None:
        lines.append(
            f"Ground-set ladders: weight {ladders['weight_tiers']} | kv {ladders['kv_tiers']}"
        )
    if coverage:
        lines.append(
            f"Layer coverage: {coverage['layers_sampled']} of "
            f"{coverage['layers_in_model']} layers sampled"
        )
    definition = report.get("definition", {})
    if "move_asymmetry" in definition:
        lines.append("!! TARGET moves are ADJACENT ladder steps; CONDITIONING moves are LARGE")
        lines.append("   bottom-to-top moves. The two sides are NOT symmetric. This is by design:")
        lines.append("   adjacent conditioning would leave S_A and S_B too close to resolve.")
    if int(scope.get("interactions_outside_ground_set", 0)):
        lines.append(
            f"!! {scope['interactions_outside_ground_set']} of "
            f"{summary['total_interactions']} interactions move along tiers OUTSIDE the ground "
            "set and are marked in_ground_set: false"
        )
    lines.extend(
        [
            "-" * width,
            "Columns: Classification (epsilon test) | stat = statistical gate | "
            "E1, E2 = effect gate clauses.",
            "A cell counts toward the branch only if stat, E1 and E2 all pass.",
            "-" * width,
            (
                f"{'Layer Pair / Modality':<47} | {'mean D':>10} | {'sd D':>8} | "
                f"{'m':>9} | {'|D|/m':>7} | {'stat':>5} | {'E1':>3} | {'E2':>3} | Classification"
            ),
            "-" * width,
        ]
    )
    for row in report["interactions"]:
        descriptor = f"[{MODALITY_LABELS[row['modality']]}] {row['label']}"
        if not row.get("in_ground_set", True):
            descriptor = "(off-ladder) " + descriptor
        resolved = bool(
            row.get(
                "gate_statistical",
                abs(float(row["second_order_difference"])) > _epsilon_for(row, epsilon.default()),
            )
        )
        if not resolved:
            classification = "unresolved (inside epsilon)"
        elif not row.get("gate_effect", True):
            classification = "immaterial (effect gate)"
        elif row["classification"] == "submodular":
            classification = "Submodular"
        else:
            classification = "SUPERMODULAR *"
        magnitude = float(row.get("effect_magnitude_m", float("nan")))
        relative = float(row.get("relative_interaction", float("nan")))
        def mark(value: Any) -> str:
            if value is None:
                return "  -"
            return "  y" if value else "  n"
        lines.append(
            f"{descriptor:<47} | {_signed(row['second_order_difference'], 10)} | "
            f"{float(row.get('second_order_stdev', 0.0)):8.4f} | {magnitude:9.4f} | "
            f"{relative:7.3f} | {mark(resolved):>5} | {mark(row.get('gate_effect_e1')):>3} | "
            f"{mark(row.get('gate_effect_e2')):>3} | {classification}"
        )
    lines.extend(["-" * width, "SUMMARY:"])
    total = summary["total_interactions"]
    headline = summary["headline_epsilon_test"]
    secondary = summary["secondary_strict_sign_test"]
    resolution = summary["resolution"]
    gates = summary.get("gates", {})
    lines.extend(
        [
            f"- Total Interactions Tested: {total}",
            "",
            "  HEADLINE - epsilon test (this is the Classification column):",
            (
                f"  - Submodular (diff >= -epsilon): {headline['submodular_pairs']}/{total} "
                f"({100.0 * headline['submodular_rate']:.2f}%)"
            ),
            (
                f"  - Supermodular (diff <  -epsilon): {headline['supermodular_pairs']}/{total} "
                f"({100.0 * headline['supermodular_rate']:.2f}%)"
            ),
            "",
            "  GATE 1, STATISTICAL - can the difference be told from zero:",
            (
                f"  - Resolved (|D| > epsilon): {resolution['resolved_pairs']}/{total} "
                f"({100.0 * resolution['resolved_rate']:.2f}%) "
                f"= {resolution['resolved_submodular_pairs']} submodular, "
                f"{resolution['resolved_supermodular_pairs']} supermodular"
            ),
            (
                f"  - Unresolved (|D| <= epsilon): {resolution['unresolved_pairs']}/{total}; "
                "evidence of neither, and NOT counted as a result"
            ),
        ]
    )
    if gates.get("effect_gate_in_force"):
        lines.extend(
            [
                "",
                "  GATE 2, EFFECT - would an allocator act on it:",
                (
                    f"  - E1 passed (target worth buying):  {gates['gate_effect_e1']}/{total}"
                ),
                (
                    f"  - E2 passed (interaction material): {gates['gate_effect_e2']}/{total}"
                ),
                (
                    f"  - Both effect clauses: {gates['gate_effect']}/{total} "
                    f"({100.0 * gates['gate_effect_rate']:.2f}%)"
                ),
                "",
                "  BRANCH DECIDES ON BOTH GATES:",
                (
                    f"  - Cells passing every gate: {gates['counts_toward_branch']}/{total} "
                    f"({100.0 * gates['counts_toward_branch_rate']:.2f}%)"
                ),
                "  - Failure reasons (a cell may fail more than one, never pooled):",
            ]
        )
        for reason, count in sorted(gates.get("failure_reasons", {}).items()):
            lines.append(f"      {reason:<38} {count}/{total}")
        spread = gates.get("relative_interaction", {})
        if spread.get("count"):
            phi = float(report.get("effect_gate", {}).get("phi", 0.0))
            lines.append(
                f"  - |D|/m distribution vs phi={phi:.2f}: min {spread['min']:.3f}, "
                f"p25 {spread['p25']:.3f}, median {spread['median']:.3f}, "
                f"p75 {spread['p75']:.3f}, p90 {spread['p90']:.3f}, max {spread['max']:.3f}"
            )
    band = summary.get("band")
    if band:
        lines.extend(
            [
                "",
                "  ROBUSTNESS BAND (DECISION.md Amendment 2 A) - the branch is called on 'point'",
                "  only; the bounds are shown beside it and never select the verdict:",
            ]
        )
        for name, reading in band["readings"].items():
            marker = "  <- BINDING" if name == band["binding_reading"] else ""
            lines.append(
                f"      {name:<12} statistical {reading['resolved_statistical']}/{total}, "
                f"both gates {reading['resolved_both_gates']}/{total} "
                f"({100.0 * reading['resolved_both_gates_rate']:.2f}%){marker}"
            )
        lines.append(f"      -> {band['note']}")
    lines.extend(
        [
            "",
            "  SECONDARY - strict sign test, epsilon ignored (not the classification):",
            (
                f"  - Strictly negative (D < 0): {secondary['strict_negative_pairs']}/{total} "
                f"({100.0 * secondary['strict_negative_rate']:.2f}%), of which "
                f"{secondary['inside_epsilon_band_pairs']} sit inside epsilon"
            ),
            "",
            (
                f"- Negative Marginal Gains: {summary['negative_marginals']}/"
                f"{summary['total_marginals']}"
            ),
        ]
    )
    gamma = summary["pairwise_gamma"]
    core = summary["by_modality"].get(WEIGHT_GIVEN_KV)
    if core is not None:
        lines.append(
            f"- Core W|KV, epsilon test: {core['supermodular']}/{core['total']} supermodular; "
            f"{core['resolved']}/{core['total']} resolved (epsilon {core['epsilon_ppl']:.5g})"
        )
    lines.append(
        "- Empirical Pairwise Gamma: " + ("n/a" if gamma is None else f"{gamma:.4f}")
    )
    lines.extend(
        [
            "",
            f"DECISION VERDICT: {summary['verdict']}",
            f"-> Action: {summary['action']}",
            "-> Paper branch rules are fixed in DECISION.md; this verdict does not amend them.",
            "=" * width,
        ]
    )
    return "\n".join(lines)


def matrix_as_dict(
    interactions: Iterable[InteractionSpec], ladders: TierLadders | None = None
) -> list[dict[str, Any]]:
    """Return a compact JSON-ready matrix, useful for a no-model dry run."""
    return [spec.as_dict(ladders) for spec in interactions]
