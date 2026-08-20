"""F(S): the perplexity based utility, cached and deterministic.

F is a deterministic function of the selected set. The calibration sequences,
their order, the sequence length, and the batch order are all fixed, nothing is
sampled per call, and every value is memoized on a canonical hash of the
allocation. The calibration split that F reads and the split that results are
reported on are different splits, and the two are checked to share no window.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .ground_set import Allocation, GroundSet
from .kvcache import RetentionController, score_sequence
from .quantize import ExpertQuantizer, find_expert_modules

CHEAP = "cheap"
FULL = "full"


def total_in_float64(values: torch.Tensor) -> float:
    """Return the sum of a tensor as a Python float, accumulated in float64.

    The move to the host and the widening to float64 are separate steps on
    purpose. Asking for both at once, as values.to("cpu", torch.float64),
    silently returns garbage when the source is on MPS: that backend has no
    float64, and the conversion is attempted before the copy. It does not
    raise, so the wrong number reaches the results table looking ordinary.
    """
    return float(values.detach().cpu().to(torch.float64).sum().item())


@dataclass(frozen=True)
class CalibrationSpec:
    """Which sequences F reads and which sequences results are reported on."""

    dataset: str = "Salesforce/wikitext"
    subset: str = "wikitext-2-raw-v1"
    calibration_split: str = "train"
    evaluation_split: str = "test"
    sequence_length: int = 4096
    cheap_sequences: int = 64
    full_sequences: int = 256
    seed: int = 0

    def describe(self) -> dict[str, Any]:
        """Return the calibration settings for result records."""
        return {
            "dataset": self.dataset,
            "subset": self.subset,
            "calibration_split": self.calibration_split,
            "evaluation_split": self.evaluation_split,
            "sequence_length": self.sequence_length,
            "cheap_sequences": self.cheap_sequences,
            "full_sequences": self.full_sequences,
            "calibration_seed": self.seed,
        }

    def size(self, fidelity: str) -> int:
        """Return how many sequences a fidelity level reads."""
        if fidelity == CHEAP:
            return self.cheap_sequences
        if fidelity == FULL:
            return self.full_sequences
        raise ValueError(f"unknown fidelity {fidelity!r}; choose {CHEAP!r} or {FULL!r}")


class SequenceStore:
    """Fixed token windows drawn once from each split.

    A split is tokenized whole, cut into windows that do not overlap, and then
    permuted once with the calibration seed. Every later request reads a slice
    of that fixed order, so no call samples anything.
    """

    def __init__(self, tokenizer: Any, spec: CalibrationSpec, cache_dir: str | Path = "cache") -> None:
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.calibration = self._windows(tokenizer, spec.calibration_split)
        self.evaluation = self._windows(tokenizer, spec.evaluation_split)
        self.assert_splits_disjoint()

    @classmethod
    def from_windows(
        cls,
        calibration: torch.Tensor,
        evaluation: torch.Tensor,
        spec: CalibrationSpec,
    ) -> "SequenceStore":
        """Build a store from token windows that are already prepared.

        The windows are taken in the order given, so the caller owns the
        ordering that makes F deterministic.
        """
        store = cls.__new__(cls)
        store.spec = spec
        store.cache_dir = Path("cache")
        store.calibration = calibration
        store.evaluation = evaluation
        store.assert_splits_disjoint()
        return store

    def _windows(self, tokenizer: Any, split: str) -> torch.Tensor:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{self.spec.dataset}/{self.spec.subset}/{split}/{self.spec.sequence_length}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"windows_{split}_{digest}.pt"
        if path.exists():
            windows = torch.load(path)
        else:
            from datasets import load_dataset

            data = load_dataset(self.spec.dataset, self.spec.subset, split=split)
            text = "\n\n".join(data["text"])
            ids = torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long)
            length = self.spec.sequence_length
            count = ids.numel() // length
            if count == 0:
                raise ValueError(f"split {split!r} is shorter than one window of {length} tokens")
            windows = ids[: count * length].reshape(count, length)
            torch.save(windows, path)
        order = torch.randperm(windows.shape[0], generator=torch.Generator().manual_seed(self.spec.seed))
        return windows[order].contiguous()

    def assert_splits_disjoint(self) -> None:
        """Check that no calibration window also appears in the evaluation pool.

        If the solver picks its allocation using the data it is scored on, the
        result means nothing, so this is checked rather than assumed.
        """
        if self.spec.calibration_split == self.spec.evaluation_split:
            raise ValueError(
                f"calibration and evaluation both read split {self.spec.calibration_split!r}"
            )
        calibration = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in self.calibration}
        evaluation = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in self.evaluation}
        shared = calibration & evaluation
        if shared:
            raise ValueError(
                f"{len(shared)} windows appear in both the calibration and evaluation pools"
            )

    def subsample(self, fidelity: str, index: int = 0) -> torch.Tensor:
        """Return the index-th block of calibration sequences at a fidelity level.

        Blocks do not overlap, so repeated evaluations of one allocation across
        indices measure how much the answer moves with the calibration draw.
        """
        size = self.spec.size(fidelity)
        start = index * size
        stop = start + size
        if stop > self.calibration.shape[0]:
            raise ValueError(
                f"calibration pool holds {self.calibration.shape[0]} windows, which is not "
                f"enough for subsample {index} of {size} sequences"
            )
        return self.calibration[start:stop]

    def max_subsamples(self, fidelity: str) -> int:
        """Return how many blocks of a fidelity level the calibration pool holds."""
        return self.calibration.shape[0] // self.spec.size(fidelity)

    def describe(self) -> dict[str, Any]:
        """Return the pool sizes for result records."""
        return {
            **self.spec.describe(),
            "calibration_windows": int(self.calibration.shape[0]),
            "evaluation_windows": int(self.evaluation.shape[0]),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """One perplexity measurement of one allocation."""

    perplexity: float
    mean_nll: float
    num_sequences: int
    num_scored_tokens: int
    fidelity: str
    subsample: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return the measurement as a plain dictionary."""
        return {
            "perplexity": self.perplexity,
            "mean_nll": self.mean_nll,
            "num_sequences": self.num_sequences,
            "num_scored_tokens": self.num_scored_tokens,
            "fidelity": self.fidelity,
            "subsample": self.subsample,
            "seconds": self.seconds,
        }


class UtilityFunction:
    """Perplexity of an allocation, memoized, and the utility built from it.

    F(S) is the perplexity the base state loses relative to the allocation S, so
    F(base) is zero by construction and F rises as perplexity falls.
    """

    def __init__(
        self,
        model: Any,
        ground_set: GroundSet,
        quantizer: ExpertQuantizer,
        controller: RetentionController,
        store: SequenceStore,
        device: str | torch.device = "cpu",
        cache_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.ground_set = ground_set
        self.quantizer = quantizer
        self.controller = controller
        self.store = store
        self.device = torch.device(device)
        self.batch_size = ground_set.kv.batch_size
        if store.spec.sequence_length != ground_set.kv.context_length:
            raise ValueError(
                f"calibration sequences are {store.spec.sequence_length} tokens but the KV "
                f"accounting declares a context of {ground_set.kv.context_length}; a retention "
                "ratio would mean two different things"
            )
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()
        self.evaluation_count = 0
        self._base_cache: dict[tuple[str, int], float] = {}

    def apply_allocation(self, allocation: Allocation) -> None:
        """Push an allocation into the quantizer and the retention controller."""
        self.quantizer.set_plan(self.ground_set.weight_bits_by_expert(allocation))
        self.controller.set_retention(self.ground_set.kv_retention_by_layer(allocation))

    def _key(self, allocation: Allocation, fidelity: str, subsample: int) -> str:
        return f"{allocation.canonical_hash()}|{fidelity}{self.store.spec.size(fidelity)}|{subsample}"

    def _load_cache(self) -> None:
        if self.cache_path is not None and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _save_cache(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))

    def evaluate(
        self,
        allocation: Allocation,
        fidelity: str = CHEAP,
        subsample: int = 0,
        use_cache: bool = True,
    ) -> EvaluationResult:
        """Return the perplexity of an allocation on a fixed calibration subsample."""
        key = self._key(allocation, fidelity, subsample)
        if use_cache and key in self._cache:
            return EvaluationResult(**self._cache[key])
        self.apply_allocation(allocation)
        return self._measure(key, fidelity, subsample)

    def evaluate_plan(
        self,
        key: str,
        weight_bits: Mapping[int, Mapping[int, int]],
        retention: Mapping[int, float],
        fidelity: str = CHEAP,
        subsample: int = 0,
        use_cache: bool = True,
    ) -> EvaluationResult:
        """Return the perplexity of a raw bit width and retention plan.

        Diagnostics reach below the ground set, for instance to drop a single
        expert to a low bit width when the ground set puts a whole layer on one
        ladder. The caller supplies a key, which must identify the plan.
        """
        full_key = f"{key}|{fidelity}{self.store.spec.size(fidelity)}|{subsample}"
        if use_cache and full_key in self._cache:
            return EvaluationResult(**self._cache[full_key])
        self.quantizer.set_plan(weight_bits)
        self.controller.set_retention(retention)
        return self._measure(full_key, fidelity, subsample)

    def _measure(self, key: str, fidelity: str, subsample: int) -> EvaluationResult:
        """Run the calibration subsample through the model as it is currently configured."""
        sequences = self.store.subsample(fidelity, subsample)
        start = time.perf_counter()
        total_nll = 0.0
        total_tokens = 0
        for begin in range(0, sequences.shape[0], self.batch_size):
            batch = sequences[begin : begin + self.batch_size].to(self.device)
            if batch.shape[0] != self.batch_size:
                # A short final batch would hold a different amount of cache
                # than the accounting charges for, so it is dropped.
                break
            losses = score_sequence(self.model, batch, self.controller)
            total_nll += total_in_float64(losses)
            total_tokens += int(losses.numel())
        seconds = time.perf_counter() - start

        mean_nll = total_nll / total_tokens
        result = EvaluationResult(
            perplexity=math.exp(mean_nll),
            mean_nll=mean_nll,
            num_sequences=int(sequences.shape[0]),
            num_scored_tokens=total_tokens,
            fidelity=fidelity,
            subsample=subsample,
            seconds=seconds,
        )
        self.evaluation_count += 1
        self._cache[key] = result.as_dict()
        self._save_cache()
        return result

    def base_perplexity(self, fidelity: str = CHEAP, subsample: int = 0) -> float:
        """Return the perplexity of the base state, which is the zero point of F."""
        cached = self._base_cache.get((fidelity, subsample))
        if cached is None:
            cached = self.evaluate(self.ground_set.base_allocation(), fidelity, subsample).perplexity
            self._base_cache[(fidelity, subsample)] = cached
        return cached

    def utility(
        self, allocation: Allocation, fidelity: str = CHEAP, subsample: int = 0
    ) -> float:
        """Return F(S), the perplexity the base state gives up relative to this allocation."""
        base = self.base_perplexity(fidelity, subsample)
        return base - self.evaluate(allocation, fidelity, subsample).perplexity

    def marginal(
        self,
        allocation: Allocation,
        increment_id: str,
        fidelity: str = CHEAP,
        subsample: int = 0,
    ) -> float:
        """Return the gain from adding one increment to an allocation."""
        moved = self.ground_set.apply(allocation, self.ground_set.increment(increment_id))
        before = self.evaluate(allocation, fidelity, subsample).perplexity
        after = self.evaluate(moved, fidelity, subsample).perplexity
        return before - after

    def noise_floor(
        self,
        allocation: Allocation,
        num_subsamples: int = 8,
        fidelity: str = CHEAP,
    ) -> dict[str, Any]:
        """Measure how far one allocation's perplexity moves with the calibration draw.

        Every gain smaller than this is not a gain. The subsamples do not
        overlap, so this is the spread over independent draws rather than over
        repeated runs of the same data.
        """
        available = self.store.max_subsamples(fidelity)
        count = min(num_subsamples, available)
        if count < 2:
            raise ValueError(
                f"the calibration pool holds {available} subsample(s) at fidelity {fidelity!r}, "
                "which is not enough to measure a spread"
            )
        values = [self.evaluate(allocation, fidelity, index).perplexity for index in range(count)]
        losses = [self.evaluate(allocation, fidelity, index).mean_nll for index in range(count)]
        return {
            "fidelity": fidelity,
            "num_subsamples": count,
            "perplexity_mean": statistics.fmean(values),
            "perplexity_stdev": statistics.stdev(values),
            "nll_mean": statistics.fmean(losses),
            "nll_stdev": statistics.stdev(losses),
            "perplexity_values": values,
        }

    def verify_determinism(self, allocation: Allocation, fidelity: str = CHEAP) -> dict[str, Any]:
        """Evaluate one allocation twice with the cache bypassed and report the difference.

        F must be a deterministic function of the selected set. If two runs of
        the same allocation disagree, every marginal gain is contaminated.
        """
        first = self.evaluate(allocation, fidelity, 0, use_cache=False)
        second = self.evaluate(allocation, fidelity, 0, use_cache=False)
        return {
            "first_perplexity": first.perplexity,
            "second_perplexity": second.perplexity,
            "absolute_difference": abs(first.perplexity - second.perplexity),
            "identical": first.perplexity == second.perplexity,
        }

    def describe(self) -> dict[str, Any]:
        """Return the utility settings for result records."""
        return {
            **self.store.describe(),
            **self.controller.describe(),
            **self.quantizer.describe(),
            "batch_size": self.batch_size,
            "device": str(self.device),
            "evaluation_count": self.evaluation_count,
        }


def build_utility(
    config: Mapping[str, Any],
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
    cache_dir: str | Path = "cache",
    shard: int = 0,
    num_shards: int = 1,
) -> tuple[Any, "UtilityFunction"]:
    """Load the model and assemble the utility function described by a config.

    The memoization file is named for the device and dtype that produced it.
    Different accelerators give different floating point results for the same
    allocation, so a cache carried from one machine to another would serve
    numbers the current machine did not produce. The shard is in the name too,
    so workers running slices of one experiment side by side never write over
    each other.

    Returns the model and the utility, so a caller that needs the model itself
    does not have to reach through the utility for it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .kvcache import EvaluationProtocol, RetentionController, build_policy
    from .memory import QuantSpec
    from .quantize import CheckpointMasterStore, ExpertQuantizer, MemoryMasterStore

    import os

    ground_set = GroundSet.from_config(config)
    runtime = dict(config.get("runtime", {}))
    source = str(model_path) if model_path is not None else ground_set.model.name
    resolved_device = torch.device(device or runtime.get("device", "cpu"))
    dtype = getattr(torch, str(runtime.get("dtype", "bfloat16")))

    policy = build_policy(config["retention"])
    implementation = runtime.get("attn_implementation") or (
        "eager" if policy.needs_attention_scores else "sdpa"
    )

    if runtime.get("deterministic", True):
        # F must be a deterministic function of the selected set. OLMoE's expert
        # forward ends in index_add_, which on CUDA accumulates with atomics in
        # an order that varies between runs, so the same allocation would score
        # differently each time and every marginal gain would carry that noise.
        if resolved_device.type == "cuda":
            workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if workspace not in (":4096:8", ":16:8"):
                raise RuntimeError(
                    "set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment before starting "
                    "the process; cuBLAS needs it for reproducible matmuls and it cannot be "
                    f"set afterwards (found {workspace!r})"
                )
        torch.use_deterministic_algorithms(True)

    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=dtype, attn_implementation=implementation
    )
    model = model.to(resolved_device).eval()
    tokenizer = AutoTokenizer.from_pretrained(source)

    quant = QuantSpec.from_mapping(config.get("quant", {}))
    if runtime.get("master_store", "checkpoint") == "checkpoint":
        if model_path is None:
            raise ValueError("the checkpoint master store needs a local snapshot path")
        quantizer = ExpertQuantizer(model, quant, master=CheckpointMasterStore(model_path))
    else:
        quantizer = ExpertQuantizer(
            model, quant, master=MemoryMasterStore(find_expert_modules(model))
        )
    # A wrong repacking order would quantize the right bytes into the wrong
    # places while every shape still matched, so this fails loudly and early.
    quantizer.verify_master(layer=0, experts=(0, 1))
    quantizer.verify_master(layer=ground_set.model.num_hidden_layers - 1, experts=(0,))

    protocol_settings = dict(config.get("protocol", {}))
    protocol = EvaluationProtocol(
        name=protocol_settings.get("name", "prefill_evict_decode"),
        prefill_tokens=int(protocol_settings.get("prefill_tokens", 3072)),
        chunk_size=int(protocol_settings.get("chunk_size", 256)),
    )
    controller = RetentionController(model, policy, ground_set.kv, protocol).attach()

    spec = CalibrationSpec(**config.get("calibration", {}))
    store = SequenceStore(tokenizer, spec, cache_dir)

    tag = f"{resolved_device.type}_{str(dtype).replace('torch.', '')}"
    if num_shards > 1:
        tag = f"{tag}_s{shard}of{num_shards}"
    utility = UtilityFunction(
        model=model,
        ground_set=ground_set,
        quantizer=quantizer,
        controller=controller,
        store=store,
        device=resolved_device,
        cache_path=Path(cache_dir) / f"utility_{ground_set.signature}_{tag}.json",
    )
    return model, utility
