"""Flexible batch creation strategy for GLiNER training.

This module provides composable, external controls over how training batches
are formed — including grouping and meta-sampling of cardinality/prevalence —
**without modifying any GLiNER source code**.

It also provides :class:`ValidationSnapshot` and :func:`build_validation_snapshot`
for fixed, reproducible validation prompts that enable F1-based checkpointing.

Supported modes (composable):

* **Default** — ``BatchStrategy()`` delegates 100% to GLiNER's original logic.
* **group_by** — batches are constrained so every instance in a batch shares
  the same group value (e.g. ``language``).  No other side-effect.
* **meta_sampling** — per-instance entity-type cardinality and prevalence
  are randomised via the Symmetric Bounding Trick (uniform K, 50 %
  expected prevalence).

Typical usage::

    from batch_strategy import BatchStrategy, patch_train_model, build_validation_snapshot

    strategy = BatchStrategy(group_by="language", meta_sampling=True)
    val_snapshot = build_validation_snapshot(val_data, model.config, seed=42)
    patch_train_model(model, strategy, val_snapshot=val_snapshot)
    model.train_model(train_data, val_data, ...)
"""

from __future__ import annotations

import logging
import os
import random
import array
import functools
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
from torch.utils.data import Sampler

from gliner.data_processing.utils import get_negatives, make_mapping
from gliner.training import Trainer as _BaseTrainer


# ---------------------------------------------------------------------------
# Monkey-patch BaseNEREvaluator to handle Span objects
# ---------------------------------------------------------------------------
from gliner.evaluation.evaluator import BaseNEREvaluator

def _patched_get_predictions(self, ents):
    all_ents = []
    for ent in ents:
        if hasattr(ent, "entity_type"):
            if ent.entity_type == "unknown":
                continue
            all_ents.append([ent.entity_type, (ent.start, ent.end)])
        elif isinstance(ent, dict):
            if ent["label"] == "unknown":
                continue
            all_ents.append([ent["label"], (ent["start"], ent["end"])])
        else:
            if ent[2] == "unknown":
                continue
            all_ents.append([ent[2], (ent[0], ent[1])])
    return all_ents

BaseNEREvaluator.get_predictions = _patched_get_predictions

# ---------------------------------------------------------------------------
# Monkey-patch Decoders to gracefully handle padding classes (KeyError: 3)
# ---------------------------------------------------------------------------
from gliner.decoding.decoder import SpanDecoder, SpanGenerativeDecoder, SpanRelexDecoder, Span

def _patched_build_span_tuple(
    self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs=None
):
    ent_type = id_to_class.get(class_idx + 1, "unknown")
    if ent_type == "unknown":
        return Span(start=start, end=start+width, entity_type="unknown", score=float("-inf"), class_probs=class_probs)
    return SpanDecoder._build_span_tuple_orig(
        self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs
    )

def _patched_build_span_tuple_generative(
    self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs=None
):
    ent_type = id_to_class.get(class_idx + 1, "unknown")
    if ent_type == "unknown":
        return Span(start=start, end=start+width, entity_type="unknown", score=float("-inf"), class_probs=class_probs)
    return SpanGenerativeDecoder._build_span_tuple_orig(
        self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs
    )

def _patched_build_span_tuple_relex(
    self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs=None
):
    ent_type = id_to_class.get(class_idx + 1, "unknown")
    if ent_type == "unknown":
        return Span(start=start, end=start+width, entity_type="unknown", score=float("-inf"), class_probs=class_probs)
    return SpanRelexDecoder._build_span_tuple_orig(
        self, start, width, class_idx, flat_idx, score, id_to_class, span_label_map, class_probs
    )


def _patch_decoders():
    if not getattr(SpanDecoder, "_build_span_tuple_orig", None):
        SpanDecoder._build_span_tuple_orig = SpanDecoder._build_span_tuple
        SpanDecoder._build_span_tuple = _patched_build_span_tuple

    if not getattr(SpanGenerativeDecoder, "_build_span_tuple_orig", None):
        SpanGenerativeDecoder._build_span_tuple_orig = SpanGenerativeDecoder._build_span_tuple
        SpanGenerativeDecoder._build_span_tuple = _patched_build_span_tuple_generative

    if not getattr(SpanRelexDecoder, "_build_span_tuple_orig", None):
        SpanRelexDecoder._build_span_tuple_orig = SpanRelexDecoder._build_span_tuple
        SpanRelexDecoder._build_span_tuple = _patched_build_span_tuple_relex

_patch_decoders()


# ---------------------------------------------------------------------------
# BatchStrategy dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatchStrategy:
    """Configurable batch-creation policy for GLiNER training.

    Every default reproduces the existing GLiNER behaviour exactly, so
    ``BatchStrategy()`` is a strict no-op.

    Attributes:
        group_by: Key in each data instance to group by (e.g. ``"language"``).
            When set, entire batches share the same group value.  This only
            affects the DataLoader sampler and does **not** alter any GLiNER
            label-generation logic.
        meta_sampling: When ``True``, per-instance cardinality and prevalence
            are randomised via the Symmetric Bounding Trick (uniform K,
            50 % expected prevalence).
        dynamic_max_types_mult: Multiplier for the per-instance ceiling on
            total entity types during meta-sampling.  ``None`` → default 2.
        shuffle_types: Shuffle the final entity-type list per instance.
        seed: Optional RNG seed for reproducibility.
    """

    group_by: Optional[str] = None
    meta_sampling: bool = False
    dynamic_max_types_mult: Optional[int] = None
    shuffle_types: bool = True
    seed: Optional[int] = None

    @property
    def is_default(self) -> bool:
        """Return ``True`` when every label-mapping setting matches the GLiNER default.

        ``group_by`` is intentionally excluded: it only affects the DataLoader
        sampler and does not alter the GLiNER label-generation math.
        When ``is_default`` is True, :class:`StrategyCollator` delegates
        **100 %** to the original GLiNER ``batch_generate_class_mappings``.
        """
        return not self.meta_sampling and self.shuffle_types


# ---------------------------------------------------------------------------
# Meta-sampling helpers (Symmetric Bounding Trick)
# ---------------------------------------------------------------------------

def _meta_sample_counts(
    n_available_pos: int,
    n_available_neg: int,
    max_types: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """Meta-sample cardinality and prevalence for a single instance.

    Uses the "Symmetric Bounding Trick" to uniformly sample the total
    cardinality (K), while anchoring the expected prevalence at exactly
    50 % for any given K (up to K = 2 × n_available_pos).

    Unlike compensating by shrinking the total set size (which distorts
    the distribution of K), this approach guarantees a perfectly uniform
    distribution of set sizes.  It achieves balance by artificially
    raising the lower bound of sampled positive items to perfectly mirror
    the natural upper bound constraint.

    The ``dynamic_max_types_mult`` (default 2) ensures that
    ``max_types = 2 × P`` in the typical case, which is exactly the
    regime where the symmetric bound ``K − P`` mirrors ``min(P, K)``
    and the 50 % prevalence anchor holds for every K.

    Returns:
        (n_pos, n_neg)
    """
    # -- Determine absolute maximum set size (K) --
    max_k = min(max_types, n_available_pos + n_available_neg)

    if max_k <= 0:
        return 0, 0

    # -- Draw total cardinality K uniformly --
    k = rng.randint(1, max_k)

    # -- Determine the upper bound for positive items --
    max_k_pos = min(n_available_pos, k)

    # -- Determine the symmetric lower bound (the Bounding Trick) --
    # k - n_available_pos: anchors the expected prevalence at exactly 50 %
    # k - n_available_neg: physical limit if we lack negatives to fill K
    raw_min_k_pos = max(0, k - n_available_neg, k - n_available_pos)

    # Clamp to max_k_pos to prevent mathematically invalid ranges
    # (safely handles the ceiling when k > 2 × n_available_pos)
    min_k_pos = min(max_k_pos, raw_min_k_pos)

    # -- Sample final counts --
    n_pos = rng.randint(min_k_pos, max_k_pos)
    n_neg = k - n_pos

    # Ensure at least one type overall (empty prompt is invalid)
    if n_pos == 0 and n_neg == 0:
        n_neg = min(1, n_available_neg)

    return n_pos, n_neg


# ---------------------------------------------------------------------------
# Core mapping builder
# ---------------------------------------------------------------------------

def build_class_mappings(
    batch_list: List[Dict],
    config: Any,
    strategy: BatchStrategy,
    rng: random.Random,
    key: str = "ner",
) -> Tuple[List[Dict[str, int]], List[Dict[int, str]], List[Dict[str, float]]]:
    """Build per-instance class ↔ ID mappings for the meta-sampling strategy.

    This replaces ``BaseProcessor.batch_generate_class_mappings`` when
    ``strategy.meta_sampling`` is active.  It is **never** called when
    ``is_default`` is True — in that case :class:`StrategyCollator` delegates
    100 % to GLiNER's original ``batch_generate_class_mappings``.

    Args:
        batch_list: Raw batch examples (list of dicts).
        config: GLiNER model config (needs ``max_neg_type_ratio``, ``max_types``).
        strategy: The active ``BatchStrategy``.
        rng: Seeded ``random.Random`` instance.
        key: Annotation key (default ``"ner"``).

    Returns:
        ``(class_to_ids, id_to_classes, meta_stats)``.
    """
    # -- batch-level negative pool (always from batch) --
    # Collect *all* unique types; per-instance selection is handled downstream
    # by _meta_sample_counts + rng.sample, so capping here is unnecessary.
    negatives = list({el[-1] for b in batch_list for el in b.get(key, [])})

    class_to_ids: List[Dict[str, int]] = []
    id_to_classes: List[Dict[int, str]] = []
    meta_stats: List[Dict[str, float]] = []

    max_types = int(config.max_types)

    for b in batch_list:
        # --- pre-defined labels: honour them verbatim, skip all sampling ---
        if f"{key}_labels" in b:
            types = b[f"{key}_labels"]
            _append_mappings(types, class_to_ids, id_to_classes)
            all_positives = list({el[-1] for el in b[key]})
            pos_in_types = sum(1 for p in all_positives if p in types)
            prevalence = pos_in_types / len(types) if len(types) > 0 else 0.0
            meta_stats.append({
                "cardinality": len(types),
                "prevalence": prevalence,
                "n_available_pos": len(all_positives),
                "n_available_neg": -1,  # pre-defined labels: pool size unknown
            })
            continue

        all_positives = list({el[-1] for el in b[key]})

        # --- meta-sampling path ---
        # Respect per-instance negative overrides (same as GLiNER default)
        if f"{key}_negatives" in b:
            instance_neg_pool = b[f"{key}_negatives"]
        else:
            instance_neg_pool = [n for n in negatives if n not in set(all_positives)]

        # Determine dynamic ceiling based on number of true positives
        P = len(all_positives)
        if strategy.dynamic_max_types_mult is not None:
            instance_max = min(strategy.dynamic_max_types_mult * P, max_types)
            instance_max = max(1, instance_max)  # Fallback if P=0
        else:
            instance_max = min(2 * P, max_types)  # Default to 2
            instance_max = max(1, instance_max)

        n_pos, n_neg = _meta_sample_counts(
            P, len(instance_neg_pool), instance_max, rng,
        )
        positives = _sample_positives_list(all_positives, n_pos, rng)
        negs_i = rng.sample(instance_neg_pool, min(n_neg, len(instance_neg_pool)))
        types = positives + negs_i
        if strategy.shuffle_types:
            rng.shuffle(types)
        _append_mappings(types, class_to_ids, id_to_classes)

        prevalence = len(positives) / len(types) if len(types) > 0 else 0.0
        meta_stats.append({
            "cardinality": len(types),
            "prevalence": prevalence,
            "n_available_pos": P,
            "n_available_neg": len(instance_neg_pool),
        })

    return class_to_ids, id_to_classes, meta_stats


def _sample_positives_list(
    all_positives: List[str], n: int, rng: random.Random,
) -> List[str]:
    """Sample exactly *n* positives (subsample or pad with duplicates)."""
    if n >= len(all_positives):
        return list(all_positives)
    return rng.sample(all_positives, n)


def _append_mappings(
    types: List[str],
    class_to_ids: List[Dict[str, int]],
    id_to_classes: List[Dict[int, str]],
) -> None:
    c2i = {k: v for v, k in enumerate(types, start=1)}
    i2c = {v: k for k, v in c2i.items()}
    class_to_ids.append(c2i)
    id_to_classes.append(i2c)


# ---------------------------------------------------------------------------
# ValidationSnapshot
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationSnapshot:
    """Frozen validation dataset with fixed, per-instance entity-type prompts.

    Created once before training begins and reused across every evaluation
    call.  Each instance is a shallow copy of the original val-dataset dict
    with an extra ``ner_labels`` key injected.  GLiNER's collator honours
    ``ner_labels`` verbatim (see ``BaseProcessor.batch_generate_class_mappings``),
    so the model always sees the same entity-type prompts for every eval.

    Attributes:
        instances: Val-dataset dicts with ``ner_labels`` injected.
        neg_mode: ``"per_instance"`` (all instances had ``ner_negatives``) or
            ``"global"`` (negatives sampled from the dataset-wide type pool).
    """

    instances: List[Dict]
    neg_mode: str


def build_validation_snapshot(
    val_dataset: List[Dict],
    config: Any,
    seed: Optional[int] = None,
    group_by: Optional[str] = None,
) -> ValidationSnapshot:
    """Build a :class:`ValidationSnapshot` from a validation dataset.

    For each instance the entity-type prompt is constructed as:

    * **Positives** — every unique entity type that appears in the instance's
      gold ``ner`` annotations (all of them, no sub-sampling).
    * **Negatives** — sampled *without replacement* from the pool to target a
      50 % positive / 50 % negative balance (``n_neg ≈ n_pos``), capped by
      ``config.max_types``.

    The negative pool is chosen as follows (priority order):

    1. If **all** instances carry a ``ner_negatives`` key → use each
       instance's own list (*per-instance* mode).  ``group_by`` is ignored.
    2. If ``group_by`` is set → build one negative pool **per group**,
       containing only entity types found in that group (*group-local* mode).
       This prevents cross-group type contamination.
    3. Otherwise → build a single pool from every entity type in the dataset
       (*global* mode).

    A :class:`UserWarning` is issued to inform the caller which mode was
    selected and how the snapshot was constructed.

    Args:
        val_dataset: List of instance dicts (same format as training data).
        config: GLiNER model config; needs ``max_types`` attribute.
        seed: Optional RNG seed for reproducible negative sampling.
        group_by: Optional key to partition instances by group.  When set,
            negatives are drawn only from entity types in the same group.

    Returns:
        A frozen :class:`ValidationSnapshot` ready for use in
        :class:`StrategyTrainer`.
    """
    rng = random.Random(seed)
    max_types = int(config.max_types)
    n = len(val_dataset)

    # ── determine negative mode ───────────────────────────────────────────
    n_with_negatives = sum(1 for inst in val_dataset if "ner_negatives" in inst)
    all_have_negatives = n_with_negatives == n

    group_pools: Dict[str, List[str]] = {}  # group_value → list of all entity types in that group

    if all_have_negatives:
        neg_mode = "per_instance"
        warnings.warn(
            f"[ValidationSnapshot] All {n} instances have 'ner_negatives'. "
            "Using per-instance negatives for validation prompt construction."
            + (" group_by is ignored in per-instance mode." if group_by else ""),
            UserWarning,
            stacklevel=2,
        )
    elif group_by is not None:
        neg_mode = "group_local"
        # Build per-group type pools
        group_types: Dict[str, set] = {}
        group_counts: Dict[str, int] = {}
        for inst in val_dataset:
            gval = str(inst.get(group_by, "__default__"))
            group_counts[gval] = group_counts.get(gval, 0) + 1
            if gval not in group_types:
                group_types[gval] = set()
            for el in inst.get("ner", []):
                group_types[gval].add(el[-1])
        group_pools = {g: list(types) for g, types in group_types.items()}

        # Warn with per-group pool sizes
        pool_info = ", ".join(
            f"{g}: {len(pool)} types ({group_counts[g]} inst)"
            for g, pool in sorted(group_pools.items())
        )
        # Check for groups with very small pools
        small_groups = [
            g for g, pool in group_pools.items()
            if len(pool) <= 1
        ]
        msg = (
            f"[ValidationSnapshot] group_by='{group_by}' → group-local negative pools. "
            f"Groups: [{pool_info}]."
        )
        if small_groups:
            msg += (
                f" WARNING: group(s) {small_groups} have ≤1 entity types — "
                "negative diversity will be limited for those instances."
            )
        if n_with_negatives > 0:
            msg += (
                f" Note: {n_with_negatives}/{n} instances have 'ner_negatives' "
                "but not all do, so group-local mode is used instead."
            )
        warnings.warn(msg, UserWarning, stacklevel=2)
    else:
        neg_mode = "global"
        global_pool = list({el[-1] for inst in val_dataset for el in inst.get("ner", [])})
        if n_with_negatives > 0:
            warnings.warn(
                f"[ValidationSnapshot] {n_with_negatives} of {n} instances have "
                f"'ner_negatives' (need ALL for per-instance mode). "
                f"Falling back to global entity-type pool (size={len(global_pool)}) "
                "for negative sampling.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"[ValidationSnapshot] No per-instance 'ner_negatives' found. "
                f"Sampling negatives from the global entity-type pool "
                f"(size={len(global_pool)}, seed={seed}).",
                UserWarning,
                stacklevel=2,
            )
        # Store as a single catch-all pool for uniform access below
        group_pools = {"__global__": global_pool}

    # ── build per-instance type lists ─────────────────────────────────────
    instances: List[Dict] = []
    total_types = 0
    n_short = 0  # instances where pool couldn't cover 50/50

    for inst in val_dataset:
        positives = list(dict.fromkeys(el[-1] for el in inst.get("ner", [])))
        n_pos = len(positives)
        pos_set = set(positives)

        # Build neg_pool (unique types only, positives excluded)
        neg_pool: List[str]
        if neg_mode == "per_instance":
            neg_pool = [str(t) for t in inst["ner_negatives"]]
        elif neg_mode == "group_local":
            gval = str(inst.get(group_by, "__default__"))
            neg_pool = [t for t in group_pools.get(gval, []) if t not in pos_set]
        else:  # global
            neg_pool = [t for t in group_pools["__global__"] if t not in pos_set]

        # Target 50/50: n_neg ≈ n_pos, capped by pool size and max_types
        n_neg = min(n_pos, len(neg_pool), max(0, max_types - n_pos))
        # Guard: never leave an instance with an empty prompt
        if n_pos == 0 and len(neg_pool) > 0:
            n_neg = 1

        if n_neg < n_pos and len(neg_pool) < n_pos:
            n_short += 1

        negs: List[str] = rng.sample(neg_pool, n_neg) if n_neg > 0 else []
        type_list: List[str] = positives + negs
        rng.shuffle(type_list)
        del type_list[max_types:]  # truncate in-place

        inst_copy = dict(inst)
        inst_copy["ner_labels"] = type_list
        instances.append(inst_copy)
        total_types += len(type_list)

    mean_types = total_types / n if n else 0.0

    if n_short > 0:
        warnings.warn(
            f"[ValidationSnapshot] {n_short}/{n} instances had fewer unique "
            f"negatives available than positives — those prompts will have "
            f">50% positive prevalence.",
            UserWarning,
            stacklevel=2,
        )

    logger.info(
        "[ValidationSnapshot] Built %d instances. Mode: %s. "
        "Mean types/instance: %.1f. max_types=%d.",
        n,
        neg_mode,
        mean_types,
        max_types,
    )

    return ValidationSnapshot(instances=instances, neg_mode=neg_mode)


# ---------------------------------------------------------------------------
# StrategyCollator
# ---------------------------------------------------------------------------

class StrategyCollator:
    """Wraps any GLiNER data collator to apply a :class:`BatchStrategy`.

    When the strategy is default (``is_default``), the original GLiNER
    ``batch_generate_class_mappings`` is called **verbatim** and its result
    is injected into the collator kwargs so that ``collate_raw_batch``
    skips its own regeneration (it checks ``if class_to_ids is None``).
    This guarantees **100 % identical** label-mapping behaviour to
    unpatched GLiNER — the same function, same arguments, same global
    ``random`` state consumption, same output.
    """

    def __init__(self, wrapped_collator: Any, strategy: BatchStrategy, config: Any) -> None:
        self.wrapped = wrapped_collator
        self.strategy = strategy
        self.config = config
        self._rng = random.Random(strategy.seed)

    def __call__(
        self,
        input_x: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:

        meta_stats = []
        key = kwargs.get("key", "ner")

        if self.strategy.is_default:
            # ── 100 % original GLiNER path ──────────────────────────────
            # Call the *exact same function* that unpatched GLiNER would
            # call inside collate_raw_batch, with the same arguments and
            # consuming the global random state at the same point.  The
            # result is then injected into kwargs so that collate_raw_batch
            # sees class_to_ids is not None and skips its own call.
            negatives = kwargs.get("negatives", None)
            class_to_ids, id_to_classes = self.wrapped.data_processor.batch_generate_class_mappings(
                input_x, negatives=negatives, key=key
            )
            # Harmless read-only telemetry (no RNG, no mutation)
            for i, b in enumerate(input_x):
                all_positives = list({el[-1] for el in b.get(key, [])})
                types = list(class_to_ids[i].keys())
                pos_in_types = sum(1 for p in all_positives if p in types)
                prevalence = pos_in_types / len(types) if len(types) > 0 else 0.0
                meta_stats.append({
                    "cardinality": len(types),
                    "prevalence": prevalence,
                    "n_available_pos": len(all_positives),
                    # Default path: the resolved negative pool lives inside
                    # batch_generate_class_mappings and cannot be observed
                    # without re-calling get_negatives (which consumes RNG).
                    # Sentinel -1 signals "not available".
                    "n_available_neg": -1,
                })
        else:
            # ── Meta-sampling path ──────────────────────────────────────
            class_to_ids, id_to_classes, meta_stats = build_class_mappings(
                input_x, self.config, self.strategy, self._rng, key=key
            )

        # Inject the computed mappings into the kwargs; GLiNER's
        # collate_raw_batch natively accepts these as keyword parameters
        # and skips batch_generate_class_mappings when they are not None.
        kwargs["class_to_ids"] = class_to_ids
        kwargs["id_to_classes"] = id_to_classes
        batch = self.wrapped(input_x, **kwargs)

        # Attach pure PyTorch telemetry to safely cross process boundaries into the trainer
        batch["empirical_cardinality"] = torch.tensor([s["cardinality"] for s in meta_stats], dtype=torch.int32)
        batch["empirical_prevalence"] = torch.tensor([s["prevalence"] for s in meta_stats], dtype=torch.float32)
        batch["empirical_n_available_pos"] = torch.tensor([s["n_available_pos"] for s in meta_stats], dtype=torch.int32)
        batch["empirical_n_available_neg"] = torch.tensor([s["n_available_neg"] for s in meta_stats], dtype=torch.int32)

        return batch


# ---------------------------------------------------------------------------
# GroupedSampler
# ---------------------------------------------------------------------------

class GroupedSampler(Sampler):
    """Sampler that guarantees every batch contains instances from the same group.

    Groups are shuffled each epoch, and instances within each group are also
    shuffled.  To prevent the DataLoader from creating cross-group batches at
    group boundaries, each group is padded to a multiple of ``batch_size``
    by repeating randomly chosen instances from the same group.  This means:

    * Every instance appears **at least once** per epoch.
    * At most ``batch_size - 1`` instances per group are duplicated.
    * Every batch is **homogeneous** (single group value).

    Args:
        dataset: The training dataset (list of dicts).
        group_key: Key to read the group value from each instance.
        batch_size: Number of instances per batch.
        drop_last: Whether to drop the last incomplete batch per group.
        generator: Optional PyTorch generator for reproducibility.
    """

    def __init__(
        self,
        dataset: List[Dict],
        group_key: str,
        batch_size: int,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.dataset = dataset
        self.group_key = group_key
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        self._build_groups()

    def _build_groups(self) -> None:
        self._groups: Dict[Any, List[int]] = {}
        for idx, item in enumerate(self.dataset):
            gval = item.get(self.group_key, "__default__")
            self._groups.setdefault(gval, []).append(idx)

    def __iter__(self):
        # 1. Shuffle instances within each group and split into batches
        batch_queues: Dict[str, List[List[int]]] = {}

        for gk, group_indices in self._groups.items():
            indices = list(group_indices)
            inner_perm = torch.randperm(len(indices), generator=self.generator).tolist()
            indices = [indices[i] for i in inner_perm]

            # Pad or drop to align with batch_size
            remainder = len(indices) % self.batch_size
            if remainder > 0:
                if self.drop_last:
                    indices = indices[: len(indices) - remainder]
                else:
                    pad_count = self.batch_size - remainder
                    pad_indices = [indices[j % len(indices)] for j in range(pad_count)]
                    indices = indices + pad_indices

            if indices:
                batches = [indices[i:i + self.batch_size]
                           for i in range(0, len(indices), self.batch_size)]
                batch_queues[gk] = batches

        # 2. Collect all batches and randomly permute them.  This gives
        #    each batch position an equal probability of holding any
        #    group's batch — no early/late bias for any group.
        all_batches: List[List[int]] = []
        for batches in batch_queues.values():
            all_batches.extend(batches)

        perm = torch.randperm(len(all_batches), generator=self.generator).tolist()
        for i in perm:
            yield from all_batches[i]

    def __len__(self) -> int:
        total = 0
        for indices in self._groups.values():
            n = len(indices)
            remainder = n % self.batch_size
            if remainder > 0:
                if self.drop_last:
                    total += n - remainder
                else:
                    total += n + (self.batch_size - remainder)
            else:
                total += n
        return total


# ---------------------------------------------------------------------------
# StrategyTrainer
# ---------------------------------------------------------------------------

class StrategyTrainer(_BaseTrainer):
    """Trainer subclass that honours :class:`BatchStrategy` grouping and
    F1-based checkpointing via :class:`ValidationSnapshot`.

    If ``strategy.group_by`` is set, ``_get_train_sampler`` returns a
    :class:`GroupedSampler`.

    If ``val_snapshot`` is provided, ``evaluate()`` computes strict/exact
    match F1 on the snapshot.  Model selection is handled by HuggingFace's
    native ``load_best_model_at_end`` / ``metric_for_best_model`` mechanism.
    """

    def __init__(
        self,
        strategy: BatchStrategy,
        val_snapshot: Optional[ValidationSnapshot] = None,
        eval_threshold: float = 0.5,
        eval_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.strategy = strategy
        self.val_snapshot = val_snapshot
        self.eval_threshold = eval_threshold
        self.eval_batch_size = eval_batch_size
        
        # High-performance C-contiguous arrays to track millions of scalar stats with ~0 overhead
        self._empirical_cardinality = array.array('i')
        self._empirical_prevalence = array.array('f')
        self._empirical_n_available_pos = array.array('i')
        self._empirical_n_available_neg = array.array('i')  # -1 sentinel when pool size is unknown
        
        super().__init__(**kwargs)
        # Auto-configure evaluation strategy AFTER super().__init__ so that
        # HF callbacks read the updated args when the training loop starts.
        self._configure_evaluation()

    def _configure_evaluation(self) -> None:
        """Auto-set the evaluation strategy to 'steps' when a snapshot is present.

        Called after ``super().__init__()`` so HF callbacks pick up the change
        at training time (they read ``self.args`` dynamically in
        ``on_step_end``).  Only activates when the current strategy is "no"
        (the HF default), leaving any explicit user configuration untouched.

        Transformers ≥ 4.41 uses ``eval_strategy``; older releases use
        ``evaluation_strategy``.  We detect which attribute the installed
        version actually reads (i.e. maps to ``IntervalStrategy``) and set
        that one, so the ``DefaultFlowCallback`` sees the change.
        """
        if self.val_snapshot is None:
            return
        # Use the attribute that the installed transformers version maps to an
        # IntervalStrategy (the one the training loop actually reads).
        # eval_strategy   → transformers >= 4.41
        # evaluation_strategy → older releases
        primary = (
            "eval_strategy"
            if hasattr(self.args, "eval_strategy")
            else "evaluation_strategy"
        )
        current = str(getattr(self.args, primary, "no")).lower()
        if "no" not in current:
            return  # already configured by user
        save_steps = int(getattr(self.args, "save_steps", 50) or 50)
        setattr(self.args, primary, "steps")
        self.args.eval_steps = save_steps
        logger.info(
            "StrategyTrainer: auto-set %s='steps', eval_steps=%d (matched save_steps).",
            primary,
            save_steps,
        )

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.strategy.group_by is not None and self.train_dataset is not None:
            return GroupedSampler(
                self.train_dataset,
                group_key=self.strategy.group_by,
                batch_size=self._train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )
        return super()._get_train_sampler()

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, Any]:
        """Compute strict/exact-match F1 on the frozen :class:`ValidationSnapshot`.

        Replaces HF Trainer's default evaluation loop entirely.  The GLiNER
        model's own ``evaluate()`` method is reused — it operates on
        pre-tokenised ``tokenized_text`` dicts without any text reconstruction.

        The entity-type list in each snapshot instance is frozen at snapshot-
        creation time (``ner_labels`` injected once by
        :func:`build_validation_snapshot`) and used **verbatim** by GLiNER's
        collator on every evaluation call — the type order presented to the
        model is therefore identical across all epochs.

        Decoding is configured for **maximum flexibility**:

        * ``flat_ner=False`` — overlapping and nested entities are allowed.
        * ``multi_label=True`` — a single span may carry multiple labels.

        ``eval_batch_size=None`` (the default) falls back to the training
        batch size (``per_device_train_batch_size``).

        Model selection is handled by HuggingFace's native
        ``load_best_model_at_end`` / ``metric_for_best_model`` mechanism:
        this method only returns the metric dict.

        Returns:
            Dict with ``"{metric_key_prefix}_f1"`` (float in [0, 1]).
            Empty dict when no snapshot is configured.
        """
        if self.val_snapshot is None:
            return {}

        # Unwrap accelerate / DDP wrapper to get the raw GLiNER model.
        gliner_model = self.accelerator.unwrap_model(self.model)

        # eval_batch_size=None → fall back to training batch size.
        eff_batch_size = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.args.per_device_train_batch_size
        )

        # GLiNER's evaluate() calls self.eval() internally.
        # flat_ner=False + multi_label=True → maximum decoder flexibility
        # (overlapping, nested, and multi-label entities are all considered).
        with torch.inference_mode():
            _, f1 = gliner_model.evaluate(
                self.val_snapshot.instances,
                flat_ner=False,
                multi_label=True,
                threshold=self.eval_threshold,
                batch_size=eff_batch_size,
            )
        f1 = float(f1)

        metrics: Dict[str, Any] = {f"{metric_key_prefix}_f1": f1}
        self.log(metrics)

        # Trigger on_evaluate callbacks (EarlyStoppingCallback, etc.)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,
        )

        step = self.state.global_step
        logger.info(
            "Step %d: %s_f1=%.4f",
            step,
            metric_key_prefix,
            f1,
        )

        return metrics

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Intercept the batch right before the forward pass to pop off empirical stats."""
        card = inputs.pop("empirical_cardinality", None)
        prev = inputs.pop("empirical_prevalence", None)
        n_apos = inputs.pop("empirical_n_available_pos", None)
        n_aneg = inputs.pop("empirical_n_available_neg", None)
        
        if card is not None:
            self._empirical_cardinality.extend(card.view(-1).tolist())
        if prev is not None:
            self._empirical_prevalence.extend(prev.view(-1).tolist())
        if n_apos is not None:
            self._empirical_n_available_pos.extend(n_apos.view(-1).tolist())
        if n_aneg is not None:
            self._empirical_n_available_neg.extend(n_aneg.view(-1).tolist())
            
        return super().training_step(model, inputs, *args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Hook into HF's native save cycle to dump the empirical stats arrays to disk."""
        super()._save(output_dir, state_dict)
        
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        if save_dir is not None:
            stats_path = os.path.join(save_dir, "empirical_stats.pt")
            torch.save({
                "cardinality": list(self._empirical_cardinality),
                "prevalence": list(self._empirical_prevalence),
                "n_available_pos": list(self._empirical_n_available_pos),
                "n_available_neg": list(self._empirical_n_available_neg),
            }, stats_path)
            logger.info("Saved %d empirical stat records to %s", len(self._empirical_cardinality), stats_path)


# ---------------------------------------------------------------------------
# patch_train_model
# ---------------------------------------------------------------------------

def patch_train_model(
    model: Any,
    strategy: BatchStrategy,
    val_snapshot: Optional[ValidationSnapshot] = None,
    eval_threshold: float = 0.5,
    eval_batch_size: Optional[int] = None,
    early_stopping_patience: int = 0,
    early_stopping_threshold: float = 0.0,
) -> None:
    """Monkey-patch ``model.train_model`` to use the given strategy.

    After calling this function, ``model.train_model(...)`` will
    automatically use :class:`StrategyCollator` and :class:`StrategyTrainer`.

    When *val_snapshot* is provided, :class:`StrategyTrainer` will compute
    strict/exact-match F1 on the snapshot at every save step.  Model
    selection is handled by HuggingFace's native ``load_best_model_at_end``
    / ``metric_for_best_model`` mechanism.  Decoding always uses
    ``flat_ner=False`` and ``multi_label=True`` for maximum decoder
    flexibility.

    Args:
        model: A ``BaseGLiNER`` instance.
        strategy: The batch-creation strategy to apply.
        val_snapshot: Optional frozen validation snapshot for F1-based
            checkpointing.  Build one with :func:`build_validation_snapshot`.
        eval_threshold: Score threshold passed to GLiNER inference (default
            0.5).
        eval_batch_size: Batch size for the evaluation data-loader.
            ``None`` (default) falls back to the training batch size
            (``per_device_train_batch_size``).
        early_stopping_patience: Number of evaluation steps without
            improvement before stopping training.  ``0`` (default) disables
            early stopping.  Requires ``val_snapshot`` and
            ``load_best_model_at_end=True`` in the ``TrainingArguments``.
        early_stopping_threshold: Minimum absolute improvement in F1 to
            count as a genuine improvement (default ``0.0``).
    """
    import transformers
    from packaging import version

    original_train_model = model.train_model.__func__  # unwrap bound method

    @functools.wraps(original_train_model)
    def patched_train_model(
        self,
        train_dataset,
        eval_dataset,
        training_args=None,
        freeze_components=None,
        compile_model=False,
        output_dir=None,
        **training_kwargs,
    ):
        from gliner.training import TrainingArguments

        # Create training arguments if not provided
        if training_args is None:
            if output_dir is None:
                msg = "Either training_args or output_dir must be provided"
                raise ValueError(msg)
            training_args = self.create_training_args(output_dir=output_dir, **training_kwargs)

        # Compile model if requested
        if compile_model:
            self.compile()

        # Freeze components if specified
        if freeze_components:
            for component_name in freeze_components:
                self.freeze_component(component_name)

        # Create data collator and wrap it with the strategy
        data_collator = self._create_data_collator()
        strategy_collator = StrategyCollator(data_collator, strategy, self.config)

        # Build trainer kwargs
        trainer_kwargs: Dict[str, Any] = {
            "strategy": strategy,
            "val_snapshot": val_snapshot,
            "eval_threshold": eval_threshold,
            "eval_batch_size": eval_batch_size,
            "model": self,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": strategy_collator,
        }

        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            trainer_kwargs["tokenizer"] = self.data_processor.transformer_tokenizer
        else:
            trainer_kwargs["processing_class"] = self.data_processor.transformer_tokenizer

        trainer = StrategyTrainer(**trainer_kwargs)

        # Register early stopping callback if configured
        if val_snapshot is not None and early_stopping_patience > 0:
            from transformers import EarlyStoppingCallback
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            ))
            logger.info(
                "[EarlyStopping] Enabled with patience=%d, threshold=%.4f",
                early_stopping_patience,
                early_stopping_threshold,
            )

        trainer.train()
        return trainer

    # Bind the patched method to the instance
    import types
    model.train_model = types.MethodType(patched_train_model, model)

