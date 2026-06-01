"""Fine-tuning script for GLiNER with exact-span F1 validation and early stopping.

Usage:
    python finetune.py \
        --config ... \
        --model  ... \
        --output_dir ...
"""

import argparse
import json
import logging
import os
import types
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
import yaml
from packaging import version
from transformers import EarlyStoppingCallback

from gliner import GLiNER
from gliner.training import TrainingArguments
from gliner.training.trainer import Trainer as GLiNERTrainer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Monkey-patches for GLiNER evaluation compatibility
# ═══════════════════════════════════════════════════════════════════
# GLiNER's _process_batches returns Span objects, but the unpatched
# BaseNEREvaluator.get_predictions assumes subscriptable tuples.
# The decoder patch prevents KeyError on padding classes.
# Both patches are idempotent (safe to apply if batch_strategy.py
# has already patched them).

from gliner.evaluation.evaluator import BaseNEREvaluator
from gliner.decoding.decoder import SpanDecoder, SpanGenerativeDecoder, SpanRelexDecoder, Span


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


def _make_safe_build_span_tuple(original_method):
    """Wrap a decoder's _build_span_tuple to handle unknown padding classes."""
    def _safe_build(self, start, width, class_idx, flat_idx, score,
                    id_to_class, span_label_map, class_probs=None):
        if id_to_class.get(class_idx + 1) is None:
            return Span(start=start, end=start + width, entity_type="unknown",
                        score=float("-inf"), class_probs=class_probs)
        return original_method(self, start, width, class_idx, flat_idx, score,
                               id_to_class, span_label_map, class_probs)
    return _safe_build


for _DecoderCls in (SpanDecoder, SpanGenerativeDecoder, SpanRelexDecoder):
    if not getattr(_DecoderCls, "_build_span_tuple_orig", None):
        _DecoderCls._build_span_tuple_orig = _DecoderCls._build_span_tuple
        _DecoderCls._build_span_tuple = _make_safe_build_span_tuple(
            _DecoderCls._build_span_tuple
        )


# ═══════════════════════════════════════════════════════════════════
# Mean-reduction loss patch
# ═══════════════════════════════════════════════════════════════════
# GLiNER's original loss(reduction="mean") divides by ALL tensor
# elements including padding zeros.  This patch divides by
# (valid_spans × valid_classes) for mathematically correct mean CE.

def _patched_loss(
    self,
    scores: torch.Tensor,
    labels: torch.Tensor,
    prompts_embedding_mask: torch.Tensor,
    span_mask: torch.Tensor,
    alpha: float = -1.0,
    gamma: float = 0.0,
    prob_margin: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "sum",
    negatives: float = 1.0,
    masking: str = "none",
    **kwargs,
) -> torch.Tensor:
    batch_size = scores.shape[0]
    num_classes = prompts_embedding_mask.shape[-1]
    BS, _, _, CL = scores.shape

    scores = scores.view(BS, -1, CL)
    labels = labels.view(BS, -1, CL)

    all_losses = self._loss(
        scores, labels, alpha, gamma, prob_margin, label_smoothing,
        negatives=negatives, masking=masking,
    )

    masked_loss = (
        all_losses.view(batch_size, -1, num_classes)
        * prompts_embedding_mask.unsqueeze(1)
    )
    all_losses = masked_loss.view(-1, num_classes)
    span_mask_flat = span_mask.view(-1, 1)
    all_losses = all_losses * span_mask_flat.float()

    if reduction == "mean":
        loss_sum = all_losses.sum()
        valid_spans = span_mask.view(batch_size, -1).sum(dim=1)
        valid_classes = prompts_embedding_mask.sum(dim=1)
        valid_elements = (valid_spans * valid_classes).sum()
        loss = loss_sum / torch.clamp(valid_elements, min=1.0)
    elif reduction == "sum":
        loss = all_losses.sum()
    else:
        loss = all_losses.sum()

    return loss


def _apply_loss_patch(model: GLiNER) -> None:
    """Monkey-patch the mean-reduction loss on the model instance."""
    from gliner.modeling.base import UniEncoderSpanModel

    if isinstance(model.model, UniEncoderSpanModel):
        model.model.loss = types.MethodType(_patched_loss, model.model)
        logger.info("Applied mean-reduction loss patch to %s.", type(model.model).__name__)
    else:
        warnings.warn(
            f"Model inner type is {type(model.model).__name__}, not UniEncoderSpanModel. "
            "Mean-reduction loss patch was NOT applied.  Verify that loss_reduction "
            "behaviour is correct for this model type.",
            UserWarning,
            stacklevel=2,
        )


# ═══════════════════════════════════════════════════════════════════
# FineTuneTrainer
# ═══════════════════════════════════════════════════════════════════

class FineTuneTrainer(GLiNERTrainer):
    """GLiNER Trainer with exact-span micro-F1 validation.

    Overrides HF Trainer's evaluate() to run GLiNER's native
    evaluation pipeline (inference + BaseNEREvaluator).

    Validation always uses flat_ner=False, multi_label=True.
    """

    def __init__(
        self,
        eval_data: List[Dict[str, Any]],
        eval_threshold: float = 0.5,
        eval_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.eval_data = eval_data
        self.eval_threshold = eval_threshold
        self.eval_batch_size = eval_batch_size
        super().__init__(**kwargs)

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, Any]:
        """Compute exact-span micro-F1 on validation data.

        Bypasses HF Trainer's default eval loop entirely.  Uses
        GLiNER's model.evaluate() which creates its own collator
        with prepare_labels=False.  Since val data has ner_labels,
        those are used verbatim (all types, alphabetical order).
        """
        gliner_model = self.accelerator.unwrap_model(self.model)
        gliner_model.eval()

        eff_batch_size = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.args.per_device_train_batch_size
        )

        data = eval_dataset if eval_dataset is not None else self.eval_data

        with torch.inference_mode():
            _, f1 = gliner_model.evaluate(
                data,
                flat_ner=False,
                multi_label=True,
                threshold=self.eval_threshold,
                batch_size=eff_batch_size,
            )

        f1 = float(f1)
        metrics: Dict[str, Any] = {f"{metric_key_prefix}_f1": f1}
        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,
        )

        step = self.state.global_step
        logger.info("Step %d: %s_f1=%.4f", step, metric_key_prefix, f1)

        return metrics


# ═══════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_effective_config(
    cfg: Dict[str, Any],
    model_name: str,
    output_dir: str,
) -> Path:
    """Save the effective config (YAML + CLI overrides) to output_dir."""
    effective = deepcopy(cfg)
    effective["_cli"] = {"model": model_name, "output_dir": output_dir}
    out_path = Path(output_dir) / "effective_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(effective, f, default_flow_style=False, sort_keys=False)
    logger.info("Effective config saved to %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════
# Data validation
# ═══════════════════════════════════════════════════════════════════

def _extract_ner_labels(data: List[Dict], split_name: str) -> List[str]:
    """Extract and validate ner_labels from a split.  Returns the canonical list."""
    if not data:
        raise ValueError(f"{split_name}: dataset is empty.")

    first_labels = data[0].get("ner_labels")
    if first_labels is None:
        raise ValueError(
            f"{split_name}, instance 0: missing 'ner_labels'.  "
            "Data must come from convert_to_new_gliner_format.py."
        )
    if not isinstance(first_labels, list) or len(first_labels) == 0:
        raise ValueError(f"{split_name}, instance 0: 'ner_labels' must be a non-empty list.")

    canonical = tuple(first_labels)

    errors: List[str] = []
    for i, inst in enumerate(data):
        nl = inst.get("ner_labels")
        if nl is None:
            errors.append(f"{split_name}, instance {i}: missing 'ner_labels'.")
            continue

        if tuple(nl) != canonical:
            errors.append(
                f"{split_name}, instance {i}: ner_labels differ from instance 0.  "
                f"Expected {len(canonical)} labels, got {len(nl)}."
            )

        if nl != sorted(nl):
            errors.append(
                f"{split_name}, instance {i}: ner_labels not sorted alphabetically."
            )

        if "ner" not in inst or not isinstance(inst.get("ner"), list):
            errors.append(f"{split_name}, instance {i}: missing or invalid 'ner' field.")
            continue

        label_set = set(nl)
        for j, ann in enumerate(inst["ner"]):
            if not (isinstance(ann, (list, tuple)) and len(ann) == 3):
                errors.append(f"{split_name}, instance {i}, ner[{j}]: expected [start, end, label].")
            elif ann[2] not in label_set:
                errors.append(
                    f"{split_name}, instance {i}, ner[{j}]: gold label '{ann[2]}' "
                    f"not in ner_labels."
                )

    if errors:
        for e in errors[:20]:
            logger.error(e)
        if len(errors) > 20:
            logger.error("... and %d more errors.", len(errors) - 20)
        raise ValueError(
            f"{len(errors)} validation error(s) in {split_name}.  See log above."
        )

    return list(canonical)


def validate_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: Optional[List[Dict]],
    max_types: int,
) -> List[str]:
    """Validate all splits and return the canonical ner_labels list."""
    train_labels = _extract_ner_labels(train_data, "train")
    val_labels = _extract_ner_labels(val_data, "val")

    if train_labels != val_labels:
        raise ValueError(
            f"ner_labels differ between train ({len(train_labels)} labels) "
            f"and val ({len(val_labels)} labels).  Both must use the same "
            f"dataset-level labels from convert_to_new_gliner_format.py."
        )

    if test_data is not None:
        test_labels = _extract_ner_labels(test_data, "test")
        if test_labels != train_labels:
            raise ValueError(
                f"ner_labels differ between train ({len(train_labels)} labels) "
                f"and test ({len(test_labels)} labels)."
            )

    n_labels = len(train_labels)
    if n_labels > max_types:
        raise ValueError(
            f"Dataset has {n_labels} entity types but model max_types={max_types}.  "
            f"The model cannot handle more types than it was configured for.  "
            f"Either use a model with max_types >= {n_labels} or reduce the label set."
        )

    logger.info(
        "Validation passed: %d entity types, max_types=%d.  "
        "Labels: %s",
        n_labels, max_types, train_labels,
    )
    return train_labels


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune GLiNER with exact-span F1 validation and early stopping.",
    )
    p.add_argument("--config", required=True, help="Path to dataset YAML config.")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path.")
    p.add_argument("--output_dir", required=True, help="Output directory for checkpoints and logs.")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)

    # ── Merge CLI overrides ──────────────────────────────────────
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("eval", {})

    output_dir = args.output_dir

    # ── Load data ────────────────────────────────────────────────
    logger.info("Loading training data from: %s", data_cfg["train"])
    with open(data_cfg["train"]) as f:
        train_data = json.load(f)
    logger.info("Training instances: %d", len(train_data))

    logger.info("Loading validation data from: %s", data_cfg["val"])
    with open(data_cfg["val"]) as f:
        val_data = json.load(f)
    logger.info("Validation instances: %d", len(val_data))

    test_data = None
    test_path = data_cfg.get("test")
    if test_path:
        logger.info("Loading test data from: %s", test_path)
        with open(test_path) as f:
            test_data = json.load(f)
        logger.info("Test instances: %d", len(test_data))

    # ── Load model ───────────────────────────────────────────────
    logger.info("Loading model from: %s", args.model)
    model = GLiNER.from_pretrained(args.model)
    model = model.to(dtype=torch.float32)
    logger.info("Model type: %s", type(model.model).__name__)

    # ── Validate data against model ──────────────────────────────
    ner_labels = validate_splits(train_data, val_data, test_data, int(model.config.max_types))

    # ── Apply mean-reduction loss patch ──────────────────────────
    _apply_loss_patch(model)

    # ── Fix tokenizer model_max_length ───────────────────────────
    if hasattr(model, "data_processor") and hasattr(model.data_processor, "transformer_tokenizer"):
        tokenizer = model.data_processor.transformer_tokenizer
        tok_max = train_cfg.get("tokenizer_max_length")
        if getattr(tokenizer, "model_max_length", 1e30) > 1e10:
            if tok_max is not None:
                tokenizer.model_max_length = int(tok_max)
            else:
                tokenizer.model_max_length = 1_000_000
                warnings.warn(
                    "Tokenizer model_max_length was unset by the model authors and "
                    "tokenizer_max_length was not provided in config.  "
                    "Set to 1,000,000 (no truncation).",
                    UserWarning,
                    stacklevel=2,
                )

    # ── Freeze components ────────────────────────────────────────
    freeze_list = train_cfg.get("freeze", [])
    for component_name in freeze_list:
        model.freeze_component(component_name)
        logger.info("Froze component: %s", component_name)

    # ── Create collator ─────────────────────────────────────────
    data_collator = model._create_data_collator()

    # ── Build TrainingArguments ──────────────────────────────────
    # Config uses user-facing names (consistent with pre-training config).
    # This block translates them to HuggingFace TrainingArguments fields.
    seed = eval_cfg.get("seed", 42)
    if args.seed is not None:
        seed = args.seed

    max_steps = train_cfg.get("num_steps", -1)

    # Epoch-based (default): evaluate + save once at the end of every epoch.
    # Step-based (num_steps > 0): evaluate + save every eval_every steps.
    if max_steps > 0:
        eval_every = train_cfg.get("eval_every")
        if eval_every is None:
            raise ValueError(
                "When using num_steps > 0, 'eval_every' must be set in the config "
                "to define how often to evaluate and checkpoint."
            )
        cadence_strategy = "steps"
        cadence_kwargs = {"save_steps": eval_every, "eval_steps": eval_every}
        logger.info("Step-based training: num_steps=%d, eval every %d steps.", max_steps, eval_every)
    else:
        cadence_strategy = "epoch"
        cadence_kwargs = {}
        logger.info(
            "Epoch-based training: %d epochs, eval+save at every epoch boundary.",
            train_cfg.get("num_train_epochs", 10),
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 10),
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg.get("train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("train_batch_size", 4),
        learning_rate=float(train_cfg.get("lr_encoder", 5e-6)),
        others_lr=float(train_cfg.get("lr_others", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay_encoder", 0.1)),
        others_weight_decay=float(train_cfg.get("weight_decay_other", 0.01)),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        lr_scheduler_type=train_cfg.get("scheduler_type", "cosine"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        save_total_limit=train_cfg.get("save_total_limit", 1),
        logging_steps=train_cfg.get("logging_steps", 10),
        dataloader_num_workers=train_cfg.get("num_workers", 0),
        seed=seed,
        bf16=train_cfg.get("bf16", False),
        report_to=train_cfg.get("report_to", "none"),
        save_safetensors=train_cfg.get("save_safetensors", False),
        # GLiNER-specific loss
        focal_loss_alpha=float(train_cfg.get("loss_alpha", -1.0)),
        focal_loss_gamma=float(train_cfg.get("loss_gamma", 0.0)),
        loss_reduction=train_cfg.get("loss_reduction", "mean"),
        negatives=float(train_cfg.get("negatives", 1.0)),
        masking=train_cfg.get("masking", "none"),
        # Evaluation and checkpointing — always aligned
        evaluation_strategy=cadence_strategy,  # HF < 4.46
        eval_strategy=cadence_strategy,        # HF >= 4.46
        save_strategy=cadence_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_only_model=train_cfg.get("save_only_model", True),
        **cadence_kwargs,
    )

    # ── Build Trainer ────────────────────────────────────────────
    trainer_kwargs: Dict[str, Any] = {
        "eval_data": val_data,
        "eval_threshold": eval_cfg.get("threshold", 0.5),
        "eval_batch_size": eval_cfg.get("batch_size"),
        "model": model,
        "args": training_args,
        "train_dataset": train_data,
        "data_collator": data_collator,
    }

    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        trainer_kwargs["tokenizer"] = model.data_processor.transformer_tokenizer
    else:
        trainer_kwargs["processing_class"] = model.data_processor.transformer_tokenizer

    trainer = FineTuneTrainer(**trainer_kwargs)

    # ── Early stopping ───────────────────────────────────────────
    patience = train_cfg.get("early_stopping_patience", 0)
    es_threshold = train_cfg.get("early_stopping_threshold", 0.001)
    if patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=es_threshold,
            )
        )
        logger.info(
            "Early stopping enabled: patience=%d, threshold=%.4f",
            patience, es_threshold,
        )

    # ── Save effective config ────────────────────────────────────
    save_effective_config(cfg, args.model, output_dir)

    # ── Train ────────────────────────────────────────────────────
    logger.info(
        "Starting fine-tuning: %d train instances, %d val instances, "
        "%d entity types, model=%s",
        len(train_data), len(val_data), len(ner_labels), args.model,
    )
    trainer.train()

    # ── Final test evaluation ────────────────────────────────────
    if test_data is not None:
        logger.info("Running final evaluation on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
        logger.info("Test results: %s", test_metrics)

        test_results_path = Path(output_dir) / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info("Test results saved to %s", test_results_path)

    logger.info("Fine-tuning complete.  Best model saved to %s", output_dir)


if __name__ == "__main__":
    main()
