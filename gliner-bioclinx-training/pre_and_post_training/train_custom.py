"""Simple training script for GLiNER with BatchStrategy support.

Usage:
    python train_custom.py --config configs/config_custom.yaml
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

# Disable rust-based tokenizers parallelism to prevent deadlocks when forking PyTorch DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import yaml
from gliner.training import TrainingArguments

from gliner import GLiNER
from batch_strategy import BatchStrategy, ValidationSnapshot, build_validation_snapshot, patch_train_model, StrategyTrainer


def patched_uni_span_loss(
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
        scores, labels, alpha, gamma, prob_margin, label_smoothing, negatives=negatives, masking=masking
    )

    masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
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

def load_config(config_path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_data(path: str) -> list:
    """Load a JSON dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_strategy(cfg: dict) -> BatchStrategy:
    """Build a BatchStrategy from the config's 'batch_strategy' section."""
    bs_cfg = cfg.get("batch_strategy", {})
    if not bs_cfg:
        return BatchStrategy()  # strict no-op default
    return BatchStrategy(**bs_cfg)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # ── Build model ──────────────────────────────────────────────
    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        model = GLiNER.from_pretrained(prev_path)
    else:
        print("Initialising model from config...")
        model = GLiNER.from_config(model_cfg)
    model = model.to(dtype=torch.float32)

    # Apply mathematical mean-reduction patch safely
    from gliner.modeling.base import UniEncoderSpanModel
    if isinstance(model.model, UniEncoderSpanModel):
        import types
        model.model.loss = types.MethodType(patched_uni_span_loss, model.model)

    print(f"Model type: {type(model.model).__name__}")

    # Fix the HuggingFace tokenizer warning: 'Asking to truncate to max_length but no...'
    if hasattr(model, "data_processor") and hasattr(model.data_processor, "transformer_tokenizer"):
        tokenizer = model.data_processor.transformer_tokenizer
        
        # If the original tokenizer authors provided a real max_length, it will be a reasonable integer (e.g. 512, 1024).
        # HuggingFace defaults 'model_max_length' to a huge number (> 1e10) if the authors did not define one.
        if getattr(tokenizer, "model_max_length", 1e30) > 1e10:
            token_len = getattr(model.config, "max_len_tokenizer", None)
            if token_len is None:
                token_len = 1_000_000
                warnings.warn(
                    f"Tokenizer 'max_len_tokenizer' was not provided in the config or was set to null, "
                    f"and the original model authors did not define a default max length. "
                    f"Setting internal tokenizer.model_max_length to {token_len} (essentially no truncation)."
                )
            tokenizer.model_max_length = int(token_len)

    # ── Load data ────────────────────────────────────────────────
    print(f"Loading training data from: {data_cfg['train_data']}")
    train_data = load_json_data(data_cfg["train_data"])
    print(f"Training samples: {len(train_data)}")

    eval_data = None
    val_path = data_cfg.get("val_data_dir", "none")
    if val_path and val_path.lower() not in ("none", "null", ""):
        eval_data = load_json_data(val_path)
        print(f"Validation samples: {len(eval_data)}")

    # ── Path with BatchStrategy ─────────────────────────────────
    strategy = build_strategy(cfg)
    print(f"BatchStrategy: {strategy}")

    # ── Build ValidationSnapshot (if val data available) ─────────
    eval_cfg = cfg.get("eval", {})
    val_snapshot: Optional[ValidationSnapshot] = None
    if eval_data is not None:
        val_snapshot = build_validation_snapshot(
            eval_data,
            model.config,
            seed=eval_cfg.get("seed"),
            group_by=strategy.group_by,
        )
        print(f"ValidationSnapshot: {len(val_snapshot.instances)} instances, mode={val_snapshot.neg_mode}")

    # ── Patch model with strategy ─────────────────────────────────
    _bs = eval_cfg.get("batch_size")
    _patience = int(train_cfg.get("early_stopping_patience", 30))
    _threshold = float(train_cfg.get("early_stopping_threshold", 0.005))
    patch_train_model(
        model,
        strategy,
        val_snapshot=val_snapshot,
        eval_threshold=float(eval_cfg.get("threshold", 0.5)),
        eval_batch_size=int(_bs) if _bs is not None else None,
        early_stopping_patience=_patience,
        early_stopping_threshold=_threshold,
    )

    # ── Output directory ─────────────────────────────────────────
    output_dir = Path(data_cfg.get("root_dir", "models_custom"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Construct Native HuggingFace TrainingArguments ───────────
    # We do this manually here so we can activate HF's optimal checkpointing,
    # which ensures full optimizer/tokenizer states are preserved on the best epoch.
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": train_cfg["num_steps"],
        "lr_scheduler_type": train_cfg.get("scheduler_type", "linear"),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.1),
        "per_device_train_batch_size": train_cfg.get("train_batch_size", 2),
        "per_device_eval_batch_size": train_cfg.get("train_batch_size", 2),
        "learning_rate": float(train_cfg.get("lr_encoder", 5e-5)),
        "others_lr": float(train_cfg.get("lr_others", 5e-5)),
        "weight_decay": float(train_cfg.get("weight_decay_encoder", 0.01)),
        "others_weight_decay": float(train_cfg.get("weight_decay_other", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "focal_loss_alpha": float(train_cfg.get("loss_alpha", -1.0)),
        "focal_loss_gamma": float(train_cfg.get("loss_gamma", 0.0)),
        "loss_reduction": train_cfg.get("loss_reduction", "mean"),
        "negatives": float(train_cfg.get("negatives", 1.0)),
        "masking": train_cfg.get("masking", "none"),
        "save_steps": train_cfg.get("eval_every", 50),
        "logging_steps": train_cfg.get("eval_every", 50),
        "save_total_limit": train_cfg.get("save_total_limit", 1),
        "save_safetensors": train_cfg.get("save_safetensors", False),
        "dataloader_num_workers": train_cfg.get("num_workers", 0),
        "bf16": train_cfg.get("bf16", False),
        "fp16": False,
    }

    # If validation data is present, instruct HuggingFace to properly load and save the BEST metric
    if eval_data is not None:
        training_args_kwargs["load_best_model_at_end"] = train_cfg.get("load_best_model_at_end", True)
        training_args_kwargs["metric_for_best_model"] = "eval_f1"  # Matches StrategyTrainer.evaluate dictionary key
        training_args_kwargs["greater_is_better"] = True
        training_args_kwargs["save_strategy"] = "steps" # StrategyTrainer will auto-enforce this, but set it explicitly
        
        # HuggingFace requires eval and save strategies to match when load_best_model_at_end=True
        training_args_kwargs["evaluation_strategy"] = "steps" # Older HF
        training_args_kwargs["eval_strategy"] = "steps"       # Newer HF

    training_args = TrainingArguments(**training_args_kwargs)

    # ── Train ────────────────────────────────────────────────────
    print("\nStarting training...")
    model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        training_args=training_args,
    )

    print(f"\n✓ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER with BatchStrategy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_custom.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
