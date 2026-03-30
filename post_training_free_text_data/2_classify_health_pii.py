import os
import re
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "nvidia/domain-classifier"

# Strongly recommended:
# Pin the exact revision you validated from the HF repo.
# Replace this with the exact commit hash you want to freeze.
# Example:
# MODEL_REVISION = "b18c7bf..."
MODEL_REVISION = None

BATCH_SIZE = 32
CHECKPOINT_EVERY_N_BATCHES = 50
CHECKPOINT_DIR = "checkpoints_domain_clf"
FINAL_OUTPUT = "pii_500k_english_with_domain_probs.parquet"

SEED = 42
MAX_LENGTH = 512  # matches model card / config context length
USE_CUDA = torch.cuda.is_available()


# ============================================================
# Reproducibility / numerical caution
# ============================================================
def set_global_determinism(seed: int = 42) -> None:
    """
    Maximize reproducibility and reduce hardware-dependent numeric drift.
    This may reduce performance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=False)

    # cuDNN / matmul settings
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Disable TF32 to avoid reduced-precision matmul on Ampere+
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Prefer highest float32 matmul precision if available
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


# ============================================================
# Domain classifier model definition
# Mirrors the model card implementation closely.
# ============================================================
class DomainClassifier(nn.Module, PyTorchModelHubMixin):
    """
    Wrapper matching the nvidia/domain-classifier architecture.

    Model card pattern:
      - AutoModel.from_pretrained(config["base_model"])
      - Dropout(config["fc_dropout"])
      - Linear(hidden_size -> num_labels)
      - softmax(outputs[:, 0, :], dim=1)
    """

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            config["base_model"],
            revision=MODEL_REVISION,
        )
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        features = outputs.last_hidden_state

        # Be explicit about float32 before the classifier head.
        # This prioritizes numerical safety over speed / memory.
        features = features.to(torch.float32)

        dropped = self.dropout(features)
        logits = self.fc(dropped)

        # Respect the model card implementation:
        # classification is based on token position 0.
        probs = torch.softmax(logits[:, 0, :], dim=1)

        # Force float32 probabilities explicitly.
        return probs.to(torch.float32)


# ============================================================
# Checkpoint helpers
# ============================================================
def get_checkpoint_paths() -> list[str]:
    pattern = os.path.join(CHECKPOINT_DIR, "checkpoint_*.parquet")
    paths = glob.glob(pattern)

    def extract_idx(path: str) -> int:
        m = re.search(r"checkpoint_(\d+)\.parquet$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(paths, key=extract_idx)


def load_completed_uids() -> set:
    paths = get_checkpoint_paths()
    if not paths:
        return set()

    uids = set()
    for p in paths:
        chunk = pd.read_parquet(p, columns=["uid"])
        uids.update(chunk["uid"].tolist())

    print(f"Found {len(paths)} checkpoint(s) with {len(uids)} already-processed UIDs.")
    return uids


def next_checkpoint_index() -> int:
    paths = get_checkpoint_paths()
    if not paths:
        return 0

    last = os.path.basename(paths[-1])
    m = re.search(r"checkpoint_(\d+)\.parquet$", last)
    if not m:
        raise RuntimeError(f"Unexpected checkpoint filename: {last}")
    return int(m.group(1)) + 1


def save_checkpoint(df_chunk: pd.DataFrame, idx: int) -> str:
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{idx:05d}.parquet")
    df_chunk.to_parquet(path, index=False)
    return path


def merge_checkpoints_to_final(prob_columns: list[str]) -> pd.DataFrame:
    paths = get_checkpoint_paths()
    if not paths:
        raise RuntimeError("No checkpoint files found. Nothing has been processed yet.")

    chunks = [pd.read_parquet(p) for p in paths]
    df_final = pd.concat(chunks, ignore_index=True)

    # Defensive dedupe in case of partial reruns / overlaps
    if "uid" not in df_final.columns:
        raise RuntimeError("Checkpoint data is missing required 'uid' column.")

    df_final = (
        df_final.sort_values("uid")
        .drop_duplicates(subset=["uid"], keep="last")
        .reset_index(drop=True)
    )

    # Convenience columns
    df_final["predicted_domain"] = (
        df_final[prob_columns].idxmax(axis=1).str.replace("prob_", "", regex=False)
    )
    df_final["predicted_prob"] = df_final[prob_columns].max(axis=1)

    # Useful health-specific convenience columns
    if "prob_Health" in df_final.columns:
        df_final["health_prob"] = df_final["prob_Health"]
        df_final["is_health_top1"] = df_final["predicted_domain"].eq("Health")

    df_final.to_parquet(FINAL_OUTPUT, index=False)
    return df_final


# ============================================================
# Data preparation
# ============================================================
def has_valid_spans(mask) -> bool:
    """
    Accept privacy_mask if it is a non-empty sequence whose elements contain
    span offset information. Handles common structures conservatively.
    """
    if mask is None:
        return False

    try:
        if len(mask) == 0:
            return False
    except TypeError:
        return False

    # Common case: list-like of dicts
    try:
        first = mask[0]
    except Exception:
        return True  # non-empty but non-indexable; keep conservatively

    if isinstance(first, dict):
        return ("start" in first) and ("end" in first)

    # Nested list/array of dicts
    try:
        if len(first) > 0 and isinstance(first[0], dict):
            return ("start" in first[0]) and ("end" in first[0])
    except Exception:
        pass

    # Unknown non-empty structure: keep rather than over-filter
    return True


def is_nonempty(x) -> bool:
    if x is None:
        return False
    try:
        return len(x) > 0
    except TypeError:
        return False


def prepare_data() -> pd.DataFrame:
    """
    Download, merge train + validation, filter English,
    clean rows, and return uid-sorted DataFrame.
    """
    print("Downloading dataset...")
    ds = load_dataset("ai4privacy/open-pii-masking-500k-ai4privacy")

    merged = concatenate_datasets([ds["train"], ds["validation"]])
    print(f"Total rows after merge: {len(merged)}")

    merged = merged.filter(lambda x: x["language"] == "en")
    print(f"English rows: {len(merged)}")

    df = merged.to_pandas()

    if "uid" not in df.columns:
        raise RuntimeError("Expected a 'uid' column for deterministic ordering/resume.")
    if "source_text" not in df.columns:
        raise RuntimeError("Expected a 'source_text' column.")
    if "privacy_mask" not in df.columns:
        raise RuntimeError("Expected a 'privacy_mask' column.")

    # Drop rows without source_text
    before = len(df)
    df = df[df["source_text"].notna() & (df["source_text"].astype(str).str.strip() != "")]
    print(f"Dropped {before - len(df)} rows with missing/empty source_text -> {len(df)}")

    source_only_backup = df.copy()

    # Diagnose privacy_mask lightly
    if len(df) > 0:
        sample_masks = df["privacy_mask"].head(3).tolist()
        print("\n[DEBUG] privacy_mask examples:")
        for i, sm in enumerate(sample_masks):
            preview = str(sm)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  row {i}: type={type(sm)} value={preview}")
        print()

    before = len(df)
    valid_mask = df["privacy_mask"].apply(has_valid_spans)
    n_valid = int(valid_mask.sum())
    print(f"Rows with valid privacy_mask spans: {n_valid} / {before}")

    if n_valid == 0:
        print("\nWARNING: No rows passed the strict privacy_mask span filter.")
        print("Falling back to non-empty privacy_mask rows.")

        df = df[df["privacy_mask"].apply(is_nonempty)]

        if len(df) == 0:
            print("WARNING: Relaxed filter still yielded 0 rows. Keeping all rows with source_text.")
            df = source_only_backup
    else:
        df = df[valid_mask]

    print(f"After privacy_mask filter -> {len(df)} rows")

    # Deterministic ordering
    df = df.sort_values("uid").reset_index(drop=True)
    return df


# ============================================================
# Inference helpers
# ============================================================
def get_device() -> torch.device:
    return torch.device("cuda" if USE_CUDA else "cpu")


def load_model_and_tokenizer():
    kwargs = {}
    if MODEL_REVISION is not None:
        kwargs["revision"] = MODEL_REVISION

    print("Loading config...")
    config = AutoConfig.from_pretrained(MODEL_NAME, **kwargs)

    id2label = config.id2label
    class_names = [id2label[i] for i in range(len(id2label))]
    prob_columns = [f"prob_{c}" for c in class_names]

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)

    print(f"Loading model from {MODEL_NAME}...")
    model = DomainClassifier.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()

    device = get_device()
    model = model.to(device)

    # Keep model parameters in float32 explicitly
    model = model.float()

    print(f"Device: {device}")
    print(f"Domain classes ({len(class_names)}): {class_names}")

    return config, tokenizer, model, class_names, prob_columns


def run_inference(
    df_todo: pd.DataFrame,
    tokenizer,
    model,
    prob_columns: list[str],
) -> None:
    checkpoint_idx = next_checkpoint_index()
    buffer = []
    batches_in_buffer = 0

    total_batches = (len(df_todo) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Running inference: {total_batches} batches | "
        f"batch_size={BATCH_SIZE} | "
        f"checkpoint_every={CHECKPOINT_EVERY_N_BATCHES}"
    )

    device = next(model.parameters()).device

    # inference_mode is stricter / lighter than no_grad for pure inference
    with torch.inference_mode():
        for start in tqdm(range(0, len(df_todo), BATCH_SIZE), total=total_batches):
            end = min(start + BATCH_SIZE, len(df_todo))
            batch_slice = df_todo.iloc[start:end].copy()
            batch_texts = batch_slice["source_text"].astype(str).tolist()

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            probs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)

            for i, col in enumerate(prob_columns):
                batch_slice[col] = probs_np[:, i]

            # Optional sanity check: each row should sum to ~1
            row_sums = probs_np.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                raise RuntimeError(
                    "Probability rows do not sum to 1 within tolerance. "
                    "This suggests a bug or numeric instability."
                )

            buffer.append(batch_slice)
            batches_in_buffer += 1

            if batches_in_buffer >= CHECKPOINT_EVERY_N_BATCHES:
                chunk = pd.concat(buffer, ignore_index=True)
                path = save_checkpoint(chunk, checkpoint_idx)
                tqdm.write(f"✓ saved checkpoint {checkpoint_idx:05d} -> {path} ({len(chunk)} rows)")
                checkpoint_idx += 1
                buffer = []
                batches_in_buffer = 0

    # Final flush
    if buffer:
        chunk = pd.concat(buffer, ignore_index=True)
        path = save_checkpoint(chunk, checkpoint_idx)
        print(f"✓ saved checkpoint {checkpoint_idx:05d} -> {path} ({len(chunk)} rows, final flush)")


# ============================================================
# Main
# ============================================================
def main():
    set_global_determinism(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 80)
    print("nvidia/domain-classifier inference")
    print("=" * 80)
    print(f"MODEL_NAME      : {MODEL_NAME}")
    print(f"MODEL_REVISION  : {MODEL_REVISION}")
    print(f"BATCH_SIZE      : {BATCH_SIZE}")
    print(f"MAX_LENGTH      : {MAX_LENGTH}")
    print(f"CHECKPOINT_DIR  : {CHECKPOINT_DIR}")
    print(f"FINAL_OUTPUT    : {FINAL_OUTPUT}")
    print(f"SEED            : {SEED}")
    print(f"CUDA AVAILABLE  : {torch.cuda.is_available()}")
    print("=" * 80)

    # Prepare dataset
    df = prepare_data()

    # Resume support
    done_uids = load_completed_uids()
    df_todo = df[~df["uid"].isin(done_uids)].reset_index(drop=True)
    print(f"Remaining to process: {len(df_todo)} / {len(df)}")

    # Load model config even if nothing remains
    _, tokenizer, model, class_names, prob_columns = load_model_and_tokenizer()

    if len(df) == 0:
        print("\nNo eligible data to process. Check your filters.")
        return

    if len(df_todo) == 0:
        print("\nAll instances already processed. Merging checkpoints...")
        df_final = merge_checkpoints_to_final(prob_columns)
        print(f"Final output: {FINAL_OUTPUT}")
        print(f"Shape: {df_final.shape}")

        print("\nPredicted domain distribution:")
        print(df_final["predicted_domain"].value_counts().to_string())

        if "prob_Health" in df_final.columns:
            print("\nTop rows by Health probability:")
            cols = ["uid", "prob_Health", "predicted_domain", "predicted_prob"]
            print(df_final.sort_values("prob_Health", ascending=False)[cols].head(10).to_string(index=False))
        return

    # Run inference
    run_inference(
        df_todo=df_todo,
        tokenizer=tokenizer,
        model=model,
        prob_columns=prob_columns,
    )

    # Merge checkpoints
    print("\nMerging all checkpoints into final output...")
    df_final = merge_checkpoints_to_final(prob_columns)

    print(f"\nSaved to {FINAL_OUTPUT}")
    print(f"Shape: {df_final.shape}")

    print("\nPredicted domain distribution:")
    print(df_final["predicted_domain"].value_counts().to_string())

    if "prob_Health" in df_final.columns:
        print("\nHealth summary:")
        print(f"Rows with top-1 Health: {int(df_final['is_health_top1'].sum())}")
        print(f"Mean Health probability: {df_final['prob_Health'].mean():.6f}")
        print("\nTop rows by Health probability:")
        cols = ["uid", "prob_Health", "predicted_domain", "predicted_prob"]
        print(df_final.sort_values("prob_Health", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()