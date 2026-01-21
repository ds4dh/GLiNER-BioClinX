#!/usr/bin/env python
"""x_swl_infer.py

Run GLiNER on a HuggingFace dataset of passages, saving one JSON-Lines record per passage.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python x_swl_infer.py --output ../data/x_swl_output/fr.jsonl --language fr --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from gliner import GLiNER
from tqdm.auto import tqdm

###############################################################################
# Helpers
###############################################################################


def batched(items: List[str], n: int):
    """Yield successive *n*-sized chunks from *items*."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def safe_predict(
    model: GLiNER,
    passages: List[str],
    threshold: float,
) -> List[List[dict]]:
    """Predict entities, recursively halving the batch on failure."""

    try:
        return model.batch_predict_entities(
            passages,
            ["entity"],
            multi_label=False,
            flat_ner=True,
            threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001
        if len(passages) == 1:
            logging.error("⚠️  Failed on single passage (%s chars): %s", len(passages[0]), exc)
            return [None]
        mid = len(passages) // 2
        logging.warning("Batch of %d failed; splitting into %d & %d", len(passages), mid, len(passages) - mid)
        return safe_predict(model, passages[:mid], threshold) + safe_predict(model, passages[mid:], threshold)

###############################################################################
# Main
###############################################################################


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Run GLiNER NER and save JSONL, with resume & progress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", default="anthonyyazdaniml/gliner-bioclin-x-free-text-balanced-12k-v1.1", help="HuggingFace dataset name")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--language", required=True, help="Only keep rows where language column equals this value")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Process only the first N passages after filtering (0 = all)")

    parser.add_argument("--model", default="anthonyyazdaniml/x-swl-large-v4.1", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Initial batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entity score threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")

    parser.add_argument("--text-column", default="text", help="Dataset column containing passages")
    parser.add_argument("--language-column", default="language", help="Dataset column with language tags")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")


    # ------------------------------------------------------------------
    # Load & filter data from HuggingFace
    # ------------------------------------------------------------------
    logging.info("Loading dataset %s", args.dataset)
    dataset = load_dataset(args.dataset, split="train") # Assuming the 'train' split
    logging.info("Dataset loaded with %d rows.", len(dataset))

    if args.language:
        dataset = dataset.filter(lambda example: example[args.language_column] == args.language)
        logging.info("Filtered to %d rows for language '%s'", len(dataset), args.language)

    passages: List[str] = dataset[args.text_column]

    if args.limit > 0:
        passages = passages[: args.limit]
        logging.info("Limiting to first %d passages", args.limit)

    if not passages:
        parser.error("No passages found after filtering/limiting. Check your parameters.")

    # ------------------------------------------------------------------
    # Determine resume point
    # ------------------------------------------------------------------
    processed = 0
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)
        logging.info("Resuming: %d passages already processed – skipping them", processed)

    passages = passages[processed:]
    if not passages:
        logging.info("Nothing left to do – exiting.")
        return

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    logging.info("Loading GLiNER model %s on %s", args.model, args.device)
    model = GLiNER.from_pretrained(args.model).to(args.device)
    model.eval()

    # ------------------------------------------------------------------
    # Inference loop with tqdm progress bar
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        pbar = tqdm(total=len(passages), unit="passage", desc="Inferring", leave=True)
        with torch.inference_mode():
            for batch in batched(passages, args.batch_size):
                preds = safe_predict(model, batch, args.threshold)
                for passage, pred in zip(batch, preds):
                    record = {"text": passage, "entities": pred}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                pbar.update(len(batch))
        pbar.close()

    logging.info("✓ All done – results saved to %s", args.output)


if __name__ == "__main__":
    main()