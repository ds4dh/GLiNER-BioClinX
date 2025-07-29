#!/usr/bin/env python3
"""Tokenize multilingual JSON data with Stanza, sample uniformly per language,
   and write two output files:
   1. train_uniform.json – 2,000 instances per language (shuffled).
   2. val_uniform.json – 400 instances per language (shuffled).
   Sampling is done before tokenization to minimize compute.
"""
import argparse
import json
import os
import random
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import stanza
from tqdm.auto import tqdm


def download_models(langs):
    """Download Stanza tokenize models for each language."""
    for lang in langs:
        stanza.download(lang, processors="tokenize", verbose=False)


def build_pipelines(langs):
    """Instantiate a Stanza Pipeline for each language."""
    return {lang: stanza.Pipeline(lang=lang, processors="tokenize", verbose=False)
            for lang in langs}


def tokenize_entry(entry, pipeline):
    """Tokenize a single entry using the given pipeline."""
    text = entry.get("lead_text", "")
    doc = pipeline(text)

    tokens = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            tokens.append({
                "text": tok.text,
                "start_char": tok.start_char,
                "end_char": tok.end_char,
            })

    spans_sorted = sorted(entry.get("spans", []), key=lambda span: span[0])
    token_spans = []
    for start, end in spans_sorted:
        i0 = next((i for i, t in enumerate(tokens)
                   if t["start_char"] <= start < t["end_char"]), None)
        i1 = next((i for i, t in enumerate(tokens)
                   if t["start_char"] < end <= t["end_char"]), None)
        token_spans.append((i0, i1))

    unique = []
    seen = set()
    for sp in token_spans:
        if sp not in seen:
            unique.append(sp)
            seen.add(sp)

    return {"tokenized_text": [tok["text"] for tok in tokens],
            "ner": [[i, j, "entity"] for i, j in unique]}


def main():
    DEFAULT_LANGS = ["en", "fr", "de", "it", "es", "nl", "uk", "ru", "zh", "ja"]

    parser = argparse.ArgumentParser(
        description="Sample and tokenize multilingual JSON data with Stanza"
    )
    parser.add_argument(
        "--input-path", "-i",
        default="./filtered_data.json",
        help="Path to the raw JSON input file",
    )
    parser.add_argument(
        "--output-train-path", "-t",
        default="./train_uniform.json",
        help="Path for the tokenized training output",
    )
    parser.add_argument(
        "--output-val-path", "-v",
        default="./val_uniform.json",
        help="Path for the tokenized validation output",
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=DEFAULT_LANGS,
        help="List of ISO language codes to sample",
    )
    parser.add_argument(
        "--download-models", "-d",
        action="store_true",
        help="Download Stanza tokenize models before processing",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, int(os.cpu_count() * 0.8)),
        help="Number of threads for tokenization (default: 80% of CPUs)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    if args.download_models:
        download_models(args.languages)

    # Load raw data
    with open(args.input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Sample entries per language before tokenization
    random.seed(args.seed)
    train_samples = {}
    val_samples = {}
    for lang in args.languages:
        entries = raw.get(lang, [])
        if len(entries) < 2_400:
            raise ValueError(f"Language '{lang}' has only {len(entries)} entries, need at least 10,000.")
        sample = random.sample(entries, 2_400)
        train_samples[lang] = sample[:2_000]
        val_samples[lang] = sample[2_000:]

    pipelines = build_pipelines(args.languages)

    # Tokenize and write each split
    for split_name, samples, out_path in [
        ("train", train_samples, args.output_train_path),
        ("val", val_samples, args.output_val_path),
    ]:
        all_tok = []
        print(f"Tokenizing {split_name} split ({sum(len(v) for v in samples.values())} entries)...")
        for lang, entries in tqdm(samples.items(), desc=f"Languages [{split_name}]"):
            pipeline = pipelines[lang]
            fn = partial(tokenize_entry, pipeline=pipeline)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(
                    tqdm(
                        executor.map(fn, entries),
                        total=len(entries),
                        desc=f"Entries [{lang}]",
                        leave=False,
                    )
                )
            all_tok.extend(results)

        # Save without indentation for compactness
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_tok, f, ensure_ascii=False)
        print(f"✅ {split_name.capitalize()} split written to {out_path}\n")


if __name__ == "__main__":
    main()