#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import queue
import re
import signal
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import semchunk
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ─────────── globals ───────────
LOG = logging.getLogger("wiki_uk")
MAX_CHARS = 5_000_000
MAX_TOKENS = 1_600_000
INTERRUPT = False
DATASET_NAME = "anthonyyazdaniml/biomedical_ukrainian_wiki_articles"
LANG = "uk"

# ─────────── SIGINT ───────────
def _handle_sigint(signum, frame):  # noqa: ARG001
    global INTERRUPT                # noqa: PLW0603
    if INTERRUPT:
        LOG.error("Second Ctrl-C → hard exit.")
        sys.exit(1)
    INTERRUPT = True
    LOG.warning("Ctrl-C received – finishing current batch …")

signal.signal(signal.SIGINT, _handle_sigint)

# ─────────── CLI ───────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum tokens per chunk (semchunk parameter)",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum tokens required for a doc and for each kept chunk",
    )
    p.add_argument(
        "--quota-per-lang",
        type=int,
        default=100_000,
        help="Target number of accepted chunks",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("./data/preprocessed/wiki_uk_chunked.csv"),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(2, int(cpu_count() * 0.75)),
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=3_000,
        help="Max queued futures before back-pressure",
    )
    p.add_argument(
        "--flush-size",
        type=int,
        default=10_000,
        help="Flush to disk every N accepted chunks",
    )
    return p.parse_args()

# ─────────── split helper ───────────
def split_sections(text: str) -> List[str]:
    """
    Split a document on lines starting with '## '.
    Discard the header lines themselves and return non-empty segments.
    """
    parts = re.split(r'(?m)^##\s.*$', text)
    return [part.strip() for part in parts if part.strip()]

# ─────────── worker helper ───────────
_tok = _enc = _chunker = None
def _worker_init(chunk_size: int):
    """
    Each worker instantiates one Qwen tokenizer + semchunker
    exposing:
        _enc(text)  → token count
        _chunker(text) → list[str] of ≤chunk_size token spans
    """
    global _tok, _enc, _chunker            # noqa: PLW0603
    _tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-32B", trust_remote_code=True, use_fast=True
    )
    _enc = lambda t: len(_tok.encode(t, add_special_tokens=False))  # type: ignore
    _chunker = semchunk.chunkerify(_enc, chunk_size=chunk_size)

def _proc(sample: Tuple[int, str], min_t: int) -> List[Tuple[str, int, str]]:
    """
    Worker-side processing for one section.
    Returns a list of (lang, tok_count, chunk) passing thresholds.
    """
    doc_id, text = sample
    if not text:
        return []
    if len(text) > MAX_CHARS or len(text) // 3 > MAX_TOKENS:
        return []
    if _enc(text) < min_t:
        return []
    out: List[Tuple[str, int, str]] = []
    for chunk in _chunker(text):
        tok = _enc(chunk)
        if tok >= min_t:
            out.append((LANG, tok, chunk))
    return out

# ─────────── CSV writer ───────────
def _flush_csv(buf: list[tuple[str, int, str]], path: Path):
    if buf:
        pd.DataFrame(buf, columns=["language", "num_tokens", "text"]).to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        buf.clear()

# ─────────── drain helper ───────────
def drain_futures(
    fq: "queue.Queue[Future]",
    buffer: list[tuple[str, int, str]],
    *,
    accepted: int,
    quota: int,
    csv_path: Path,
    flush_size: int,
    bar: tqdm,
) -> int:
    """
    Pull completed futures, collect acceptable chunks, update progress bar,
    flush to disk when needed, and return new accepted count.
    """
    while not fq.empty() and accepted < quota:
        rows = fq.get().result()
        if not rows:
            continue
        for row in rows:
            if accepted >= quota:
                break
            buffer.append(row)
            accepted += 1
            bar.update(1)
            if len(buffer) >= flush_size:
                _flush_csv(buffer, csv_path)
    return accepted

# ─────────── main ───────────
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    LOG.info(f"Loading ‘{DATASET_NAME}’ locally…")
    ds = load_dataset(DATASET_NAME, split="train")  # full local load
    ds_iter = ds

    # prepare output CSV
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.exists():
        args.output_path.unlink()
    _flush_csv([], args.output_path)  # write header

    accepted = 0
    buffer: list[tuple[str, int, str]] = []
    fut_q: "queue.Queue[Future]" = queue.Queue(maxsize=args.max_tasks)

    sections_bar = tqdm(desc="sections read", unit="section")
    chunks_bar = tqdm(total=args.quota_per_lang, desc="accepted chunks", unit="chunk")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.chunk_size,),
    ) as pool:
        for sample in ds_iter:
            if INTERRUPT or accepted >= args.quota_per_lang:
                break

            full_text = sample.get("text", "") or ""
            sections = split_sections(full_text)

            for sec in sections:
                sections_bar.update(1)
                fut_q.put(pool.submit(_proc, (sample["id"], sec), args.min_tokens))

                # drain every 1000 sections to keep the pipeline flowing
                if sections_bar.n % 1000 == 0:
                    accepted = drain_futures(
                        fut_q,
                        buffer,
                        accepted=accepted,
                        quota=args.quota_per_lang,
                        csv_path=args.output_path,
                        flush_size=args.flush_size,
                        bar=chunks_bar,
                    )
                    if accepted >= args.quota_per_lang:
                        break

            # back-pressure drain
            if fut_q.qsize() >= args.max_tasks:
                accepted = drain_futures(
                    fut_q,
                    buffer,
                    accepted=accepted,
                    quota=args.quota_per_lang,
                    csv_path=args.output_path,
                    flush_size=args.flush_size,
                    bar=chunks_bar,
                )
                if accepted >= args.quota_per_lang:
                    break

        # final drain
        accepted = drain_futures(
            fut_q,
            buffer,
            accepted=accepted,
            quota=args.quota_per_lang,
            csv_path=args.output_path,
            flush_size=0,
            bar=chunks_bar,
        )

    # flush any remaining buffer
    if buffer:
        _flush_csv(buffer, args.output_path)

    sections_bar.close()
    chunks_bar.close()
    LOG.info("Finished: %d chunks written.", accepted)

    # Stats
    LOG.info("Computing stats …")
    df = pd.read_csv(args.output_path, usecols=["language", "num_tokens"])
    stats = df.groupby("language")["num_tokens"].describe().reset_index()

    print("\nToken-length statistics\n")
    print(stats.to_string(index=False))

    stats_path = args.output_path.with_name(
        args.output_path.stem + "_stats" + args.output_path.suffix
    )
    stats.to_csv(stats_path, index=False)
    LOG.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()