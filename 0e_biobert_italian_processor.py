#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, gc, logging, queue, signal, sys
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import semchunk
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ───────────────────────── constants & globals ──────────────────────────
LOG = logging.getLogger("biobert_it")
MAX_CHARS = 500_000       # hard cutoff on raw characters
MAX_TOKENS = 64_000       # “junk” cutoff on rough token length
INTERRUPT = False         # for graceful CTRL-C handling

# ───────────────────────────── SIGINT handler ───────────────────────────
def _handle_sigint(signum, frame):  # noqa: ARG001
    global INTERRUPT                 # noqa: PLW0603
    if INTERRUPT:
        LOG.error("Second Ctrl-C → exiting immediately.")
        sys.exit(1)
    INTERRUPT = True
    LOG.warning("Ctrl-C received – finishing current batch…")

signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────── CLI parsing ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Maximum tokens per semchunk")
    p.add_argument("--min-tokens", type=int, default=100,
                   help="Minimum tokens for a doc *and* each retained chunk")
    p.add_argument("--quota-per-lang", type=int, default=100_000,
                   help="Stop after this many accepted chunks (language = it)")
    p.add_argument("--output-path", type=Path,
                   default=Path("./data/preprocessed/biobert_it_chunked.csv"))
    p.add_argument("--workers", type=int, default=max(2, int(cpu_count() * 0.75)))
    p.add_argument("--max-tasks", type=int, default=3_000,
                   help="Max queued futures before back-pressure")
    p.add_argument("--flush-size", type=int, default=10_000,
                   help="Flush to disk every N accepted chunks")
    return p.parse_args()

# ───────────────────────────── worker setup ────────────────────────────
_tok = _enc = _chunker = None
def _worker_init(chunk_size: int):
    """Each worker holds one Qwen tokenizer + semchunker."""
    global _tok, _enc, _chunker     # noqa: PLW0603
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B",
                                         trust_remote_code=True,
                                         use_fast=True)
    _enc = lambda t: len(_tok.encode(t, add_special_tokens=False))       # type: ignore
    _chunker = semchunk.chunkerify(_enc, chunk_size=chunk_size)

def _proc(text: str, min_t: int) -> List[Tuple[str, int, str]]:
    """Worker-side processing for one document → list of accepted chunks."""
    if not text:
        return []
    if len(text) > MAX_CHARS or len(text) // 3 > MAX_TOKENS:
        return []
    if _enc(text) < min_t:          # document too short
        return []
    out: List[Tuple[str, int, str]] = []
    for chunk in _chunker(text):
        tok = _enc(chunk)
        if tok >= min_t:
            out.append(("it", tok, chunk))
    return out

# ─────────────────────────── incremental writers ───────────────────────
def _flush_csv(buf: list[tuple[str,int,str]], path: Path):
    if buf:
        pd.DataFrame(buf, columns=["language", "num_tokens", "text"]).to_csv(
            path, mode="a", header=not path.exists(), index=False,
            quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        buf.clear()

class ParquetAppender:
    def __init__(self, path: Path):
        self.path = path
        self._writer: pq.ParquetWriter | None = None
    def write(self, buf):
        if not buf:
            return
        table = pa.Table.from_pandas(pd.DataFrame(
            buf, columns=["language", "num_tokens", "text"]))
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        buf.clear()
    def close(self):
        if self._writer:
            self._writer.close()

# ───────────────────────── drain helper (quota-aware) ───────────────────
def drain_futures(fq: "queue.Queue[Future]",
                  buffer: list[tuple[str,int,str]],
                  *,
                  accepted: int,
                  quota: int,
                  csv_path: Path | None,
                  pq_app: ParquetAppender | None,
                  flush_size: int,
                  bar: tqdm) -> int:
    while not fq.empty() and accepted < quota:
        rows = fq.get().result()
        for row in rows:
            if accepted >= quota:
                break
            buffer.append(row)
            accepted += 1
            bar.update(1)
            if len(buffer) >= flush_size:
                if csv_path:
                    _flush_csv(buffer, csv_path)
                else:
                    pq_app.write(buffer)         # type: ignore[arg-type]
    return accepted

# ─────────────────────────────────── main ───────────────────────────────
def main():  # noqa: C901
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Streaming dataset iterator
    LOG.info("Loading IVN-RIN/BioBERT_Italian (streaming)…")
    ds_iter = load_dataset("IVN-RIN/BioBERT_Italian",
                           split="train", streaming=True)

    docs_bar = tqdm(desc="docs read", unit="doc")          # unbounded

    # Output setup
    is_csv = args.output_path.suffix.lower() == ".csv"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.exists():
        args.output_path.unlink()
    csv_path = args.output_path if is_csv else None
    if is_csv:
        _flush_csv([], csv_path)              # header
        pq_app = None
    else:
        pq_app = ParquetAppender(args.output_path)

    buffer: list[tuple[str,int,str]] = []
    fut_q: "queue.Queue[Future]" = queue.Queue(maxsize=args.max_tasks)
    accepted = 0

    chunks_bar = tqdm(total=args.quota_per_lang,
                      desc="accepted chunks",
                      unit="chunk")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(args.chunk_size,)) as pool:

        for idx, sample in enumerate(ds_iter):
            if INTERRUPT or accepted >= args.quota_per_lang:
                break

            docs_bar.update(1)
            text = sample["text"]              # the only column we need
            fut_q.put(pool.submit(_proc, text, args.min_tokens))

            # Periodic drain / back-pressure
            if (idx + 1) % 10_000 == 0 or fut_q.qsize() >= args.max_tasks:
                accepted = drain_futures(
                    fut_q, buffer,
                    accepted=accepted, quota=args.quota_per_lang,
                    csv_path=csv_path, pq_app=pq_app,
                    flush_size=args.flush_size,
                    bar=chunks_bar)

        # Final drain
        accepted = drain_futures(
            fut_q, buffer,
            accepted=accepted, quota=args.quota_per_lang,
            csv_path=csv_path, pq_app=pq_app,
            flush_size=0,
            bar=chunks_bar)

    # Flush remaining buffer
    if buffer:
        if is_csv:
            _flush_csv(buffer, csv_path)
        else:
            pq_app.write(buffer)               # type: ignore[arg-type]
    if pq_app:
        pq_app.close()

    docs_bar.close()
    chunks_bar.close()
    LOG.info("Finished – %d chunks written.", accepted)

    # Stats
    LOG.info("Computing stats …")
    if is_csv:
        df = pd.read_csv(args.output_path, usecols=["language", "num_tokens"])
    else:
        df = pd.read_parquet(args.output_path, columns=["language", "num_tokens"])
    stats = df.groupby("language")["num_tokens"].describe().reset_index()
    print("\nToken-length statistics\n")
    print(stats.to_string(index=False))

    stats_path = args.output_path.with_name(
        args.output_path.stem + "_stats" + args.output_path.suffix)
    if is_csv:
        stats.to_csv(stats_path, index=False)
    else:
        stats.to_parquet(stats_path, index=False)
    LOG.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()