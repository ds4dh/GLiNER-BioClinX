#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import logging
import queue
import signal
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import semchunk
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm  # type: ignore
from transformers import AutoTokenizer

# ───────────────────────── constants & globals ──────────────────────────
LOG = logging.getLogger("mmedc")
RETRY_BACKOFF = 30
MAX_RETRIES = 6
MAX_CHARS = 500_000
MAX_TOKENS = 64_000
INTERRUPT = False

# ───────────────────────────── SIGINT handler ───────────────────────────
def _handle_sigint(signum, frame):  # noqa: ARG001
    global INTERRUPT                 # noqa: PLW0603
    if INTERRUPT:
        LOG.error("Second Ctrl-C – force exit.")
        sys.exit(1)
    INTERRUPT = True
    LOG.warning("Ctrl-C received – finishing current batch…")

signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────── CLI parsing ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-id", default="Henrychur/MMedC")
    p.add_argument("--local-path", type=Path, default=Path("~/.cache/mmedc").expanduser())
    p.add_argument("--languages", nargs="*", help="Optional whitelist (zip stems)")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=100, help="Minimum number of tokens to retain a sample/chunk")
    p.add_argument("--quota-per-lang", type=int, default=100_000, help="Target chunks per language")
    p.add_argument("--output-path", type=Path, default=Path("./data/preprocessed/mmedc_chunked.csv"))
    p.add_argument("--workers", type=int, default=max(2, int(cpu_count() * 0.75)))
    p.add_argument("--max-tasks", type=int, default=3_000, help="Max queued futures before back-pressure")
    p.add_argument("--flush-size", type=int, default=10_000, help="Flush to disk every N rows")
    p.add_argument("--hf-token")
    return p.parse_args()

# ───────────────────────────── downloader ──────────────────────────────
def _sleep_backoff(i: int):
    time.sleep(RETRY_BACKOFF * (2 ** (i - 1)))

def _download(rel: str, repo: str, cache: Path, token: str | None) -> Path:
    tgt = cache / rel
    if tgt.exists():
        return tgt
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return Path(hf_hub_download(repo, rel, repo_type="dataset",
                                        cache_dir=str(cache),
                                        resume_download=True, token=token))
        except Exception as exc:                # noqa: BLE001
            LOG.warning("%s (%d/%d): %s", rel, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)
    raise RuntimeError("unreachable")

def ensure_zips(args) -> List[Path]:
    args.local_path.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(args.dataset_id, repo_type="dataset")
    zips = [f for f in repo_files if f.lower().endswith(".zip")]
    if args.languages:
        want = {f"{l}.zip" for l in args.languages}
        zips = [z for z in zips if Path(z).name in want]
    LOG.info("Preparing %d ZIP archives …", len(zips))
    return [_download(rel, args.dataset_id, args.local_path, args.hf_token)
            for rel in tqdm(zips, desc="Download", unit="zip")]

# ───────────────────────────── worker setup ────────────────────────────
_tok = _enc = _chunker = None
def _worker_init(chunk_size: int):
    global _tok, _enc, _chunker     # noqa: PLW0603
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True, use_fast=True)
    _enc = lambda t: len(_tok.encode(t, add_special_tokens=False))      # type: ignore
    _chunker = semchunk.chunkerify(_enc, chunk_size=chunk_size)

def _proc(sample: Tuple[str, str], min_t: int) -> List[Tuple[str, int, str]]:
    """Process one (language, text) pair in a worker."""
    lang, text = sample
    if len(text) > MAX_CHARS or len(text) // 3 > MAX_TOKENS:
        return []
    if _enc(text) < min_t:                      # document too short
        return []
    try:
        rows: List[Tuple[str, int, str]] = []
        for chunk in _chunker(text):
            tok_count = _enc(chunk)
            if tok_count >= min_t:              # keep only substantial chunks
                rows.append((lang, tok_count, chunk))
        return rows
    except Exception:
        return []

# ─────────────────────────── incremental writers ───────────────────────
def _flush_csv(buf: List[Tuple[str, int, str]], path: Path):
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
        table = pa.Table.from_pandas(pd.DataFrame(buf, columns=["language", "num_tokens", "text"]))
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        buf.clear()
    def close(self):
        if self._writer:
            self._writer.close()

# ───────────────────────── drain helper (quota-aware) ───────────────────
def drain_futures(fq: "queue.Queue[Future]",
                  buffer: List[Tuple[str, int, str]],
                  *,
                  counts: Dict[str, int],
                  quota: int,
                  done: set[str],
                  csv_path: Path | None,
                  pq_app: ParquetAppender | None,
                  flush_size: int) -> set[str]:
    while not fq.empty():
        rows = fq.get().result()
        if not rows:
            continue
        for lang, tok, txt in rows:
            if counts[lang] >= quota:
                continue
            counts[lang] += 1
            buffer.append((lang, tok, txt))
            if counts[lang] >= quota:
                done.add(lang)
            if len(buffer) >= flush_size:
                if csv_path:
                    _flush_csv(buffer, csv_path)
                else:
                    pq_app.write(buffer)             # type: ignore[arg-type]
    return done

# ─────────────────────────────────── main ───────────────────────────────
def main():  # noqa: C901
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    zip_paths = ensure_zips(args)
    if not zip_paths:
        LOG.error("No ZIP files – exit.")
        return

    target_langs = {Path(z).stem for z in zip_paths} if not args.languages else set(args.languages)

    # set up output
    is_csv = args.output_path.suffix.lower() == ".csv"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.exists():
        args.output_path.unlink()
    csv_path = args.output_path if is_csv else None
    if is_csv:
        _flush_csv([], csv_path)                          # header
        pq_app = None
    else:
        pq_app = ParquetAppender(args.output_path)

    counts: Dict[str, int] = {lang: 0 for lang in target_langs}
    finished: set[str] = set()
    buffer: List[Tuple[str, int, str]] = []
    fut_q: "queue.Queue[Future]" = queue.Queue(maxsize=args.max_tasks)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(args.chunk_size,)) as pool:
        for zp in zip_paths:
            lang = Path(zp).stem
            if lang in finished:
                LOG.info("%s skipped – quota already met.", zp.name)
                continue
            if INTERRUPT:
                break

            LOG.info("Processing %s", zp.name)
            with ZipFile(zp) as zf:
                txt_entries = [e for e in zf.infolist()
                               if e.filename.endswith(".txt") and
                               "cultural_filtered_data_used" not in e.filename]
                for ix, entry in enumerate(tqdm(txt_entries, desc=lang, unit="file", leave=False)):
                    if INTERRUPT or lang in finished:
                        break
                    text = zf.read(entry).decode("utf-8", "ignore").strip()
                    fut_q.put(pool.submit(_proc, (lang, text), args.min_tokens))
                    if ix % 100 == 0 or fut_q.qsize() >= args.max_tasks:
                        finished = drain_futures(
                            fut_q, buffer,
                            counts=counts, quota=args.quota_per_lang,
                            done=finished,
                            csv_path=csv_path, pq_app=pq_app,
                            flush_size=args.flush_size)
                finished = drain_futures(
                    fut_q, buffer,
                    counts=counts, quota=args.quota_per_lang,
                    done=finished,
                    csv_path=csv_path, pq_app=pq_app,
                    flush_size=args.flush_size)
            gc.collect()
            if finished == target_langs:
                LOG.info("All quotas reached – stopping.")
                break

        # final drain
        finished = drain_futures(
            fut_q, buffer,
            counts=counts, quota=args.quota_per_lang,
            done=finished,
            csv_path=csv_path, pq_app=pq_app,
            flush_size=0)

    # final flush
    if is_csv:
        _flush_csv(buffer, csv_path)
    else:
        pq_app.write(buffer)               # type: ignore[arg-type]
        pq_app.close()

    # stats
    LOG.info("Computing stats …")
    if is_csv:
        df = pd.read_csv(args.output_path, usecols=["language", "num_tokens"])
    else:
        df = pd.read_parquet(args.output_path, columns=["language", "num_tokens"])
    stats = df.groupby("language")["num_tokens"].describe().reset_index()
    print("\nToken-length statistics by language\n")
    print(stats.to_string(index=False))

    stats_path = args.output_path.with_name(args.output_path.stem + "_stats" + args.output_path.suffix)
    if is_csv:
        stats.to_csv(stats_path, index=False)
    else:
        stats.to_parquet(stats_path, index=False)
    LOG.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()