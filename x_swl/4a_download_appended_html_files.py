#!/usr/bin/env python3
"""
4a_download_appended_html_files.py

Download Parsoid HTML only for newly appended language(s) when moving from:
  old CSV: multilingual_ids.csv
  new CSV: multilingual_ids_with_pt.csv

Strategy:
1) "Smart" mode: detect appended languages by lang-set difference (new_langs - old_langs).
   Example: adding Portuguese => {"pt"}.
   Then download only rows in NEW CSV whose lang is in appended languages.
2) Fallback: if no appended language is detected, compute row diff by (cluster_id, lang)
   and download only those pairs present in NEW but not in OLD.

Also uses a crash-safe journal (lang,title) to avoid redownloading across runs.

```bash
python 4a_download_appended_html_files.py \
  --old-csv multilingual_ids.csv \
  --new-csv multilingual_ids_with_pt.csv
```

If you want to be explicit and **force** only Portuguese (even if auto-detection would work):

```bash
python 4a_download_appended_html_files.py \
  --old-csv multilingual_ids.csv \
  --new-csv multilingual_ids_with_pt.csv \
  --langs pt
```
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm
from urllib.parse import quote


@dataclass(frozen=True)
class Config:
    outdir: Path
    headers: dict
    base_delay: float
    max_retries: int
    timeout: int
    sem_limit: int
    max_workers: int
    journal_path: Path


_JOURNAL_LOCK = threading.Lock()


def fetch_html(cfg: Config, lang: str, title: str) -> Optional[Tuple[str, str]]:
    """Return (slug, html) or None if the page ultimately fails."""
    slug = quote(title.replace(" ", "_"), safe="()")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{slug}"
    tries = 0

    while True:
        resp = requests.get(url, headers=cfg.headers, timeout=cfg.timeout)

        # page not found → skip it
        if resp.status_code == 404:
            return None

        if resp.status_code != 429:
            resp.raise_for_status()
            return slug, resp.text

        # rate limited
        wait = int(resp.headers.get("retry-after", 1))
        time.sleep(wait)

        tries += 1
        if tries > cfg.max_retries:
            print(f"gave up after {cfg.max_retries} retries: {lang}:{title}")
            return None


def save_html(cfg: Config, lang: str, slug: str, html: str) -> None:
    """Write the HTML to outdir/<lang>/<slug>.html, truncating at filesystem NAME_MAX."""
    lang_dir = cfg.outdir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    try:
        name_max = os.pathconf(str(lang_dir), "PC_NAME_MAX")
    except Exception:
        name_max = 255  # safe fallback

    ext = ".html"
    filename = slug + ext
    if len(filename) > name_max:
        allowed = max(1, name_max - len(ext))
        filename = filename[:allowed] + ext

    (lang_dir / filename).write_text(html, encoding="utf-8")


def _load_downloaded(journal_path: Path) -> Set[Tuple[str, str]]:
    """Return a set {(lang, title), …} already logged."""
    if not journal_path.exists():
        return set()
    with journal_path.open(newline="", encoding="utf-8") as fh:
        return {(lang, title) for lang, title in csv.reader(fh)}


def _log_download(journal_path: Path, lang: str, title: str) -> None:
    """Append one entry to the journal and fsync so it is crash-safe."""
    with _JOURNAL_LOCK:  # multiple threads → single writer
        journal_path.parent.mkdir(exist_ok=True)
        with journal_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([lang, title])
            fh.flush()
            os.fsync(fh.fileno())


def fetch_and_save(cfg: Config, sem: threading.Semaphore, lang: str, title: str) -> None:
    with sem:
        result = fetch_html(cfg, lang, title)
    if result:
        slug, html = result
        save_html(cfg, lang, slug, html)
        _log_download(cfg.journal_path, lang, title)
        time.sleep(cfg.base_delay)


def _normalize_langs_arg(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    langs = {x.strip() for x in s.split(",") if x.strip()}
    return langs or None


def _require_columns(df: pd.DataFrame, path: Path, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}. Found: {list(df.columns)}")


def select_appended_rows(
    old_csv: Path,
    new_csv: Path,
    forced_langs: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Return a dataframe with columns [lang, title] containing ONLY rows to download.
    Prefers appended language detection; falls back to (cluster_id, lang) diff.
    """
    df_old = pd.read_csv(old_csv)
    df_new = pd.read_csv(new_csv)

    _require_columns(df_old, old_csv, ["lang", "title", "cluster_id"])
    _require_columns(df_new, new_csv, ["lang", "title", "cluster_id"])

    # Clean up
    df_old = df_old.dropna(subset=["lang", "cluster_id"])
    df_new = df_new.dropna(subset=["lang", "cluster_id"])

    df_old["lang"] = df_old["lang"].astype(str)
    df_new["lang"] = df_new["lang"].astype(str)

    if forced_langs is not None:
        target_langs = forced_langs
        mode = f"forced langs={sorted(target_langs)}"
    else:
        old_langs = set(df_old["lang"].unique())
        new_langs = set(df_new["lang"].unique())
        appended_langs = new_langs - old_langs
        target_langs = appended_langs
        mode = f"auto-detected appended langs={sorted(target_langs)}"

    if target_langs:
        # Smart path: only download appended languages from NEW CSV.
        picked = df_new[df_new["lang"].isin(target_langs)]
        picked = picked.dropna(subset=["title"])[["lang", "title"]]
        print(f"[select] Using {mode}. Rows selected: {len(picked)}")
        return picked

    # Fallback: no new lang codes detected (or CSVs already include same langs)
    # Compute exact pair diff by (cluster_id, lang).
    old_pairs = set(zip(df_old["cluster_id"].astype(int), df_old["lang"]))
    new_pairs = set(zip(df_new["cluster_id"].astype(int), df_new["lang"]))
    only_new = new_pairs - old_pairs

    if not only_new:
        print("[select] No appended work detected (no new langs, and no new (cluster_id, lang) pairs).")
        return pd.DataFrame(columns=["lang", "title"])

    # Filter df_new to only those (cluster_id, lang) pairs.
    df_new_pairs = df_new[["cluster_id", "lang", "title"]].dropna(subset=["title"])
    mask = [
        (int(cid), lang) in only_new
        for cid, lang in zip(df_new_pairs["cluster_id"], df_new_pairs["lang"])
    ]
    picked = df_new_pairs.loc[mask, ["lang", "title"]]
    print(f"[select] Fallback diff mode (cluster_id,lang). Rows selected: {len(picked)}")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download HTML only for newly appended Wikipedia rows (e.g., appended pt)."
    )
    ap.add_argument("--old-csv", default="multilingual_ids.csv", help="old CSV (before append)")
    ap.add_argument("--new-csv", default="multilingual_ids_with_pt.csv", help="new CSV (after append)")
    ap.add_argument("--outdir", default="/scratch/yazdani0/multilingual_wikimed_html", help="output directory")
    ap.add_argument("--journal", default="downloaded_journal.csv", help="crash-safe journal CSV")

    ap.add_argument("--langs", default=None,
                    help="optional comma-separated langs to download (overrides auto-detection), e.g. 'pt'")
    ap.add_argument("--max-workers", type=int, default=50, help="thread pool size")
    ap.add_argument("--sem-limit", type=int, default=30, help="max concurrent HTTP requests")
    ap.add_argument("--base-delay", type=float, default=0.20, help="sleep after each successful save")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--max-retries", type=int, default=60, help="max 429 retries per page")
    args = ap.parse_args()

    cfg = Config(
        outdir=Path(args.outdir),
        headers={"User-Agent": "OfflineParsoid/0.4 (you@example.com)"},
        base_delay=float(args.base_delay),
        max_retries=int(args.max_retries),
        timeout=int(args.timeout),
        sem_limit=int(args.sem_limit),
        max_workers=int(args.max_workers),
        journal_path=Path(args.journal),
    )
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    old_csv = Path(args.old_csv)
    new_csv = Path(args.new_csv)
    forced_langs = _normalize_langs_arg(args.langs)

    rows_df = select_appended_rows(old_csv, new_csv, forced_langs=forced_langs)
    if rows_df.empty:
        print("Nothing to download ✨")
        return

    already = _load_downloaded(cfg.journal_path)

    # Build to_do from selected rows, respecting journal
    to_do = []
    for lang, title in rows_df.itertuples(index=False, name=None):
        # Normalize to plain Python strings (pandas can give numpy scalars)
        lang = str(lang)
        title = str(title)
        if (lang, title) not in already:
            to_do.append((lang, title))

    if not to_do:
        print("Everything in the appended set is already downloaded ✨")
        return

    sem = threading.Semaphore(cfg.sem_limit)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool, tqdm(
        total=len(to_do), desc="Downloading appended"
    ) as bar:
        futures = [pool.submit(fetch_and_save, cfg, sem, lang, title) for lang, title in to_do]
        for f in as_completed(futures):
            f.result()
            bar.update(1)


if __name__ == "__main__":
    main()