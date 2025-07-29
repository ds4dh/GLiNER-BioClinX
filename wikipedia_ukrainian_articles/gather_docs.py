#!/usr/bin/env python3
"""
Collect the extracted Wikipedia docs that match a list of page IDs and dump them
to one CSV file, excluding any pages that had template errors during extraction.

Folder layout created by wikiextractor (default):
    full_extract/AA/....txt
    full_extract/AB/....txt
    ...

Each .txt file contains many blocks of the form:
    <doc id="123" ...>
    text ...
    </doc>

Dependencies: pandas, tqdm

Example
-------
python gather_docs.py \
       --csv article_ids.csv \
       --indir full_extract \
       --warnlog extract_warnings.log \
       --out docs.csv \
       --workers 8
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
DOC_RE = re.compile(
    r'<doc id="(?P<id>\d+)"[^>]*>\s*(?P<text>.*?)\s*</doc>',
    re.S | re.MULTILINE,
)

WARN_RE = re.compile(
    r"WARNING: Template errors in article '.*?' \((?P<id>\d+)\):"
)


def load_target_ids(csv_path: Path) -> set[int]:
    """Read the `pageid` column from the CSV into a set of ints."""
    return (
        pd.read_csv(csv_path, usecols=["pageid"])
        .dropna()["pageid"]
        .astype(int)
        .pipe(set)
    )


def load_bad_ids(log_path: Path) -> set[int]:
    """
    Read the extraction warnings log and pull out any page IDs
    that had template errors, so we can exclude them.
    """
    bad_ids: set[int] = set()
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = WARN_RE.search(line)
            if m:
                bad_ids.add(int(m.group("id")))
    return bad_ids


def extract_from_file(file_path: Path, wanted: set[int]) -> List[Dict]:
    """
    Return a list of {"id": ..., "text": ...} for matches in one file.
    """
    out: List[Dict] = []
    with file_path.open("r", encoding="utf-8") as fh:
        content = fh.read()

    for m in DOC_RE.finditer(content):
        doc_id = int(m.group("id"))
        if doc_id in wanted:
            out.append({"id": doc_id, "text": m.group("text").strip()})
    return out


def iter_txt_files(root: Path) -> Iterable[Path]:
    """Yield every regular file under *root* (wikiextractor uses many dirs)."""
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            yield Path(dirpath) / fname


def gather_docs(
    csv_path: Path,
    indir: Path,
    out_path: Path,
    warnlog: Path,
    workers: int,
) -> None:
    # load desired IDs and subtract any that had template errors
    wanted = load_target_ids(csv_path)
    bad = load_bad_ids(warnlog)
    original_count = len(wanted)
    wanted -= bad
    print(f"Loaded {original_count} target IDs; excluding {len(bad)} bad IDs, {len(wanted)} remain.")

    all_files = list(iter_txt_files(indir))
    extractor = partial(extract_from_file, wanted=wanted)
    results: List[Dict] = []

    with Pool(workers) as pool, tqdm(
        total=len(all_files),
        desc="Scanning files",
        unit="file",
    ) as bar:
        for docs in pool.imap_unordered(extractor, all_files, chunksize=10):
            if docs:
                results.extend(docs)
            bar.update(1)

    # Convert to DataFrame and write CSV
    df = pd.DataFrame(results, columns=["id", "text"])
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"✅ Saved {len(df):,} docs to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Wikipedia docs by ID, excluding pages with template errors, and dump to CSV."
    )
    p.add_argument(
        "--csv", default="article_ids.csv", type=Path,
        help="CSV file with a 'pageid' column (default: %(default)s)"
    )
    p.add_argument(
        "--indir", default="full_extract", type=Path,
        help="Root folder produced by wikiextractor (default: %(default)s)"
    )
    p.add_argument(
        "--warnlog", default="extract_warnings.log", type=Path,
        help="Path to the extract warnings log (default: %(default)s)"
    )
    p.add_argument(
        "--out", default="docs.csv", type=Path,
        help="Output CSV file (default: %(default)s)"
    )
    p.add_argument(
        "--workers", type=int, default=int(0.9 * cpu_count()),
        help="Parallel workers (default: 90%% of CPUs)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gather_docs(args.csv, args.indir, args.out, args.warnlog, args.workers)