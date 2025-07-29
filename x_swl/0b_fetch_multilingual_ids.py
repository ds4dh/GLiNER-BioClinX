#!/usr/bin/env python3
"""fetch_multilingual_ids.py – Fetch multilingual Wikipedia page IDs (CSV)

Highlights
~~~~~~~~~~
* **Concurrent** resolution of foreign `pageid`s with an asyncio semaphore
  capped at **50**.
* Robust **3-try exponential back-off** on the resolver requests.
* Same CSV schema::  cluster_id, lang, pageid, title
* Progress bars for every phase.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import pathlib
from typing import Dict, List, Tuple, Set

import aiohttp
import async_timeout
import requests
from tqdm.auto import tqdm

API_EN   = "https://en.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "LangLinkFetcher/1.1 (you@example.com)"}
BATCH    = 50          # ids per langlinks call
TIMEOUT  = 30          # seconds for every HTTP request
RESOLVE_IDS = True
RESOLVE_CONCURRENCY = 50  # semaphore size for pageid resolver
RETRIES  = 10           # per-URL retry budget
BACKOFF  = 2.0         # exponential factor

# ───────────────────────── Exceptions ───────────────────────────────

class WikipediaTruncatedError(RuntimeError):
    """The API silently dropped pages we asked for."""

# ───────────────────────── Utilities ────────────────────────────────

def read_ids(path: pathlib.Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.lstrip().startswith("#"):
            try:
                ids.append(int(s))
            except ValueError as exc:
                raise ValueError(f"Invalid integer {s!r} in {path}") from exc
    if not ids:
        raise ValueError("No numeric IDs found in input file.")
    return ids

# ───────────────────────── Sync API (langlinks) ─────────────────────

def mw_get_sync(url: str, **params):
    resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_langlinks(pageids: List[int], wanted_langs: List[str]) -> Dict[int, Dict[str, Dict]]:
    """Return {en_pid -> {lang -> {pageid, title}}} (pageid may be None)."""
    result: Dict[int, Dict[str, Dict]] = {}

    for start in tqdm(range(0, len(pageids), BATCH), desc="Langlinks", unit="batch", leave=False):
        chunk = pageids[start : start + BATCH]
        data = mw_get_sync(
            API_EN,
            action="query",
            pageids="|".join(map(str, chunk)),
            prop="langlinks",
            lllimit="max",
            format="json",
            formatversion="2",
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            raise WikipediaTruncatedError("API returned 0 pages for non-empty chunk")

        received, skipped = set(), set()

        for page in pages:
            pid = page.get("pageid")
            if pid is None:
                continue
            if "title" not in page:  # deleted
                skipped.add(pid)
                continue
            received.add(pid)
            bucket = result.setdefault(pid, {})
            bucket["en"] = {"pageid": pid, "title": page["title"]}
            for ll in page.get("langlinks", []):
                lang = ll["lang"]
                if wanted_langs and lang not in wanted_langs:
                    continue
                bucket[lang] = {"pageid": ll.get("pageid"), "title": ll["title"]}

        if received != (set(chunk) - skipped):
            diff = (set(chunk) - skipped) - received
            raise WikipediaTruncatedError(f"Wikimedia truncated: missing {sorted(diff)}")

    return result

# ───────────────────────── Async resolver ───────────────────────────

async def resolve_single(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                         lang: str, title: str) -> int | None:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "info",
        "format": "json",
        "formatversion": "2",
    }
    backoff = 1.0
    for attempt in range(RETRIES):
        try:
            async with sem, async_timeout.timeout(TIMEOUT):
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("query", {}).get("pages", [{}])[0].get("pageid")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == RETRIES - 1:
                return None
            await asyncio.sleep(backoff)
            backoff *= BACKOFF
    return None


async def fill_missing_pageids(mapping: Dict[int, Dict[str, Dict]]):
    to_resolve: List[Tuple[str, str, Dict]] = []  # (lang, title, info-dict)
    seen: Set[Tuple[str, str]] = set()

    if not RESOLVE_IDS:
        return

    for bucket in mapping.values():
        for lang, info in bucket.items():
            if info["pageid"] is None and lang != "en":
                key = (lang, info["title"])
                if key not in seen:
                    to_resolve.append((lang, info["title"], info))
                    seen.add(key)

    if not to_resolve:
        return

    sem = asyncio.Semaphore(RESOLVE_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None, connect=TIMEOUT, sock_read=TIMEOUT)
    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
        bar = tqdm(total=len(to_resolve), desc="Resolving pageids", unit="page", leave=False)
        tasks = []
        for lang, title, info in to_resolve:
            task = asyncio.create_task(resolve_single(session, sem, lang, title))
            task.add_done_callback(lambda _fut: bar.update(1))
            tasks.append((task, info))
        for task, info in tasks:
            info["pageid"] = await task
        bar.close()

# ───────────────────────── CSV writer ───────────────────────────────

def write_csv(mapping: Dict[int, Dict[str, Dict]], path: pathlib.Path) -> None:
    rows = sum(len(v) for v in mapping.values())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "lang", "pageid", "title"])
        with tqdm(total=rows, desc="CSV", unit="row", leave=False) as bar:
            for en_pid, langs in mapping.items():
                for lang, info in sorted(langs.items()):
                    writer.writerow([en_pid, lang, info["pageid"] or "", info["title"]])
                    bar.update(1)
    print(f"Wrote {path} ({rows} rows, {len(mapping)} clusters)")

# ───────────────────────── CLI ──────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve multilingual Wikipedia page-IDs (numeric).")
    ap.add_argument("ids_file", help="file with English page-IDs, one per line")
    ap.add_argument("--langs", default="fr,de,it,es,nl,uk,ru,zh,ja",
                    help="comma-separated language codes (default major EU + Asian)")
    ap.add_argument("--output", default="multilingual_ids.csv",
                    help="CSV output (default multilingual_ids.csv)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ids   = read_ids(pathlib.Path(args.ids_file))
    langs = [s for s in (x.strip() for x in args.langs.split(",")) if s]

    try:
        mapping = fetch_langlinks(ids, langs)
        asyncio.run(fill_missing_pageids(mapping))
        write_csv(mapping, pathlib.Path(args.output))
    except WikipediaTruncatedError as exc:
        logging.error("Aborted: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()