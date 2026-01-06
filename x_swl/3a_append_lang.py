#!/usr/bin/env python3
"""
append_lang.py — Append one or more Wikipedia languages to an existing multilingual_ids.csv

Behavior
~~~~~~~~
- Reads an existing CSV with schema: cluster_id, lang, pageid, title
- Computes which cluster_id (English pageids) are missing the requested language(s)
- Fetches ONLY those missing langlinks from enwiki
- Optionally resolves missing foreign pageids (async, semaphore-limited, retry/backoff)
- Writes a merged CSV without losing existing rows

Typical usage
~~~~~~~~~~~~~
If your existing results are in `multilingual_ids.csv`:

```bash
python 3a_append_lang.py multilingual_ids.csv --langs pt
```

Or write to a new file:

```bash
python 3a_append_lang.py multilingual_ids.csv --langs pt --output multilingual_ids_with_pt.csv
```
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import pathlib
from typing import Dict, List, Tuple, Set, Optional

import aiohttp
import async_timeout
import requests
from tqdm.auto import tqdm

API_EN   = "https://en.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "LangLinkFetcher/1.1 (you@example.com)"}
BATCH    = 50
TIMEOUT  = 30

# Resolver defaults (match your original intent)
RESOLVE_IDS = True
RESOLVE_CONCURRENCY = 50
RETRIES  = 10
BACKOFF  = 2.0


class WikipediaTruncatedError(RuntimeError):
    """The API silently dropped pages we asked for."""


def mw_get_sync(url: str, **params):
    resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def read_existing_csv(path: pathlib.Path) -> Dict[int, Dict[str, Dict[str, Optional[object]]]]:
    """
    Returns mapping: {cluster_id -> {lang -> {"pageid": int|None, "title": str}}}
    """
    mapping: Dict[int, Dict[str, Dict[str, Optional[object]]]] = {}

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        expected = {"cluster_id", "lang", "pageid", "title"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must contain columns {sorted(expected)}; found {reader.fieldnames}"
            )

        for row in reader:
            cid_raw = (row.get("cluster_id") or "").strip()
            lang = (row.get("lang") or "").strip()
            title = row.get("title") or ""
            pageid_raw = (row.get("pageid") or "").strip()

            if not cid_raw or not lang:
                continue

            try:
                cid = int(cid_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid cluster_id {cid_raw!r} in {path}") from exc

            pageid: Optional[int]
            if pageid_raw == "":
                pageid = None
            else:
                try:
                    pageid = int(pageid_raw)
                except ValueError:
                    pageid = None  # tolerate older messy exports

            bucket = mapping.setdefault(cid, {})
            # If duplicates exist, prefer the first non-empty pageid/title we saw.
            if lang not in bucket:
                bucket[lang] = {"pageid": pageid, "title": title}
            else:
                if bucket[lang].get("pageid") is None and pageid is not None:
                    bucket[lang]["pageid"] = pageid
                if not (bucket[lang].get("title") or "") and title:
                    bucket[lang]["title"] = title

    if not mapping:
        raise ValueError(f"No rows found in {path}")
    return mapping


def fetch_langlinks(pageids: List[int], wanted_langs: List[str]) -> Dict[int, Dict[str, Dict]]:
    """
    Return {en_pid -> {lang -> {pageid, title}}} (pageid may be None).
    Includes an "en" entry for each en_pid with its title.
    """
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
            if "title" not in page:  # deleted/missing
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


async def resolve_single(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    lang: str,
    title: str,
) -> Optional[int]:
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


async def fill_missing_pageids(mapping: Dict[int, Dict[str, Dict]], only_langs: Set[str] | None = None) -> None:
    """
    Resolve pageid where missing (None) for non-en languages.
    If only_langs is provided, resolve only those language codes.
    """
    if not RESOLVE_IDS:
        return

    to_resolve: List[Tuple[str, str, Dict]] = []
    seen: Set[Tuple[str, str]] = set()

    for bucket in mapping.values():
        for lang, info in bucket.items():
            if lang == "en":
                continue
            if only_langs is not None and lang not in only_langs:
                continue
            if info.get("pageid") is None and info.get("title"):
                key = (lang, str(info["title"]))
                if key not in seen:
                    to_resolve.append((lang, str(info["title"]), info))
                    seen.add(key)

    if not to_resolve:
        return

    sem = asyncio.Semaphore(RESOLVE_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=None, connect=TIMEOUT, sock_read=TIMEOUT)

    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
        bar = tqdm(total=len(to_resolve), desc="Resolving pageids", unit="page", leave=False)
        tasks: List[Tuple[asyncio.Task, Dict]] = []
        for lang, title, info in to_resolve:
            task = asyncio.create_task(resolve_single(session, sem, lang, title))
            task.add_done_callback(lambda _fut: bar.update(1))
            tasks.append((task, info))

        for task, info in tasks:
            info["pageid"] = await task

        bar.close()


def write_csv(mapping: Dict[int, Dict[str, Dict]], path: pathlib.Path) -> None:
    rows = sum(len(v) for v in mapping.values())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "lang", "pageid", "title"])
        with tqdm(total=rows, desc="CSV", unit="row", leave=False) as bar:
            for en_pid in sorted(mapping.keys()):
                langs = mapping[en_pid]
                for lang, info in sorted(langs.items()):
                    writer.writerow([en_pid, lang, info.get("pageid") or "", info.get("title") or ""])
                    bar.update(1)
    print(f"Wrote {path} ({rows} rows, {len(mapping)} clusters)")


def main() -> None:
    global RESOLVE_IDS, RESOLVE_CONCURRENCY, RETRIES, TIMEOUT
    ap = argparse.ArgumentParser(description="Append languages to an existing multilingual_ids.csv")
    ap.add_argument("csv_file", help="existing CSV produced by fetch_multilingual_ids.py")
    ap.add_argument("--langs", default="pt", help="comma-separated language codes to append (default: pt)")
    ap.add_argument(
        "--output",
        default=None,
        help="output CSV (default: overwrite input CSV safely)",
    )
    ap.add_argument("--no-resolve-ids", action="store_true", help="do not resolve missing foreign pageids")
    ap.add_argument("--resolve-concurrency", type=int, default=RESOLVE_CONCURRENCY, help="resolver semaphore size")
    ap.add_argument("--retries", type=int, default=RETRIES, help="retry budget for resolver")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="HTTP timeout seconds")
    args = ap.parse_args()
    
    RESOLVE_IDS = not args.no_resolve_ids
    RESOLVE_CONCURRENCY = max(1, args.resolve_concurrency)
    RETRIES = max(1, args.retries)
    TIMEOUT = max(1, args.timeout)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    in_path = pathlib.Path(args.csv_file)
    out_path = pathlib.Path(args.output) if args.output else in_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    wanted_langs = [s for s in (x.strip() for x in args.langs.split(",")) if s]
    wanted_set = set(wanted_langs)

    mapping = read_existing_csv(in_path)

    # Determine which clusters need fetching for these langs
    cluster_ids = sorted(mapping.keys())
    missing_per_lang: Dict[str, List[int]] = {lang: [] for lang in wanted_langs}

    for cid in cluster_ids:
        bucket = mapping[cid]
        for lang in wanted_langs:
            if lang not in bucket:
                missing_per_lang[lang].append(cid)

    total_missing = sum(len(v) for v in missing_per_lang.values())
    if total_missing == 0:
        print(f"Nothing to do: all requested languages already exist in {in_path}")
        return

    # Fetch per language (keeps requests focused and progress clearer)
    for lang in wanted_langs:
        ids_needed = missing_per_lang[lang]
        if not ids_needed:
            continue

        logging.info("Fetching %s for %d clusters", lang, len(ids_needed))
        fetched = fetch_langlinks(ids_needed, [lang])

        # Merge fetched info into existing mapping
        for cid, langs in fetched.items():
            bucket = mapping.setdefault(cid, {})
            # Ensure we keep or add en title
            if "en" not in bucket and "en" in langs:
                bucket["en"] = langs["en"]
            # Add the requested language if present
            if lang in langs:
                bucket[lang] = langs[lang]

    # Resolve missing pageids only for the newly requested languages
    if RESOLVE_IDS:
        asyncio.run(fill_missing_pageids(mapping, only_langs=wanted_set))

    # Write merged output safely
    write_csv(mapping, tmp_path)
    tmp_path.replace(out_path)
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()