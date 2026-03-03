#!/usr/bin/env python
"""label_infer.py  – robust, resumable, complete

Pipeline
========
1. Read a GLiNER JSON‑Lines file (*text + entities*).
2. For every record
   • tokenise with `LanguageSplitter`
   • build `id_map`, `unique_ordered_spans`, `uid_to_tags` (via `build_id_map`)
3. Send a prompt to OpenAI with a semaphore of at most `--concurrency` in‑flight
   requests (retries on failure with exponential back‑off).
4. **Immediately flush** the following structure to the output JSONL so a
   crash never loses more than one record:

   ```json
   {
     "uid": <int>,
     "tokenized_text": [...],
     "id_map": {...},
     "unique_ordered_spans": [...],
     "uid_to_tags": {...},
     "annotations": {...},
     "reasoning": "..."
   }
   ```
5. On resume the script loads the output file, builds a set of processed `uid`s
   and skips them – order doesn’t matter.

Usage
-----
python label_infer.py \
    --input ./x_swl_data/mmc_chunked_fr_1000.jsonl \
    --output ./x_swl_data/mmc_chunked_fr_1000_labeled.jsonl \
    --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import backoff
from openai import AsyncOpenAI
from tqdm.auto import tqdm

try:
    from src.few_shot_utils import EN_SYS_PROMPT, UI_1, AO_1, UI_2, AO_2
    from src.label_x_swl import (
        one_chat,
        SCHEMA_LABELING,
        GENConfig,
        LanguageSplitter,
        build_id_map,
    )
except ModuleNotFoundError as e:
    raise SystemExit(f"❌ Import error: {e}. Adjust PYTHONPATH.")

logging.getLogger("httpx").setLevel(logging.WARNING)   # hides the “HTTP Request: POST … 200 OK” line
logging.getLogger("openai._base_client").setLevel(logging.DEBUG)
print("\nPackages loaded.\n")

###############################################################################
# Helpers – tokenisation & span reconstruction
###############################################################################

_splitter = LanguageSplitter()

def reconstruct(passage: str, entities: List[Dict[str, Any]]):
    tokens = _splitter.split(passage, level="token")
    id_map, unique_spans, uid2tags = build_id_map(
        text=passage,
        entities=entities,
        greedy=True,
        case_sensitive=False,
        allow_nested=True,
        tokens=tokens,
        require_token_boundaries=True,
    )
    return tokens, id_map, unique_spans, uid2tags


def prompt_for(passage: str, spans: List[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": EN_SYS_PROMPT},
        {"role": "user", "content": UI_1},
        {"role": "assistant", "content": AO_1},
        {"role": "user", "content": UI_2},
        {"role": "assistant", "content": AO_2},
        {"role": "user", "content": f"""Context:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\nCandidate (ID, span) for Step 1, Step 2 and Step 3:\n```\n{spans}\n```"""}
    ]

###############################################################################
# OpenAI caller with retry + semaphore
###############################################################################

def log_backoff(details):
    exc = details.get("exception")
    logging.warning(
        "Retrying after %.1fs (attempt %d) — %s",
        details.get("wait", 0),
        details["tries"],
        exc,
    )


def make_caller(sem: asyncio.Semaphore, client: AsyncOpenAI, cfg: GENConfig, max_retries: int):
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=max_retries,
        jitter=backoff.full_jitter,
        on_backoff=log_backoff,
    )
    async def _call(prompt):
        async with sem:
            return await one_chat(client, prompt, cfg=cfg, guided_json=SCHEMA_LABELING)

    return _call

###############################################################################
# Async pipeline
###############################################################################

async def process(records: List[Dict[str, Any]], done_uids: Set[int], args):
    cfg = GENConfig()
    if args.base_url:
        cfg.base_url = args.base_url
        logging.info("Overriding base URL to %s", cfg.base_url)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    openai_call = make_caller(sem, client, cfg, args.max_retries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fout = args.output.open("a", encoding="utf-8")

    todo = [r for r in records if r["uid"] not in done_uids]
    pbar = tqdm(total=len(todo), desc="Labeling", unit="passage", leave=True)

    async def handle(rec):
        try:
            tokens, id_map, uniq_spans, uid2tags = reconstruct(rec["text"], rec["entities"])
        except Exception as e:
            logging.error("Span reconstruction failed (uid=%d): %s", rec["uid"], e)
            fout.write(
                json.dumps(
                    {
                        "uid": rec["uid"],
                        "text": None,
                        "tokenized_text": None,
                        "id_map": None,
                        "unique_ordered_spans": None,
                        "uid_to_tags": None,
                        "annotations": None,
                        "reasoning": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
            pbar.update(1)
            return                                  # ← skip OpenAI call for this record

        # ── Span reconstruction succeeded → proceed with OpenAI ─────────────────
        resp = await openai_call(prompt_for(rec["text"], uniq_spans))
        try:
            annotations = json.loads(resp.choices[0].message.content)
            reasoning = resp.choices[0].message.reasoning_content
        except Exception as e:                      # assistant JSON parse error
            logging.error("Assistant parse error (uid=%d): %s", rec["uid"], e)
            annotations, reasoning = None, None

        fout.write(
            json.dumps(
                {
                    "uid": rec["uid"],
                    "text": rec["text"],
                    "tokenized_text": tokens,
                    "id_map": id_map,
                    "unique_ordered_spans": uniq_spans,
                    "uid_to_tags": uid2tags,
                    "annotations": annotations,
                    "reasoning": reasoning,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fout.flush()
        pbar.update(1)

    tasks = [asyncio.create_task(handle(r)) for r in todo]
    for t in asyncio.as_completed(tasks):
        await t

    pbar.close()
    fout.close()

###############################################################################
# CLI & orchestration
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Semaphore OpenAI labeling with full metadata")
    parser.add_argument("--input", type=Path, required=True, help="GLiNER JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--concurrency", type=int, default=10, help="Max simultaneous OpenAI calls")
    parser.add_argument("--max-retries", type=int, default=6, help="Retry attempts")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Process first N records")
    parser.add_argument("--base-url", type=str, default=None, help="Override API base URL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Verbosity")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    
    if not args.input.exists():
        parser.error(f"Input file {args.input} does not exist")

    # Load input with uid assignment
    records: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            rec["uid"] = idx  # stable integer uid
            records.append(rec)
    if args.limit > 0:
        records = records[: args.limit]
        logging.info("Limiting to first %d records", args.limit)
    if not records:
        parser.error("No records after limiting/filtering")

    # Determine already‑processed uids from output file
    done_uids: Set[int] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_uids.add(json.loads(line)["uid"])
                except Exception:  # corrupted line? ignore
                    continue
        logging.info("Resuming – %d records already done", len(done_uids))

    asyncio.run(process(records, done_uids, args))
    logging.info("✓ Completed – results stored in %s", args.output)


if __name__ == "__main__":
    main()