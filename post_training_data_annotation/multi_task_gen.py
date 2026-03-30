#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Union

import backoff
from tqdm.auto import tqdm

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from src.gen_utils import (
    run_ner_chat_call,
    run_hard_ner_chat_call,
    run_rel_chat_call,
    run_extractive_qa_chat_call,
    run_open_task_chat_call,
    TokenFixedLanguageSplitter,
    build_ner_gliner_format,
    build_hard_ner_gliner_format,
    build_rel_gliner_format,
    build_extractive_qa_gliner_format,
    build_open_task_gliner_format,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
print("\nSemaphore GLiNER labeler loaded.\n")


# ------------------------------- Helpers ------------------------------------ #

def nfc(s: str) -> str:
    """Normalize to NFC for stable char offsets (accents, apostrophes)."""
    return unicodedata.normalize("NFC", s)


def load_passages_json(input_path: Path) -> List[str]:
    """
    Load a JSON file containing a list of strings.
    Example: ["text 1", "text 2", ...]
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("--input must be a JSON file with a list of strings.")
    return data


def read_done_uids_if_any(output_path: Path) -> Set[int]:
    done: Set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                uid = obj.get("uid")
                if isinstance(uid, int):
                    done.add(uid)
            except Exception:
                # ignore any partially written/corrupt lines
                continue
    return done


# ------------------------ Task dispatch & building --------------------------- #

class TaskRunner:
    """
    Encapsulates per-task calling + GLiNER formatting.

    Retry strategy:
      - Inner layer (@backoff.on_exception): retry exceptions with the SAME seed.
      - Outer layer (for attempt in max_attempts): retry with seed+1 when:
          * JSON parsing fails (parsed_ok == False), or
          * Coverage check fails (ok_min == False).
    """

    def __init__(
        self,
        language: str,
        require_token_boundaries: bool,
        *,
        max_attempts: int = 3,
        base_seed: int = 42,
    ):
        self.language = language
        self.require_token_boundaries = require_token_boundaries
        self.splitter = TokenFixedLanguageSplitter(language)
        self.max_attempts = max_attempts
        self.base_seed = base_seed

    def _giveup(e: BaseException) -> bool:
        return isinstance(e, APITimeoutError)

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIConnectionError, APIError),
        max_tries=6,
        jitter=backoff.full_jitter,
        giveup=_giveup,
        on_backoff=lambda d: logging.warning(
            "Retrying after %.1fs (attempt %d) — %s",
            d.get("wait", 0.0), d.get("tries", 0), d.get("exception"),
        ),
    )
    
    async def _call_model(self, task: str, passage: str, *, seed: int):
        if task == "ner":
            return await run_ner_chat_call(passage, seed=seed)
        if task == "hard_ner":
            return await run_hard_ner_chat_call(passage, seed=seed)
        if task == "rel":
            return await run_rel_chat_call(passage, seed=seed)
        if task == "extractive_qa":
            return await run_extractive_qa_chat_call(passage, seed=seed)
        if task == "open_task":
            return await run_open_task_chat_call(passage, seed=seed)
        raise ValueError(f"Unsupported task: {task}")

    async def run_one(
        self,
        task: str,
        raw_passage: str,
    ) -> Tuple[
        Optional[str],                        # content (JSON string from model)
        Optional[str],                        # reasoning (if available)
        Optional[Dict[str, Any]],             # gliner_format
        bool,                                 # ok_min (always boolean)
        Optional[str],                        # effective_input_text
        Optional[List[Dict[str, Union[int, str]]]], # tokens
        bool,                                 # model_ok (transport/format success)
    ]:
        passage = nfc(raw_passage.strip())

        last_content: Optional[str] = None
        last_reasoning: Optional[str] = None
        last_gliner: Optional[Dict[str, Any]] = None
        last_ok_min: Optional[bool] = None
        last_effective: Optional[str] = None
        last_tokens: Optional[List[Dict[str, Union[int, str]]]] = None
        model_ok: bool = False
        last_error: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            seed = self.base_seed + attempt
            try:
                resp = await self._call_model(task, passage, seed=seed)

                # Extract content / reasoning defensively
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    content = None
                try:
                    reasoning = resp.choices[0].message.reasoning_content
                except Exception:
                    reasoning = None

                # Parse JSON
                try:
                    parsed = json.loads(content) if content else None
                except Exception:
                    parsed = None

                parsed_ok = parsed is not None

                # If JSON didn't parse, retry with the next seed (no builder call)
                if not parsed_ok:
                    last_content = content
                    last_reasoning = reasoning
                    logging.debug(
                        "Retry trigger on attempt %d with seed=%d — parsed_ok=False; next seed=%d",
                        attempt + 1, seed, seed + 1
                    )
                    continue

                # ---------- Build effective text + tokens + GLiNER ----------
                if task == "ner":
                    effective = passage
                    tokens = self.splitter.split(effective)
                    gliner, ok_min = build_ner_gliner_format(
                        effective, tokens, parsed,
                        require_token_boundaries=self.require_token_boundaries
                    )

                elif task == "hard_ner":
                    prefix = (parsed or {}).get("prompt", "")
                    effective = nfc(f"{prefix.strip()} {passage}".strip())
                    tokens = self.splitter.split(effective)
                    gliner, ok_min = build_hard_ner_gliner_format(
                        effective, tokens, parsed,
                        require_token_boundaries=self.require_token_boundaries
                    )

                elif task == "rel":
                    effective = passage
                    tokens = self.splitter.split(effective)
                    gliner, ok_min = build_rel_gliner_format(
                        effective, tokens, parsed,
                        require_token_boundaries=self.require_token_boundaries
                    )

                elif task == "extractive_qa":
                    q = (parsed or {}).get("question", "")
                    effective = nfc(f"{q.strip()} {passage}".strip())
                    tokens = self.splitter.split(effective)
                    gliner, ok_min = build_extractive_qa_gliner_format(
                        effective, tokens, parsed,
                        require_token_boundaries=self.require_token_boundaries
                    )

                elif task == "open_task":
                    tp = (parsed or {}).get("task_prompt", "")
                    effective = nfc(f"{tp.strip()} {passage}".strip())
                    tokens = self.splitter.split(effective)
                    gliner, ok_min = build_open_task_gliner_format(
                        effective, tokens, parsed,
                        require_token_boundaries=self.require_token_boundaries
                    )

                else:
                    raise ValueError(f"Unsupported task: {task}")

                # Transport + build succeeded
                model_ok = True

                # Save latest attempt results
                last_content = content
                last_reasoning = reasoning
                last_gliner = gliner
                last_ok_min = ok_min
                last_effective = effective
                last_tokens = tokens

                # Coverage failure => retry with next seed
                if not ok_min:
                    logging.debug(
                        "Retry trigger on attempt %d with seed=%d — ok_min=False; next seed=%d",
                        attempt + 1, seed, seed + 1
                    )
                    continue

                # Parsed OK and coverage OK — stop retrying
                break

            except APITimeoutError as te:
                if attempt == self.max_attempts - 1:
                    # surface a clear error on the last attempt
                    last_error = te
                    model_ok = False
                logging.debug("Timeout on attempt %d with seed=%d — switching seed", attempt + 1, seed)
                continue

            except Exception as e:
                # Transport or build failure; try with seed+1
                model_ok = False
                last_error = e
                last_content = None
                last_reasoning = None
                last_gliner = None
                last_ok_min = None
                last_effective = None
                last_tokens = None
                logging.debug(
                    "Model/build failed on attempt %d with seed=%d: %s",
                    attempt + 1, seed, e
                )
                # continue to next attempt

        # Coerce ok_min to boolean for stable downstream consumers
        final_ok_min: bool = bool(last_ok_min) if last_ok_min is not None else False

        if not model_ok or not final_ok_min:
            if last_error:
                logging.error("All attempts failed: %s", last_error)
            elif not final_ok_min:
                logging.warning("All attempts exhausted without meeting ok_min coverage.")

        return (
            last_content,
            last_reasoning,
            last_gliner,
            final_ok_min,
            last_effective,
            last_tokens,
            model_ok,
        )

# ----------------------------- Orchestrator ---------------------------------- #

async def process_passages(
    passages: List[str],
    output_path: Path,
    task: str,
    language: str,
    require_token_boundaries: bool,
    concurrency: int,
    limit: int,
) -> None:
    """
    Concurrently process passages, with resume and JSONL output.
    """
    # Validate
    task = task.lower()
    valid = {"ner", "hard_ner", "rel", "extractive_qa", "open_task"}
    if task not in valid:
        raise ValueError(f"--task must be one of {sorted(valid)}")

    runner = TaskRunner(language, require_token_boundaries)

    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fout = output_path.open("a", encoding="utf-8")

    # Resume support
    done = read_done_uids_if_any(output_path)

    # Apply limit
    if limit and limit > 0:
        passages = passages[:limit]
        logging.info("Limiting to first %d passages", limit)

    # Plan work
    records = [(uid, txt) for uid, txt in enumerate(passages) if uid not in done]
    if not records:
        logging.info("Nothing to do (all done or limit=0).")
        fout.close()
        return

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(records), desc=f"{task} labeling", unit="passage", leave=True)

    async def handle(uid: int, txt: str):
        # Each record is isolated; failures are recorded, not fatal.
        async with sem:
            try:
                content, reasoning, gliner, ok_min, effective, tokens, model_ok = await runner.run_one(task, txt)
            except Exception as e:
                logging.error("uid=%d failed: %s", uid, e)
                content, reasoning, gliner, ok_min, effective, tokens, model_ok = None, None, None, None, None, None, False

            out = {
                "uid": uid,
                "ok": model_ok,
                "require_token_boundaries": require_token_boundaries,
                "min_expected_ok": ok_min,
                "task": task,
                "language": language,
                "gliner_format": gliner,
                "passage": txt,
                "reasoning": reasoning,
                "model_output": content,
                "char_input": effective,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.update(1)

    tasks = [asyncio.create_task(handle(uid, txt)) for uid, txt in records]

    # Drain as they complete (so failures don't block progress)
    for coro in asyncio.as_completed(tasks):
        await coro

    pbar.close()
    fout.close()


# --------------------------------- CLI -------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Smart-semaphore GLiNER labeling over a JSON list of strings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON file containing a list of strings (passages).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL (append/resume-safe).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ner", "hard_ner", "rel", "extractive_qa", "open_task"],
        help="Type of extraction to perform.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code for TokenFixedLanguageSplitter (e.g., 'fr', 'en').",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=48,
        help="Max simultaneous model calls.",
    )
    parser.add_argument(
        "--max",
        dest="limit",
        type=int,
        default=0,
        help="Process first N passages (0 = all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--require-token-boundaries",
        dest="require_token_boundaries",
        action="store_true",
        help="Enforce that spans align to token boundaries.",
    )
    group.add_argument(
        "--no-require-token-boundaries",
        dest="require_token_boundaries",
        action="store_false",
        help="Allow spans to start/end mid-token (not recommended).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        parser.error(f"Input file {args.input} does not exist")

    try:
        passages = load_passages_json(args.input)
    except Exception as e:
        parser.error(str(e))

    # Run async processing
    asyncio.run(
        process_passages(
            passages=passages,
            output_path=args.output,
            task=args.task,
            language=args.language,
            require_token_boundaries=args.require_token_boundaries,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )
    logging.info("✓ Completed — results stored in %s", args.output)


if __name__ == "__main__":
    main()