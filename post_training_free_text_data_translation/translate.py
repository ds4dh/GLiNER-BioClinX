#!/usr/bin/env python3
"""
Biomedical text translation via local vLLM (OpenAI-compatible API).

Translates a JSON array of strings into one or more target languages
with crash-recovery checkpointing and concurrent requests.

Usage
-----
    python translate.py corpus.json
    python translate.py corpus.json -t French German Japanese -s English
    python translate.py corpus.json --batch-size 64 --seed 123
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SOURCE = "English"
DEFAULT_TARGETS = [
    "French", "Italian", "German", "Spanish", "Dutch",
    "Portuguese", "Ukrainian", "Russian", "Chinese", "Japanese",
]

SYSTEM_PROMPT = (
    "You are an expert biomedical and clinical translator. "
    "Translate the following text from {src} to {tgt}. "
    "Return only the translation."
)

# ── data types ───────────────────────────────────────────────────────────────


@dataclass
class TranslationJob:
    source_lang: str
    target_lang: str
    texts: list[str]
    output_path: Path
    checkpoint_path: Path
    results: list[dict[str, str | None] | None] = field(default_factory=list)
    completed: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.results:
            self.results = [None] * len(self.texts)

    @property
    def pending_indices(self) -> list[int]:
        return [i for i in range(len(self.texts)) if i not in self.completed]

    def load_checkpoint(self) -> int:
        if not self.checkpoint_path.exists():
            return 0
        data: dict[str, Any] = json.loads(self.checkpoint_path.read_text("utf-8"))
        recovered = 0
        for idx_str, entry in data.get("translations", {}).items():
            idx = int(idx_str)
            if 0 <= idx < len(self.texts) and entry is not None:
                self.results[idx] = entry
                self.completed.add(idx)
                recovered += 1
        return recovered

    def save_checkpoint(self) -> None:
        payload = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "total": len(self.texts),
            "completed": len(self.completed),
            "translations": {
                str(i): self.results[i] for i in sorted(self.completed)
            },
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.checkpoint_path)

    def save_output(self) -> None:
        self.output_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2), "utf-8"
        )


# ── API interaction ──────────────────────────────────────────────────────────


async def translate_one(
    client: httpx.AsyncClient,
    idx: int,
    text: str,
    system: str,
    api_base: str,
    model: str,
    seed: int,
    timeout: float,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    error_log: Path | None = None,
    target_lang: str = "",
) -> dict[str, str | None]:
    """Acquire semaphore slot, send request, return on completion.

    Retry strategy: first half of attempts use the original seed,
    second half use seed+1 to work around potential generation artifacts.
    """
    url = f"{api_base}/v1/chat/completions"
    halfway = max_retries // 2

    async with semaphore:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            current_seed = seed if attempt <= halfway else seed + 1
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "seed": current_seed,
                "stream": False,
            }

            try:
                resp = await client.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                return {
                    "translation": (message.get("content") or "").strip(),
                    "reasoning": (message.get("reasoning_content") or "").strip() or None,
                }
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as exc:
                last_err = exc
                wait = min(2 ** attempt, 60)
                phase = "seed" if attempt <= halfway else "seed+1"

                # Log every failed attempt to file
                if error_log is not None:
                    entry: dict[str, Any] = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "index": idx,
                        "target_lang": target_lang,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "phase": phase,
                        "seed_used": current_seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "text_preview": text[:200],
                        "text_length": len(text),
                    }
                    if isinstance(exc, httpx.HTTPStatusError):
                        entry["status_code"] = exc.response.status_code
                        try:
                            entry["response_body"] = exc.response.text[:500]
                        except Exception:
                            pass
                    # Append JSON line (atomic-ish for short writes)
                    with open(error_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                logging.warning(
                    "idx=%d attempt %d/%d [%s] failed (%s: %s). Retrying in %ds…",
                    idx, attempt, max_retries, phase,
                    type(exc).__name__, exc, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Index {idx}: all {max_retries} attempts failed. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        ) from last_err


async def run_job(
    job: TranslationJob,
    api_base: str,
    model: str,
    seed: int,
    timeout: float,
    max_retries: int,
    batch_size: int,
    checkpoint_every: int,
    error_log: Path | None = None,
    progress: Progress | None = None,
    lang_task: TaskID | None = None,
    overall_task: TaskID | None = None,
) -> int:
    """Run all pending translations. Returns the number of failures."""
    system = SYSTEM_PROMPT.format(src=job.source_lang, tgt=job.target_lang)
    pending = job.pending_indices
    semaphore = asyncio.Semaphore(batch_size)
    completions_since_ckpt = 0
    failures = 0

    def tick_progress() -> None:
        nonlocal completions_since_ckpt
        completions_since_ckpt += 1
        if progress is not None:
            if lang_task is not None:
                progress.advance(lang_task)
            if overall_task is not None:
                progress.advance(overall_task)
        if completions_since_ckpt >= checkpoint_every:
            job.save_checkpoint()
            completions_since_ckpt = 0

    # Handle empty strings synchronously
    for idx in pending[:]:
        text = job.texts[idx]
        if not text or not text.strip():
            job.results[idx] = {"source": text, "translation": text, "reasoning": None}
            job.completed.add(idx)
            tick_progress()

    # Wrapper: translate, store result, advance progress — all per-task
    async def process(client: httpx.AsyncClient, idx: int) -> None:
        nonlocal failures
        try:
            result = await translate_one(
                client, idx, job.texts[idx], system, api_base, model,
                seed, timeout, max_retries, semaphore,
                error_log=error_log,
                target_lang=job.target_lang,
            )
            result["source"] = job.texts[idx]
            job.results[idx] = result
            job.completed.add(idx)
            tick_progress()
        except Exception as exc:
            failures += 1
            logging.error("Index %d failed permanently: %s", idx, exc)

    # Rebuild pending after empties were handled
    pending = job.pending_indices

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(process(client, idx)) for idx in pending]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Always checkpoint whatever succeeded
    job.save_checkpoint()
    job.save_output()
    return failures


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Translate a JSON list of biomedical strings via vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="JSON file containing a list of strings")
    p.add_argument(
        "-s", "--source-lang", default=DEFAULT_SOURCE,
        help="source language name (default: %(default)s)",
    )
    p.add_argument(
        "-t", "--target-langs", nargs="+", default=DEFAULT_TARGETS,
        help="target language names (default: %(default)s)",
    )
    p.add_argument("-o", "--output-dir", type=Path, default=None)
    p.add_argument("--api-base", default="http://localhost:8847")
    p.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=48,
                    help="max concurrent requests to vLLM (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    console = Console(stderr=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path: Path = args.input
    if not input_path.exists():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        sys.exit(1)

    texts: list[str] = json.loads(input_path.read_text("utf-8"))
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        console.print("[red]Input must be a JSON array of strings.[/red]")
        sys.exit(1)

    out_dir: Path = args.output_dir or input_path.with_name(f"{input_path.stem}_translations")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── filter targets ───────────────────────────────────────────────────
    targets = [t for t in args.target_langs if t.lower() != args.source_lang.lower()]

    # ── manifest ─────────────────────────────────────────────────────────
    manifest = {
        "source_file": str(input_path.resolve()),
        "source_lang": args.source_lang,
        "target_langs": targets,
        "model": args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "system_prompt": SYSTEM_PROMPT,
        "num_strings": len(texts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

    # ── error log ────────────────────────────────────────────────────────
    error_log = out_dir / "errors.jsonl"

    # ── build jobs & load checkpoints ────────────────────────────────────
    jobs: list[TranslationJob] = []
    already_done_total = 0
    for tgt in targets:
        tag = tgt.lower().replace(" ", "_")
        job = TranslationJob(
            source_lang=args.source_lang,
            target_lang=tgt,
            texts=texts,
            output_path=out_dir / f"{tag}.json",
            checkpoint_path=out_dir / f".{tag}.checkpoint.json",
        )
        if args.resume:
            job.load_checkpoint()
        already_done_total += len(job.completed)
        jobs.append(job)

    overall_total = len(texts) * len(targets)

    console.print(
        f"\n  [bold]{len(texts)}[/bold] strings × "
        f"[bold]{len(targets)}[/bold] languages = "
        f"[bold]{overall_total}[/bold] translations  "
        f"(concurrency: [bold]{args.batch_size}[/bold])"
    )
    if already_done_total:
        console.print(
            f"  [green]{already_done_total}[/green] already done (resumed)\n"
        )
    else:
        console.print()

    # ── progress display ─────────────────────────────────────────────────
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    )

    total_failures = 0

    with progress:
        overall_task = progress.add_task(
            "Overall", total=overall_total, completed=already_done_total,
        )

        for job in jobs:
            if not job.pending_indices:
                job.save_output()
                progress.add_task(
                    f"  {job.target_lang}", total=len(texts),
                    completed=len(texts), visible=True,
                )
                continue

            lang_task = progress.add_task(
                f"  {job.target_lang}", total=len(texts),
                completed=len(job.completed),
            )

            failures = await run_job(
                job,
                api_base=args.api_base,
                model=args.model,
                seed=args.seed,
                timeout=args.timeout,
                max_retries=args.max_retries,
                batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every,
                error_log=error_log,
                progress=progress,
                lang_task=lang_task,
                overall_task=overall_task,
            )
            total_failures += failures

    if total_failures:
        console.print(
            f"\n  [bold yellow]Partial completion.[/bold yellow] "
            f"{total_failures} translations failed. "
            f"Re-run the same command to retry them.\n"
            f"  Error log: [underline]{error_log}[/underline]"
        )
    console.print(f"  Results in [underline]{out_dir}/[/underline]\n")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()