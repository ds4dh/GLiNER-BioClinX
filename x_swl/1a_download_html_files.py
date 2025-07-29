import time
import os
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import requests
from urllib.parse import quote


CSV_PATH = "multilingual_ids.csv"
OUTDIR = Path("/scratch/multilingual_wikimed_html")
OUTDIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "OfflineParsoid/0.4 (you@example.com)"}
BASE_DELAY = 0.20
MAX_RETRIES = 60
TIMEOUT = 30
SEM = threading.Semaphore(30)

JOURNAL_PATH = Path("downloaded_journal.csv")
_JOURNAL_LOCK = threading.Lock()


def fetch_html(lang: str, title: str):
    """Return (slug, html) or None if the page ultimately fails."""
    slug = quote(title.replace(" ", "_"), safe="()")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{slug}"
    tries = 0

    while True:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        # page not found → skip it
        if resp.status_code == 404:
            return None

        if resp.status_code != 429:
            resp.raise_for_status()
            return slug, resp.text

        wait = int(resp.headers.get("retry-after", 1))
        time.sleep(wait)

        tries += 1
        if tries > MAX_RETRIES:
            print(f"gave up after {MAX_RETRIES} retries: {lang}:{title}")
            return None


def save_html(lang: str, slug: str, html: str) -> None:
    """Write the HTML to html/<lang>/<slug>.html, truncating at the filesystem’s NAME_MAX."""
    lang_dir = OUTDIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    name_max = os.pathconf(str(lang_dir), 'PC_NAME_MAX')
    ext = ".html"
    filename = slug + ext
    if len(filename) > name_max:
        allowed = name_max - len(ext)
        filename = filename[:allowed] + ext
    (lang_dir / filename).write_text(html, encoding="utf-8")


def _load_downloaded() -> set[tuple[str, str]]:
    """Return a set {(lang, title), …} already logged."""
    if not JOURNAL_PATH.exists():
        return set()
    with JOURNAL_PATH.open(newline="", encoding="utf-8") as fh:
        return {(lang, title) for lang, title in csv.reader(fh)}


def _log_download(lang: str, title: str) -> None:
    """Append one entry to the journal and fsync so it is crash-safe."""
    with _JOURNAL_LOCK:  # multiple threads → single writer
        JOURNAL_PATH.parent.mkdir(exist_ok=True)
        with JOURNAL_PATH.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([lang, title])
            fh.flush()
            os.fsync(fh.fileno())


def fetch_and_save(lang: str, title: str) -> None:
    with SEM:
        result = fetch_html(lang, title)
    if result:
        slug, html = result
        save_html(lang, slug, html)
        _log_download(lang, title)
        time.sleep(BASE_DELAY)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    rows = df.dropna(subset=["title"])[["lang", "title"]]

    already = _load_downloaded()
    to_do = [
        (lang, title)
        for lang, title in rows.itertuples(index=False, name=None)
        if (lang, title) not in already
    ]

    if not to_do:
        print("Everything already downloaded ✨")
        return

    with ThreadPoolExecutor(max_workers=50) as pool, tqdm(
        total=len(to_do), desc="Downloading"
    ) as bar:

        futures = [pool.submit(fetch_and_save, lang, title) for lang, title in to_do]

        for f in as_completed(futures):
            f.result()
            bar.update(1)


if __name__ == "__main__":
    main()