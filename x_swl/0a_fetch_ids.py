#!/usr/bin/env python3
"""
fetch_ids.py – download WikiMed.zip, extract WikiMed.json, stream every
`id_` value into ids.txt (newline-delimited JSON version).

Requires: requests, tqdm   →  pip install requests tqdm
"""

from __future__ import annotations
import io, json, pathlib, sys, zipfile
from typing import Any, Iterable

import requests
from tqdm import tqdm

URL          = "https://zenodo.org/records/5755155/files/WikiMed.zip?download=1"
ZIP_NAME     = "WikiMed.zip"
JSON_MEMBER  = "WikiMed.json"
IDS_OUTFILE  = "ids.txt"
CHUNK        = 1 << 15      # 32 KiB


# --------------------------------------------------------------------------- helpers
def download(url: str, dest: pathlib.Path) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
    if dest.exists():
        print(f"[skip] {dest} already present")
        return
    print(f"[↓] Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        with tqdm(total=total, unit="B", unit_scale=True) as bar, \
             open(dest, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def find_id_keys(obj: Any) -> Iterable[str]:
    """Recursively yield every value whose key is exactly '_id'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_id":
                yield str(v)
            yield from find_id_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_id_keys(item)


# --------------------------------------------------------------------------- main
def main() -> None:
    workdir = pathlib.Path(".").resolve()
    zip_path = workdir / ZIP_NAME

    # 1) download (if needed)
    try:
        download(URL, zip_path)
    except Exception as exc:
        sys.exit(f"[error] download failed: {exc}")

    # 2) open the ZIP & member
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if JSON_MEMBER not in zf.namelist():
                sys.exit(f"[error] {JSON_MEMBER} not found in ZIP")
            zf_member = zf.open(JSON_MEMBER)
            # wrap in TextIOWrapper for automatic UTF-8 decode
            stream = io.TextIOWrapper(zf_member, encoding="utf-8")
            # 3) stream parse & 4) write ids
            ids_path = workdir / IDS_OUTFILE
            with open(ids_path, "w", encoding="utf-8") as out, \
                 tqdm(unit="lines", desc="Scanning") as pbar:
                for line in stream:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    for _id in find_id_keys(obj):
                        out.write(f"{_id}\n")
                    pbar.update()
    except zipfile.BadZipFile:
        sys.exit("[error] downloaded file is not a valid ZIP archive")

    print(f"[✓] ids.txt written at {ids_path.absolute()}")


if __name__ == "__main__":
    main()