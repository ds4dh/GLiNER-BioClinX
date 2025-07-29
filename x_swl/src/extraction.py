# src/extraction.py

from __future__ import annotations
from pathlib import Path
import os
from typing import Any, List, Tuple, Dict
import re
import json

from bs4 import BeautifulSoup, SoupStrainer
from lxml import html as lxml_html
import unicodedata as ud
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# --- All your other functions are correct and unchanged ---

def normalise_spaces(text: str) -> str:
    if not hasattr(normalise_spaces, "_table"):
        normalise_spaces._table = {cp: 0x20 for cp in range(0x110000) if ud.category(chr(cp)) == "Zs" and cp != 0x20}
    return text.translate(normalise_spaces._table)

Span = Tuple[int, int]

def _collect_spans(text: str, tokens: set[str]) -> List[Span]:
    spans: list[Span] = []
    for token in sorted(tokens, key=len, reverse=True):
        if not token: continue
        pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    return spans

def extract_wikimedia_lead(src: str | Path, *, encoding: str = "utf-8", filter_short_tokens: bool = True) -> dict[str, Any]:
    html_source: str
    if isinstance(src, (str, Path)) and Path(src).is_file():
        html_source = Path(src).read_text(encoding=encoding)
    else:
        html_source = str(src)
    strainer = SoupStrainer("section", attrs={"data-mw-section-id": "0"})
    soup = BeautifulSoup(html_source, "lxml", parse_only=strainer)
    lead = soup.select_one('section[data-mw-section-id="0"]')
    if lead is None:
        raise ValueError("No lead section found")
    for el in lead.select(".shortdescription, style, script, link, .infobox, .navbox, .mw-empty-elt, sup.mw-ref, sup.reference"):
        el.decompose()
    lines = [norm for tag in lead.find_all(["p", "ul", "ol", "dl"], recursive=False) if (norm := normalise_spaces(lxml_html.fromstring(str(tag)).text_content()).strip())]
    lead_text = "\n".join(lines)
    bold_set = {b.get_text(" ", strip=True) for b in lead.select("b, strong")}
    italic_set = {i.get_text(" ", strip=True) for i in lead.select("i, em")}
    links = {txt for a in lead.select("a") if (txt := a.get_text(" ", strip=True))}
    bold_italic, bold_only, italic_only = bold_set & italic_set, bold_set - italic_set, italic_set - bold_set
    tokens = bold_italic | bold_only | italic_only | links
    if filter_short_tokens:
        tokens = {tok for tok in tokens if len(tok) >= 3}
    spans: list[Span] = _collect_spans(lead_text, tokens)
    return {"lead_text": lead_text, "bold_italic": bold_italic, "bold_only": bold_only, "italic_only": italic_only, "links": links, "spans": spans}

def collect_samples(root, n=None, *, ext=".html"):
    root = os.path.expanduser(str(root))
    samples = {}
    for lang_entry in os.scandir(root):
        if not lang_entry.is_dir(): continue
        lang_files = [Path(f.path) for f in os.scandir(lang_entry.path) if f.is_file() and f.name.endswith(ext)][:n or None]
        if lang_files: samples[lang_entry.name] = lang_files
    return samples

# Your original tokenizer initialization. It works on Linux/macOS due to fork's
# copy-on-write, and it is not the source of the hang.
_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

def process_wikimedia_lead(instance):
    try:
        info = extract_wikimedia_lead(instance)
    except ValueError:
        return None
    lead_text: str = info["lead_text"]
    spans: list[tuple[int, int]] = info["spans"]
    char_len = len(lead_text)
    richness = (sum(e - s for s, e in spans) / char_len) if char_len else 0.0
    qwen_tokens = len(_tokenizer.tokenize(lead_text))
    return {"lead_text": lead_text, "spans": spans, "richness": richness, "char_len": char_len, "qwen_tokens": qwen_tokens}

# --- THE CORRECTED PARALLEL FUNCTION ---

def build_info_result_parallel(
    src: Dict[str, List[Path]], *, workers: int | None = None
) -> Dict[str, List[dict]]:
    """
    Computes results in parallel using the user's fast language-by-language
    approach, but with manual chunking to prevent hangs on very large datasets.
    """
    if workers is None:
        cpu_total = os.cpu_count() or 1
        workers = max(1, int(cpu_total * 0.8))

    # This constant defines the maximum number of tasks to submit at once.
    # It prevents the initial hang without sacrificing performance.
    CHUNK_SIZE = 10_000

    info_result = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for lang, lang_data in tqdm(src.items(), desc="Lang", unit="lang"):
            
            lang_results = []
            # This is the tqdm bar for the articles within a single language.
            with tqdm(total=len(lang_data), unit="art", desc=f"{lang} articles") as pbar:
                # Manually iterate over the data in chunks
                for i in range(0, len(lang_data), CHUNK_SIZE):
                    chunk = lang_data[i:i + CHUNK_SIZE]
                    
                    # executor.map is now only ever given a reasonably-sized chunk.
                    # This call is now fast and will not hang.
                    results_iter = executor.map(process_wikimedia_lead, chunk)
                    
                    # Collect results from the chunk and filter out Nones
                    chunk_results = [r for r in results_iter if r is not None]
                    lang_results.extend(chunk_results)
                    
                    # Update the progress bar after each chunk is processed
                    pbar.update(len(chunk))
            
            info_result[lang] = lang_results
            
    return info_result


def filter_info_result(info_result, min_tokens, max_tokens, min_richness, max_richness):
    def is_good(inst):
        return (min_tokens <= inst["qwen_tokens"] <= max_tokens and min_richness <= inst["richness"] <= max_richness)
    return {lang: [inst for inst in instances if is_good(inst)] for lang, instances in info_result.items() if any(is_good(inst) for inst in instances)}

def save_info_result(info_result: Dict[str, List[dict]], outfile: str | os.PathLike):
    with Path(outfile).open("w", encoding="utf-8") as fp:
        json.dump(info_result, fp, ensure_ascii=False, separators=(",", ":"))