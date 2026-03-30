from datasets import load_dataset, concatenate_datasets
import stanza, json
from tqdm.auto import tqdm

# --- load ---
ds = load_dataset("bigbio/medmentions", name="medmentions_full_source", trust_remote_code=True)
ds = concatenate_datasets([ds["train"], ds["test"], ds["validation"]])

with open("semmap.json", "r") as f:
    semmap = json.load(f)
    semmap = {k: v.lower().capitalize() for k, v in semmap.items()}

# --- stanza ---
try:
    nlp = stanza.Pipeline("en", processors="tokenize", tokenize_pretokenized=False, use_gpu=True, verbose=False)
except:
    stanza.download("en")
    nlp = stanza.Pipeline("en", processors="tokenize", tokenize_pretokenized=False, use_gpu=True, verbose=False)

def build_text_and_remapper(passages):
    kept = []
    for p in sorted(passages, key=lambda x: x["offsets"][0][0]):        # original order
        if "title" in (p.get("type") or "").lower():
            continue
        t = "".join(p["text"])
        os, oe = p["offsets"][0]
        kept.append((os, oe, t))
    new_starts, acc = [], 0
    for i, (_, _, t) in enumerate(kept):
        new_starts.append(acc + i)                                      # i newlines before chunk i
        acc += len(t)
    new_text = "\n".join(t for _, _, t in kept)
    return kept, new_starts, new_text

def remap_span_to_new(s, e, kept, new_starts):
    for i, (os, oe, _) in enumerate(kept):
        if s >= os and e <= oe:                                         # fully inside one kept chunk
            d = new_starts[i] - os
            return s + d, e + d
    return -1, -1

def tokenize(text):
    doc = nlp(text)
    toks, spans = [], []
    for sent in doc.sentences:
        for tok in sent.tokens:
            toks.append(tok.text)
            spans.append((tok.start_char, tok.end_char))
    return toks, spans

def char_to_tok(cs, ce, spans):
    si = ej = -1
    for i, (ts, te) in enumerate(spans):
        if ts < ce and te > cs:                                         # any overlap
            if si == -1: si = i
            ej = i
    return si, ej

def process(ex):
    kept, new_starts, text = build_text_and_remapper(ex["passages"])
    toks, spans = tokenize(text)
    triples = set()                                                      # (si, ej, type)
    for ent in ex.get("entities", []):
        for (s, e) in ent.get("offsets", []):
            ns, ne = remap_span_to_new(s, e, kept, new_starts)
            if ns < 0: 
                continue
            si, ej = char_to_tok(ns, ne, spans)
            if si < 0: 
                continue
            for t in (ent.get("semantic_type_id") or []):
                triples.add((si, ej, t))
    ner = [[int(si), int(ej), str(t)] for (si, ej, t) in sorted(triples)]
    return {"tokenized_text": toks, "ner": ner}

# --- build outputs without Arrow ---
out = [process(ex) for ex in tqdm(ds)]

# Map and save
filtered_out = []  # step 3 result

for rec in out:
    # --- Step 1: map labels through semmap, drop if semmap.get returns None
    ner = []
    for i, j, lbl in rec["ner"]:
        mapped = semmap.get(lbl)
        if mapped is not None:
            ner.append([i, j, mapped])

    # --- Step 2: skip examples with no surviving NER triples
    if len(ner) < 1:
        continue

    # --- Step 3: add to final list
    filtered_out.append({"lang": "en", "script": "Latn", "tokenized_text": rec["tokenized_text"], "ner": ner})

mapped_out = filtered_out

with open("./medmentions.json", "w", encoding="utf-8") as f:
    json.dump(mapped_out, f, ensure_ascii=False, indent=2)