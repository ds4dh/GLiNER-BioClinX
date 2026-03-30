"""
Convert the filtered PII health dataset into GLiNER training format
using stanza English word tokenizer.

GLiNER format (per sample):
{
    "lang": "en",
    "script": "Latn",
    "tokenized_text": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    "ner": [[0, 1, "Person"], [5, 5, "Location"]]
}

Where ner entries are [start_token_idx, end_token_idx, label] with INCLUSIVE indices.

Labels are minimally transformed from the original dataset codes:
  - Spacing and casing are normalized for GLiNER compatibility
  - Semantic content is preserved to stay faithful to the dataset creators' intent

Optionally, contiguous Given name + Surname spans can be merged into
a composite "Full name" label (enabled by default; disable with --no-merge-names).
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import stanza
from tqdm import tqdm


# ============================================================
# Label mapping: dataset codes → GLiNER-friendly labels
#
# Design principles:
#   1. Stay faithful to the original dataset label semantics
#   2. Only first letter capitalized, unless conventional (e.g. "ID")
#   3. Expand abbreviations for readability, but don't rename concepts
# ============================================================
LABEL_MAP = {
    "GIVENNAME":        "Given name",
    "SURNAME":          "Surname",
    "TELEPHONENUM":     "Phone number",
    "TIME":             "Time",
    "DATE":             "Date",
    "TITLE":            "Title",
    "CITY":             "City",
    "EMAIL":            "Email address",
    "STREET":           "Street address",
    "BUILDINGNUM":      "Building number",
    "AGE":              "Age",
    "IDCARDNUM":        "ID card number",
    "SEX":              "Sex",
    "GENDER":           "Gender",
    "ZIPCODE":          "Zip code",
    "PASSPORTNUM":      "Passport number",
    "DRIVERLICENSENUM": "Driver's license number",
    "SOCIALNUM":        "Social security number",
    "TAXNUM":           "Tax ID number",
    "CREDITCARDNUMBER": "Credit card number",
}

# Label used for merged given name + surname spans
MERGED_NAME_LABEL = "Full name"


def map_label(raw_label: str) -> str | None:
    """Map a raw dataset label to a GLiNER-friendly label.
    Returns None if the label is unknown (will be skipped)."""
    return LABEL_MAP.get(raw_label)


# ============================================================
# Character-offset → token-index alignment
# ============================================================
def build_token_char_map(tokens: list[dict]) -> list[tuple[str, int, int]]:
    """
    From stanza tokens build a list of (text, char_start, char_end)
    where char_end is EXCLUSIVE (matching Python slice semantics).
    """
    return [(tok["text"], tok["start_char"], tok["end_char"]) for tok in tokens]


def char_span_to_token_span(
    entity_start: int,
    entity_end: int,
    token_map: list[tuple[str, int, int]],
) -> tuple[int, int] | None:
    """
    Given a character-level span [entity_start, entity_end) and a list of
    (token_text, char_start, char_end), return the inclusive token-level
    span [first_tok_idx, last_tok_idx] that overlaps with the entity.

    Returns None if no token overlaps.
    """
    first_tok = None
    last_tok = None

    for tok_idx, (_text, tok_start, tok_end) in enumerate(token_map):
        # Check overlap: token [tok_start, tok_end) ∩ entity [entity_start, entity_end)
        if tok_start < entity_end and tok_end > entity_start:
            if first_tok is None:
                first_tok = tok_idx
            last_tok = tok_idx

    if first_tok is None:
        return None

    return (first_tok, last_tok)


# ============================================================
# Name merging logic
# ============================================================
def merge_contiguous_names(ner: list[list]) -> list[list]:
    """
    Find contiguous (Given name, Surname) or (Surname, Given name) pairs
    and add a merged "Full name" span covering both.

    Two spans are considered contiguous if end_of_first + 1 == start_of_second
    (i.e. adjacent token indices with no gap).

    The original fine-grained entities are preserved; the merged entity
    is added alongside them.
    """
    name_labels = {"Given name", "Surname"}

    # Collect name entities with their original indices
    name_entities = [
        (i, ner_entry) for i, ner_entry in enumerate(ner)
        if ner_entry[2] in name_labels
    ]

    merged = []
    used = set()

    for i in range(len(name_entities)):
        if i in used:
            continue

        idx_a, (start_a, end_a, label_a) = name_entities[i]

        for j in range(i + 1, len(name_entities)):
            if j in used:
                continue

            idx_b, (start_b, end_b, label_b) = name_entities[j]

            # Must be different label types (given + surname)
            if label_a == label_b:
                continue

            # Must be contiguous (adjacent tokens, no gap)
            if end_a + 1 == start_b:
                merged.append([start_a, end_b, MERGED_NAME_LABEL])
                used.add(i)
                used.add(j)
                break
            elif end_b + 1 == start_a:
                merged.append([start_b, end_a, MERGED_NAME_LABEL])
                used.add(i)
                used.add(j)
                break

    return ner + merged


# ============================================================
# Convert one row
# ============================================================
def convert_row(
    source_text: str,
    privacy_mask,  # numpy array or list of dicts
    stanza_nlp,
    merge_names: bool = False,
) -> dict | None:
    """
    Convert a single dataset row to GLiNER format.

    Returns a dict with keys "tokenized_text" and "ner",
    or None if the row should be skipped.
    """
    # Tokenize with stanza
    doc = stanza_nlp(source_text)

    # Flatten all tokens across sentences, preserving char offsets
    all_tokens = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            all_tokens.append({
                "text": token.text,
                "start_char": token.start_char,
                "end_char": token.end_char,
            })

    if not all_tokens:
        return None

    tokenized_text = [tok["text"] for tok in all_tokens]
    token_map = build_token_char_map(all_tokens)

    # Convert entity spans
    ner = []
    for entity in privacy_mask:
        if not isinstance(entity, dict):
            continue

        raw_label = entity.get("label")
        char_start = entity.get("start")
        char_end = entity.get("end")

        if raw_label is None or char_start is None or char_end is None:
            continue

        mapped_label = map_label(raw_label)
        if mapped_label is None:
            # Unknown label — skip this entity
            continue

        token_span = char_span_to_token_span(char_start, char_end, token_map)
        if token_span is None:
            continue

        ner.append([token_span[0], token_span[1], mapped_label])

    # Skip rows with zero entities after mapping
    if not ner:
        return None

    # Optionally merge contiguous name spans
    if merge_names:
        ner = merge_contiguous_names(ner)

    return {
        "lang": "en",
        "script": "Latn",
        "tokenized_text": tokenized_text,
        "ner": ner,
    }


# ============================================================
# Main conversion pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert PII health parquet to GLiNER JSON format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/ai4privacy-pii/pii_health.parquet",
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./pii_health.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--no-merge-names",
        action="store_true",
        default=False,
        help="Disable adding composite 'Full name' spans from contiguous Given name + Surname.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional: limit number of rows to process (for debugging).",
    )
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    if args.max_rows is not None:
        df = df.head(args.max_rows)
        print(f"Limited to {len(df)} rows (--max-rows)")

    merge_names = not args.no_merge_names

    # ── Print config ──
    print(f"\nConfiguration:")
    print(f"  merge-names: {merge_names}")
    print(f"  output:      {args.output}")
    print(f"\nLabel mapping:")
    for code, label in LABEL_MAP.items():
        print(f"  {code:25s} → {label}")
    if merge_names:
        print(f"  {'(merged)':25s} → {MERGED_NAME_LABEL}")
    print()

    # ── Initialize stanza ──
    print("Initializing stanza English tokenizer...")
    stanza.download("en", processors="tokenize", verbose=False)
    nlp = stanza.Pipeline("en", processors="tokenize", verbose=False)

    # ── Convert ──
    gliner_data = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        source_text = row.get("source_text")
        privacy_mask = row.get("privacy_mask")

        if not isinstance(source_text, str) or not source_text.strip():
            skipped += 1
            continue

        result = convert_row(
            source_text, privacy_mask, nlp, merge_names=merge_names
        )

        if result is None:
            skipped += 1
            continue

        gliner_data.append(result)

    print(f"\nConverted: {len(gliner_data)}")
    print(f"Skipped:   {skipped}")

    # ── Collect label stats ──
    label_counts = {}
    for sample in gliner_data:
        for _, _, label in sample["ner"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution in GLiNER output:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} {count:>6d}")

    # ── Validate: spot-check a few samples ──
    print("\n── Validation: first 3 samples ──")
    for i, sample in enumerate(gliner_data[:3]):
        tokens = sample["tokenized_text"]
        print(f"\n  Sample {i}:")
        print(f"    Text (first 15 tokens): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        for start, end, label in sample["ner"]:
            entity_tokens = tokens[start : end + 1]
            print(f"    [{start}, {end}] {label:30s} → \"{' '.join(entity_tokens)}\"")

    # ── Save ──
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gliner_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(gliner_data)} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()