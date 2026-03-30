#!/usr/bin/env python3
"""
Format MIMIC translations for annotation.

Takes the original source JSON (list of strings) and the translation directory 
(list of dicts containing source, translation, and reasoning), and outputs a 
single directory where every file is a clean list of strings named by its 
language code (e.g., mimic_en.json, mimic_fr.json).

Usage
-----
python format_mimic_for_annotation.py \
    --original ../post_training_free_text_data/data/MIMIC/mimic_en.json \
    --translations ../post_training_free_text_data/data/MIMIC/mimic_en_translations \
    --output ../post_training_free_text_data/data/MIMIC/annotation_ready
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Map target language names to standard ISO language codes
LANG_CODES = {
    "english": "en",
    "french": "fr",
    "italian": "it",
    "german": "de",
    "spanish": "es",
    "dutch": "nl",
    "portuguese": "pt",
    "ukrainian": "uk",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Format translated JSONs into lists of strings.")
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to the original source file (e.g., mimic_en.json)"
    )
    parser.add_argument(
        "--translations",
        type=Path,
        required=True,
        help="Path to the directory containing the translated JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to 'annotation_ready' next to the original file."
    )
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.original.exists():
        logging.error(f"Original file not found: {args.original}")
        sys.exit(1)

    if not args.translations.exists() or not args.translations.is_dir():
        logging.error(f"Translations directory not found or is not a dir: {args.translations}")
        sys.exit(1)

    # Set up output directory
    out_dir = args.output
    if out_dir is None:
        out_dir = args.original.parent / "annotation_ready"

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing formatted files to: {out_dir}")

    # 1. Process the original file
    try:
        original_data = json.loads(args.original.read_text("utf-8"))
        if not isinstance(original_data, list):
            raise ValueError("Original JSON must be a list of strings.")

        # Save the original file to the output directory using its standardized name
        source_lang_code = LANG_CODES.get("english", "en")  # Assuming English is the source
        original_out_path = out_dir / f"mimic_{source_lang_code}.json"

        original_out_path.write_text(json.dumps(original_data, ensure_ascii=False, indent=2), "utf-8")
        logging.info(f"Saved original (en) -> {original_out_path.name} ({len(original_data)} items)")

    except Exception as e:
        logging.error(f"Failed to process original file: {e}")
        sys.exit(1)

    # 2. Process the translated files
    for translation_file in args.translations.glob("*.json"):
        # Skip manifests and checkpoints
        if translation_file.name == "manifest.json" or translation_file.name.startswith("."):
            continue

        lang_name = translation_file.stem.lower()  # e.g., "french"
        lang_code = LANG_CODES.get(lang_name, lang_name)  # e.g., "fr" (fallback to full name if not in dict)

        try:
            trans_data = json.loads(translation_file.read_text("utf-8"))

            # Extract just the "translation" string, handling null/failed entries
            formatted_texts = []
            for i, item in enumerate(trans_data):
                if item and "translation" in item and item["translation"] is not None:
                    formatted_texts.append(item["translation"])
                else:
                    # Fallback if a translation failed permanently or was left null
                    logging.warning(f"Missing translation in {lang_name} at index {i}. Inserting empty string.")
                    formatted_texts.append("")

            # Check length alignment
            if len(formatted_texts) != len(original_data):
                logging.warning(
                    f"Count mismatch for {lang_name}: original has {len(original_data)}, "
                    f"translation has {len(formatted_texts)}."
                )

            # Write to the new formatted directory
            out_file = out_dir / f"mimic_{lang_code}.json"
            out_file.write_text(json.dumps(formatted_texts, ensure_ascii=False, indent=2), "utf-8")
            logging.info(f"Saved {lang_name} ({lang_code}) -> {out_file.name} ({len(formatted_texts)} items)")

        except Exception as e:
            logging.error(f"Failed to process {translation_file.name}: {e}")

    logging.info("Formatting complete.")

if __name__ == "__main__":
    main()