import pandas as pd
from multiprocessing import Pool
from tqdm.auto import tqdm
import argparse
import os
import stanza
import sys
from cleaning_worker import process_single_text, SUPPORTED_LANGUAGES, LANG_MAP

def evaluate_text(args):
    """
    Wrapper for passing arguments to process_single_text for multiprocessing.
    """
    text, lang, thresholds = args
    return process_single_text(text, lang, thresholds)

def process_csv(input_csv: str, output_csv: str, language: str, lang_csv_label: str, thresholds: dict):
    """
    Load, process, and save a CSV file with quality-based filtering.
    - Filters by a specific language label
    - Applies multilingual cleaning and quality thresholds
    - Saves the filtered data and a detailed report
    """
    # 1. Load and prepare the DataFrame
    print(f"Loading CSV: {input_csv}")
    try:
        df = pd.read_csv(input_csv, usecols=["language", "num_tokens", "text"])
    except ValueError as e:
        print(f"Error: Could not find required columns. Ensure CSV contains 'language', 'num_tokens', 'text'. Details: {e}")
        return

    # Filter by the language label provided by the user
    print(f"Filtering for rows where 'language' column is '{lang_csv_label}'...")
    initial_rows = len(df)
    df = df[df['language'] == lang_csv_label].copy()
    lang_filtered_rows = len(df)
    print(f"Found {lang_filtered_rows} rows for label '{lang_csv_label}' (out of {initial_rows} total).")

    if lang_filtered_rows == 0:
        print("No rows match the specified language label. Exiting.")
        return

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df.dropna(subset=['text'], inplace=True)
    df = df[df["text"] != ""].copy()
    total_rows = len(df)
    print(f"Processing {total_rows} non-empty rows.")

    # 2. Process all texts using multiprocessing
    args_list = [(row['text'], language, thresholds) for _, row in df.iterrows()]

    print(f"Processing texts using a pool of {20} workers...")
    with Pool(processes=20) as pool:
        results = list(tqdm(pool.imap(evaluate_text, args_list), total=total_rows))

    # 3. Collect results and analyze failures
    df[['passed', 'scores']] = pd.DataFrame(results, index=df.index)

    # Initialize counters for removal reasons
    removal_reasons = {f"failed_threshold_{key}": 0 for key in thresholds.keys()}
    removal_reasons.update({
        "language_mismatch": 0,
        "language_detection_failed": 0,
        "pipeline_failure": 0,
        "empty_or_invalid": 0,
        "other": 0
    })

    rejected_df = df[~df['passed']]
    for _, row in rejected_df.iterrows():
        scores = row.get('scores', {})
        reason = scores.get('reason', 'other')
        if reason in removal_reasons:
            removal_reasons[reason] += 1
        else:
            removal_reasons["other"] += 1
            
    # 4. Filter, clean, and save the final DataFrame
    filtered_df = df[df['passed']].drop(columns=['passed', 'scores'])
    filtered_df = filtered_df.drop_duplicates(subset="text")
    remaining_rows = len(filtered_df)

    stanza_lang_code = LANG_MAP.get(language, language)
    filtered_df['language'] = stanza_lang_code

    print(f"\nSaving {remaining_rows} cleaned rows to: {output_csv}")
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    print("Cleaned CSV saved successfully!")

    # 5. Generate and save the summary report
    removed_count = total_rows - remaining_rows
    report_path = os.path.splitext(output_csv)[0] + "_report.txt"
    
    report_content = (
        f"--- Quality Filtering Summary Report ---\n"
        f"Input File: {input_csv}\n"
        f"Output File: {output_csv}\n\n"
        f"Total rows before filtering: {total_rows}\n"
        f"Total rows after filtering:  {remaining_rows}\n"
        f"Total rows removed:          {removed_count} ({removed_count / total_rows:.2%})\n\n"
        f"--- Reasons for Removal ---\n"
        f"(Note: Each document is counted once based on the first failure condition it met)\n"
    )
    
    # Add reasons to the report string, renaming for clarity
    for reason, count in removal_reasons.items():
        if count > 0:
            clean_reason = reason.replace("failed_threshold_", "")
            report_content += f"- {clean_reason:<30}: {count}\n"

    print("\n" + report_content)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and filter a CSV file using multilingual quality heuristics.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument('--language', type=str, required=True, help="The langdetect language code to process (e.g., 'en', 'fr', 'zh-cn').")
    parser.add_argument('--lang_csv_label', type=str, required=True, help="The string in the 'language' column of the CSV to filter by.")
    
    # Threshold arguments
    parser.add_argument('--download_stanza_models', type=bool, default=False)
    parser.add_argument('--special_char_ratio', type=float, default=0.3, help="Max ratio of non-alphanumeric tokens.")
    parser.add_argument('--min_sentence_count', type=int, default=4, help="Min number of sentences.")
    parser.add_argument('--min_avg_words_per_sentence', type=float, default=10, help="Min average words per sentence.")
    parser.add_argument('--max_capitalization_ratio', type=float, default=0.2, help="Max ratio of uppercase to all letters.")
    parser.add_argument('--min_lexical_diversity', type=float, default=0.1, help="Min ratio of unique tokens to total tokens.")
    parser.add_argument('--max_repetition_score', type=float, default=0.2, help="Max frequency of the most common token.")
    parser.add_argument('--max_newline_to_sentence_ratio', type=float, default=1/3, help="Max ratio of newline groups to sentences.")
    
    args = parser.parse_args()

    # Validate that the requested language is supported by the worker
    if args.language not in SUPPORTED_LANGUAGES:
        print(f"Error: Language '{args.language}' is not supported by the cleaning worker.", file=sys.stderr)
        print(f"Supported languages are: {', '.join(sorted(SUPPORTED_LANGUAGES))}", file=sys.stderr)
        sys.exit(1)

    # Create thresholds dictionary from command-line arguments
    thresholds = {
        "special_char_ratio": ("max", args.special_char_ratio),
        "sentence_count": ("min", args.min_sentence_count),
        "avg_words_per_sentence": ("min", args.min_avg_words_per_sentence),
        "capitalization_ratio": ("max", args.max_capitalization_ratio),
        "lexical_diversity": ("min", args.min_lexical_diversity),
        "repetition_score": ("max", args.max_repetition_score),
        "newline_to_sentence_ratio": ("max", args.max_newline_to_sentence_ratio),
    }

    if args.download_stanza_models:
        print("Downloading Stanza models...")
        for lang in {"en", "fr", "de", "it", "es", "nl", "uk", "ru", "zh", "ja"}:
            stanza.download(lang, verbose=True, logging_level='INFO')
        print("Stanza model download complete.")

    process_csv(args.input_csv, args.output_csv, args.language, args.lang_csv_label, thresholds)