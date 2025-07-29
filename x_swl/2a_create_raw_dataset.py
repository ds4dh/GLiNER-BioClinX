import argparse
import logging
from src.extraction import (
    collect_samples,
    build_info_result_parallel,
    save_info_result,
    filter_info_result,
)

def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("Collecting samples from: %s", args.root)

    # Collect raw samples
    data = collect_samples(root=args.root, n=args.n, ext=args.ext)
    logging.info("Collected %d languages", len(data))

    # Build full info result in parallel
    info_result = build_info_result_parallel(data)
    logging.info("Built info results in parallel")

    # Save the full (unfiltered) dataset
    save_info_result(info_result, args.full_output)
    logging.info("Saved full results to %s", args.full_output)

    # Apply filters
    filtered = filter_info_result(
        info_result,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_richness=args.min_richness,
        max_richness=args.max_richness,
    )
    total_filtered = sum(len(s) for s in filtered.values())
    logging.info("Filtered down to %d total entries", total_filtered)

    # Print number of samples per language
    logging.info("Samples per language:")
    for lang, samples in filtered.items():
        logging.info("  %s: %d samples", lang, len(samples))

    # Save the filtered results
    save_info_result(filtered, args.output)
    logging.info("Saved filtered results to %s", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and filter multilingual WikiMed HTML samples")
    parser.add_argument("--root", type=str, default="/scratch/multilingual_wikimed_html",
                        help="Root directory containing HTML files")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of files to process (default: all)")
    parser.add_argument("--ext", type=str, default=".html",
                        help="File extension to filter by (default: .html)")
    parser.add_argument("--min_tokens", type=int, default=100,
                        help="Minimum number of tokens (default: 100)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens (default: 512)")
    parser.add_argument("--min_richness", type=float, default=0.4,
                        help="Minimum richness threshold (default: 0.4)")
    parser.add_argument("--max_richness", type=float, default=0.99,
                        help="Maximum richness threshold (default: 0.99)")
    parser.add_argument("--output", type=str, default="filtered_data.json",
                        help="Output file path for filtered results (default: filtered_data.json)")
    parser.add_argument("--full_output", type=str, default="full_data.json",
                        help="Output file path for full (unfiltered) results (default: full_data.json)")

    args = parser.parse_args()
    main(args)