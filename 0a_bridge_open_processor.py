#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer
import semchunk


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load, filter, chunk, and compute stats on the BRIDGE-Open dataset."
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=Path(
            "/home/users/y/yazdani0/.cache/huggingface/hub/datasets--YLab-Open--BRIDGE-Open"
        ),
        help="Path to the local dataset snapshot directory",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="YLab-Open/BRIDGE-Open",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum token length for each text chunk",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum number of tokens to retain a sample",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./data/preprocessed/bridge_open_chunked.csv"),
        help="Output CSV file path for the chunked dataset",
    )
    return parser.parse_args()


def encode_fn_factory(tokenizer: AutoTokenizer) -> callable:
    """
    Create a function to encode text and return token count.
    """
    def encode_fn(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return encode_fn


def main() -> None:
    """Main execution flow."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1) Determine and load the dataset (from local cache or HF snapshot)
    dataset_path = args.local_path
    if not dataset_path.exists():
        logging.info("Local path %s not found. Downloading snapshot...", dataset_path)
        dataset_path = Path(
            snapshot_download(args.dataset_id, repo_type="dataset")
        )
        data_files = dataset_path / "Dataset" / "*.json"
    else:
        data_files = dataset_path / "snapshots" / "*" / "Dataset" / "*.json"

    try:
        logging.info("Loading dataset from %s", data_files)
        ds = load_dataset(
            "json", data_files=str(data_files), split="train"
        )
    except Exception as e:
        logging.warning(
            "Failed to load dataset from %s: %s", data_files, e
        )
        logging.info("Downloading snapshot from Hugging Face hub: %s", args.dataset_id)
        dataset_path = Path(
            snapshot_download(args.dataset_id, repo_type="dataset")
        )
        data_files = dataset_path / "Dataset" / "*.json"
        logging.info("Retry loading dataset from %s", data_files)
        ds = load_dataset(
            "json", data_files=str(data_files), split="train"
        )

    df = ds.to_pandas()

    # 2) Tokenizer and chunker setup
    logging.info(
        "Initializing tokenizer and chunker (chunk size: %d)", args.chunk_size
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-32B", trust_remote_code=True, use_fast=True
    )
    encode_fn = encode_fn_factory(tokenizer)
    chunker = semchunk.chunkerify(encode_fn, chunk_size=args.chunk_size)

    # 3) Chunk and explode
    logging.info("Chunking and exploding text samples...")
    df["chunks"] = df["input"].apply(
        lambda txt: chunker(txt) if isinstance(txt, str) else []
    )
    df = (
        df
        .explode("chunks")
        .rename(columns={"chunks": "text"})
        .reset_index(drop=True)
    )

    # 4) Compute token counts
    logging.info("Computing token counts for each chunk...")
    df["num_tokens"] = df["text"].apply(encode_fn)

    # 5) Filter by minimum token requirement
    logging.info(
        "Filtering samples with at least %d tokens", args.min_tokens
    )
    df = df[df["num_tokens"] >= args.min_tokens].copy()

    # 6) Save processed dataset
    result = df[["language", "num_tokens", "text"]]
    output_path = args.output_path
    logging.info("Saving processed dataset to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logging.info("Processed dataset saved successfully.")

    # 7) Compute and display statistics
    length_stats = (
        result.groupby("language")["num_tokens"]
        .describe()
        .reset_index()
    )
    print(length_stats.to_string(index=False))

    # 8) Save stats CSV alongside, with `_stats` suffix
    stats_path = output_path.with_name(
        output_path.stem + "_stats" + output_path.suffix
    )
    logging.info("Saving statistics to %s", stats_path)
    length_stats.to_csv(stats_path, index=False)
    logging.info("Statistics saved successfully.")


if __name__ == "__main__":
    main()