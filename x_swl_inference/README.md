# Span-proposal inference

This folder applies the span-proposal model to the multilingual pre-training free-text corpus.

The prepared candidate-span dataset is available on [Hugging Face](https://huggingface.co/datasets/anthonyyazdaniml/x-swl-large_gliner-bioclin-x-free-text-balanced-12k-v1.1).

## Input and model

- [Pre-training free-text corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-free-text-balanced-12k-v1.1)
- [Span-proposal model](https://huggingface.co/anthonyyazdaniml/x-swl-large-v4.1)

`x_swl_infer.py` selects the requested language from the `language` column and reads text from the `text` column.

## Output

The script writes one JSONL record per text chunk, containing the original text and predicted candidate spans with character offsets, the generic label `entity`, and confidence scores.

## Run

Examples are provided in `inference_commands.txt`.
