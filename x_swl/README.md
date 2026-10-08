# Span-proposal training

This folder prepares multilingual Wikipedia training data and trains the span-proposal model used before LLM-based entity typing.

The prepared [span-proposal training dataset](https://huggingface.co/datasets/anthonyyazdaniml/x-swl-v4.1) and trained [span-proposal model](https://huggingface.co/anthonyyazdaniml/x-swl-large-v4.1) are available on Hugging Face.

The model predicts candidate mention boundaries using the generic label `entity`. Entity types are assigned in the subsequent annotation stage.

## Pipeline

1. `0a_fetch_ids.py` downloads WikiMed and extracts English Wikipedia page IDs.
2. `0b_fetch_multilingual_ids.py` resolves the corresponding Wikipedia pages in other languages.
3. `1a_download_html_files.py` downloads the Wikipedia HTML.
4. `2a_create_raw_dataset.py` extracts lead-section text and candidate spans from bold, italic, and linked text, then filters the resulting instances.
5. `2b_create_tokenized_dataset.py` samples 2,000 training and 400 validation instances per language, tokenizes them with Stanza, and writes `train_uniform.json` and `val_uniform.json`.
6. `train_x_swl.py` trains the model using `configs/config-large.yaml`.

`3a_append_lang.py` and `4a_download_appended_html_files.py` support extending the collection to additional languages.
