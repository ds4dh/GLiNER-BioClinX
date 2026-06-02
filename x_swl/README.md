# x-SWL: Multilingual Span Weak Labeling

This folder builds the x-SWL training data and trains the multilingual span proposal model.

x-SWL is not a biomedical entity classifier. It learns one task: propose entity-like spans across languages with the generic label `entity`. The later LLM annotation stage assigns source-language entity types.

## Public Dataset

Training and validation splits:

- https://huggingface.co/datasets/anthonyyazdaniml/x-swl-v4.1

## Pipeline

1. `0a_fetch_ids.py`: download WikiMed from Zenodo and extract English Wikipedia page IDs.
2. `0b_fetch_multilingual_ids.py`: resolve multilingual Wikipedia equivalents with langlinks.
3. `1a_download_html_files.py`: download Parsoid HTML for each language/title pair.
4. `2a_create_raw_dataset.py`: extract lead-section text and weak spans from bold, italic, and linked text.
5. `2b_create_tokenized_dataset.py`: filter, uniformly sample, tokenize with Stanza, and write `train_uniform.json` and `val_uniform.json`.
6. `train_x_swl.py`: train the GLiNER span proposal model with `configs/config-large.yaml`.

For adding a language after the first run:

1. `3a_append_lang.py`: append missing langlinks to an existing multilingual ID CSV.
2. `4a_download_appended_html_files.py`: download only the newly appended language HTML.
3. Re-run `2a_create_raw_dataset.py` and `2b_create_tokenized_dataset.py`.

## Key Design

- Source: WikiMed medical Wikipedia IDs, expanded to multilingual Wikipedia.
- Weak signals: lead-section bold text, italic text, and anchor text.
- Raw filtering: 100-512 Qwen tokens and richness between 0.4 and 0.99.
- Uniform split: 2,000 train and 400 validation examples per language.
- Training label: `entity` only.
- Model role: high-recall multilingual span proposal before LLM entity type annotation.
