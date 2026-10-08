# Pre-training free-text corpus

This folder prepares the multilingual medical free-text corpus used before entity annotation.

The prepared corpus is available on [Hugging Face](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-free-text-balanced-12k-v1.1).

## Sources

- `0a_mmedc_processor.py` — [MMedC](https://huggingface.co/datasets/Henrychur/MMedC), for Chinese, German, Japanese, and Russian.
- `0b_mmc_processor.py` — [Multilingual Medical Corpus](https://huggingface.co/datasets/HiTZ/Multilingual-Medical-Corpus), for English, Italian, Spanish, and French.
- `0d_wiki_uk_processor.py` — [biomedical Ukrainian Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_ukrainian_wiki_articles).
- `0e_wiki_pt_processor.py` — [biomedical Portuguese Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_portuguese_wiki_articles).
- `0f_wiki_nl_processor.py` — [biomedical Dutch Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_dutch_wiki_articles).

## Pipeline

1. Run the source scripts listed above to extract and chunk text into `data/preprocessed/`.
2. Run `1a_cleaning_main.py` using `cleaning_commands.txt`.
3. Run `2a_deduplicate.py` using `deduplication_commands.txt`.
4. Run `3a_create_combined_datasets.py` to produce `data/final/balanced_12k.csv`, containing 12,000 text chunks per language.
