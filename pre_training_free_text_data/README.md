This folder builds the multilingual free-text pre-training corpus used before annotation.

## Pipeline

1. Run the `0*` scripts to extract and chunk raw text into `data/preprocessed/`.
2. Run `1a_cleaning_main.py` using `cleaning_commands.txt`.
3. Run `2a_deduplicate.py` using `deduplication_commands.txt`.
4. Run `3a_create_combined_datasets.py` to create `all.csv`, `balanced.csv`, and `balanced_12k.csv`.

## Raw Sources

- `0a_mmedc_processor.py`: [MMedC](https://huggingface.co/datasets/Henrychur/MMedC) (`zh`, `de`, `ja`, `ru`).
- `0b_mmc_processor.py`: [Multilingual Medical Corpus](https://huggingface.co/datasets/HiTZ/Multilingual-Medical-Corpus) (`en`, `it`, `es`, `fr`).
- `0c_biolord_processor.py`: [BioLord Dutch MarianMT](https://huggingface.co/datasets/UMCU/biolord_dutch_marianmt) (`nl`, legacy; not used in the final paper pipeline).
- `0d_wiki_uk_processor.py`: [biomedical Ukrainian Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_ukrainian_wiki_articles) (`uk`).
- `0e_wiki_pt_processor.py`: [biomedical Portuguese Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_portuguese_wiki_articles) (`pt`).
- `0f_wiki_nl_processor.py`: [biomedical Dutch Wikipedia](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_dutch_wiki_articles) (`nl`).

## Updating `balanced_12k.csv`

Use `4a_manage_12k_dataset.py` when a language subset changes and you do not want to rerun the full pipeline.

Example used here: replace legacy Dutch with Wikipedia Dutch and add Portuguese.

```bash
python 4a_manage_12k_dataset.py --replace nl
```
