# Biomedical Wikipedia collection

This folder collects biomedical Wikipedia articles used to prepare the multilingual pre-training free-text corpus for GLiNER-BioClinX.

The prepared collections are available on Hugging Face:

- [Biomedical Dutch Wikipedia articles](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_dutch_wiki_articles)
- [Biomedical Portuguese Wikipedia articles](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_portuguese_wiki_articles)
- [Biomedical Ukrainian Wikipedia articles](https://huggingface.co/datasets/anthonyyazdaniml/biomedical_ukrainian_wiki_articles)

## Collection

Articles were selected using PetScan category queries with a category depth of two. Text was extracted from Wikipedia dumps using WikiExtractor.

Each language directory contains:

- `article_ids_query.txt` — inclusion and exclusion categories.
- `article_ids.csv` — selected Wikipedia page IDs.
- `wikiextractor_command.txt` — text-extraction commands.
- `gather_docs.py` — collects the selected articles into a CSV file, using extraction warning logs to exclude pages with template errors.

The resulting articles are used to prepare the [pre-training free-text corpus](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-free-text-balanced-12k-v1.1).
