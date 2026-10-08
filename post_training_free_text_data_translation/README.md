# Clinical transcription translation

This folder translates English clinical transcriptions and formats them for entity annotation.

The translated clinical transcriptions are available on [Hugging Face](https://huggingface.co/datasets/anthonyyazdaniml/multilingual-mtsamples).

## Scripts

- `translate.py` translates the texts through an OpenAI-compatible vLLM endpoint.
- `format_mtsamples_for_annotation.py` formats the original English texts and translations into one JSON file per language.
