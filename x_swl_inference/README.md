# x-SWL Inference

This folder runs the trained x-SWL span proposal model over the balanced GLiNER-BioClinX biomedical free-text corpus.

## Public Dataset

Per-language inference outputs:

- https://huggingface.co/datasets/anthonyyazdaniml/x-swl-large_gliner-bioclin-x-free-text-balanced-12k-v1.1

## Input

Default input dataset:

- https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-free-text-balanced-12k-v1.1

The script filters this dataset by the `language` column and reads passages from the `text` column.

## Model

Default model:

- `anthonyyazdaniml/x-swl-large-v4.1`

x-SWL predicts one generic label: `entity`. These predictions are candidate spans for later LLM-based entity type annotation.

## Output

Each language is written as JSONL, for example:

- `en.jsonl`
- `fr.jsonl`
- `de.jsonl`
- `it.jsonl`
- `es.jsonl`
- `nl.jsonl`
- `pt.jsonl`
- `uk.jsonl`
- `ru.jsonl`
- `zh.jsonl`
- `ja.jsonl`

Each line contains:

```json
{"text":"...","entities":[{"start":0,"end":0,"text":"...","label":"entity","score":0.0}]}
```

## Run

Examples are listed in `inference_commands.txt`.

`x_swl_infer.py` supports resume-by-line-count, batch inference, JSONL streaming, and recursive batch splitting when a batch fails.
