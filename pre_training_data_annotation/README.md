# Pre-Training Data Annotation

This folder converts x-SWL candidate spans into GLiNER-BioClinX pre-training data.

x-SWL proposes multilingual candidate spans. The LLM annotation step assigns source-language entity types to those candidates under a structured JSON schema. The pooling step reconstructs token-level spans, maps repeated candidates back to all occurrences, and writes balanced GLiNER-format train/validation files.

## Public Dataset

- https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-pre-training-v1.1

## Pipeline

1. `label_infer.py`: read x-SWL JSONL files, reconstruct token-aligned candidate spans, call Qwen3 through an OpenAI-compatible vLLM endpoint, and write labeled JSONL with full metadata.
2. `pool_pre_training_dataset.py`: convert labeled JSONL files into GLiNER samples, balance by language, and write train/validation splits plus statistics.

## Design

- Entity types are assigned in the source language of each passage.
- Candidate IDs are canonicalized so repeated strings can be mapped back to all token-aligned occurrences.
- Training uses 11,800 valid samples per language with random seed 42.
- Validation is balanced by the smallest remaining valid language group.
