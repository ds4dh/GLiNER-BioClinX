# Pre-training data annotation

This folder assigns entity types to candidate mentions and assembles the multilingual pre-training NER dataset.

The prepared pre-training dataset is available on [Hugging Face](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-pre-training-v1.1).

## Pipeline

1. `label_infer.py` reads candidate-span JSONL files and assigns source-language entity types through an OpenAI-compatible vLLM endpoint.
2. `pool_pre_training_dataset.py` reconstructs token-level annotations, balances the data by language, and writes GLiNER-format training and validation files with summary statistics.

Annotation examples are provided in `label_infer_example_command.txt`, and the model-serving command is provided in `vllm_server_command.txt`.
