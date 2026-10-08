# Model training and fine-tuning

This folder contains code for pre-training, post-training, and dataset-specific fine-tuning of GLiNER-BioClinX.

GLiNER-BioClinX is available on [Hugging Face](https://huggingface.co/anthonyyazdaniml/GLiNER-BioClinX).

## Training datasets

- [Pre-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-pre-training-v1.1)
- [Post-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-open-post-training-v1.1)

## Contents

- `pre_and_post_training/` — training code in `train_custom.py`, with separate pre-training and post-training configurations.
- `fine_tuning/` — dataset-specific adaptation code in `finetune.py`, with an example configuration.
