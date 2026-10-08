# GLiNER-BioClinX

Data preparation, annotation, training, and fine-tuning code for GLiNER-BioClinX.

## Repository contents

- `wikipedia_articles/` — biomedical Wikipedia collection.
- `pre_training_free_text_data/` — multilingual corpus preparation.
- `x_swl/` and `x_swl_inference/` — span-proposal training and inference.
- `pre_training_data_annotation/` — entity typing and pre-training dataset assembly.
- `post_training_free_text_data/`, `post_training_free_text_data_translation/`,
  and `post_training_data_annotation/` — post-training data preparation,
  translation, and annotation.
- `gliner-bioclinx-training/` — pre-training, post-training, and fine-tuning.

## Resources

- [Model](https://huggingface.co/anthonyyazdaniml/GLiNER-BioClinX)
- [Pre-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-pre-training-v1.1)
- [Post-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-open-post-training-v1.1)

Scripts, configurations, and additional resource links are provided in the
individual directories.

## License

The code is released under the MIT License. Model and dataset licenses are
provided on their respective Hugging Face pages.
