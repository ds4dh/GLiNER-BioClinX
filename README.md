# GLiNER-BioClinX

GLiNER-BioClinX is a small multilingual model for medical named entity recognition and personally identifiable information detection. Entity types are specified using natural-language labels.

This repository contains data preparation, annotation, training, and fine-tuning code.

## Resources

- [GLiNER-BioClinX model](https://huggingface.co/anthonyyazdaniml/GLiNER-BioClinX)
- [Pre-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-pre-training-v1.1)
- [Post-training dataset](https://huggingface.co/datasets/anthonyyazdaniml/gliner-bioclin-x-open-post-training-v1.1)

## Inference

Install GLiNER with Stanza support:

```bash
pip install "gliner[stanza]==0.2.29"
```

Load the model and specify the entity types to extract in `labels`:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("anthonyyazdaniml/GLiNER-BioClinX")

text = "John Smith was prescribed metformin for type 2 diabetes."
labels = ["person name", "medication", "disease or condition"]

entities = model.predict_entities(text, labels, threshold=0.5)
for entity in entities:
    print(f"{entity['text']} -> {entity['label']}")
```

Expected output:

```text
John Smith -> person name
metformin -> medication
type 2 diabetes -> disease or condition
```

## Repository contents

- `wikipedia_articles/` — biomedical Wikipedia collection.
- `pre_training_free_text_data/` — multilingual corpus preparation.
- `x_swl/` and `x_swl_inference/` — span-proposal training and inference.
- `pre_training_data_annotation/` — entity typing and pre-training dataset assembly.
- `post_training_free_text_data/`, `post_training_free_text_data_translation/`,
  and `post_training_data_annotation/` — post-training data preparation,
  translation, and annotation.
- `gliner-bioclinx-training/` — pre-training, post-training, and fine-tuning.

Scripts, configurations, and additional resource links are provided in the
individual directories.

## License

The code is released under the MIT License. Model and dataset licenses are
provided on their respective Hugging Face pages.
