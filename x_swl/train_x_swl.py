import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback

from gliner import GLiNER, GLiNERConfig
from gliner.data_processing.collator import DataCollator, DataCollatorWithPadding
from gliner.data_processing import WordsSplitter, GLiNERDataset
from gliner.training import Trainer, TrainingArguments
from gliner.utils import load_config_as_namespace


def load_data(file_path):
    """Load and validate dataset file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found")

    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dataset at {file_path} should be a list of examples")

    return data


def main(args):
    # Set TOKENIZERS_PARALLELISM to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Load configuration from YAML file
    config = load_config_as_namespace(args.config)

    # Use the CLI --output_dir, which is a required argument
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Data Loading ---
    print(f"Loading training data from: {config.train_data}")
    train_data = load_data(config.train_data)
    
    # Use 'val_data_dir' from the config for the validation path
    print(f"Loading validation data from: {config.val_data_dir}")
    val_data = load_data(config.val_data_dir)

    print(f'Training dataset size: {len(train_data)}')
    print(f'Validation dataset size: {len(val_data)}')
    
    random.seed(42)
    random.shuffle(train_data)
    print('Training dataset is shuffled...')

    # --- Device Setup ---
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Model and Tokenizer Initialization (Preserving original logic) ---
    if hasattr(config, 'prev_path') and config.prev_path is not None and config.prev_path.lower() != 'null':
        print(f"Fine-tuning existing GLiNER model from: {config.prev_path}")
        model = GLiNER.from_pretrained(config.prev_path)
        tokenizer = AutoTokenizer.from_pretrained(config.prev_path)
        model_config = model.config
    else:
        print(f"Initializing new GLiNER model from base: {config.model_name}")
        model_config = GLiNERConfig(**vars(config))
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length=config.max_len)
        words_splitter = WordsSplitter(config.words_splitter_type)
        model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)

        if not config.labels_encoder:
            print("Adding special tokens for entity markers.")
            model_config.class_token_index=len(tokenizer)
            tokenizer.add_tokens([config.ent_token, config.sep_token], special_tokens=True)
            model_config.vocab_size = len(tokenizer)
            model.resize_token_embeddings(
                [config.ent_token, config.sep_token], 
                set_class_token_index=False,
                add_tokens_to_tokenizer=False
            )
    
    model.to(device)

    # --- Optional features driven by the config file ---
    if hasattr(config, 'freeze_token_rep') and config.freeze_token_rep:
        print("Freezing the language model weights.")
        model.model.token_rep_layer.bert_layer.model.requires_grad_(False)
    else:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(True)
        
    # --- Data Collator and Dataset Selection (Preserving original logic) ---
    if hasattr(config, 'new_data_schema') and config.new_data_schema:
        print("Using new data schema with GLiNERDataset wrapper.")
        train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
        eval_dataset = GLiNERDataset(val_data, model_config, tokenizer, words_splitter, get_negatives=False)
        data_collator = DataCollatorWithPadding(model_config)
    else:
        print("Using default data schema and DataCollator.")
        train_dataset = train_data
        eval_dataset = val_data
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # --- Training Arguments Setup ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        save_strategy="steps",
        save_steps=config.eval_every,
        logging_steps=config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=8,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        use_cpu=False,
        report_to="none",
        bf16=True,
    )

    # --- Callback Setup ---
    callbacks = []
    if hasattr(config, 'early_stopping_patience') and config.early_stopping_patience > 0:
        print(f"Early stopping enabled with patience: {config.early_stopping_patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    # --- Trainer Initialization ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # --- Execute Training ---
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---\n")


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=4 python train_gliner.py --config ./configs/config-large.yaml --output_dir ./gliner-x-swl-large-v4
    parser = argparse.ArgumentParser(description="GLiNER Custom Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints and logs.")
    
    script_args = parser.parse_args()
    main(script_args)