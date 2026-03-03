import json
import glob
import random
import os
import statistics
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42

# Training target
TRAIN_SIZE_PER_LANG = 11800

# Mapping language codes to their primary writing scripts (ISO 15924)
LANG_TO_SCRIPT = {
    "de": "Latn", "en": "Latn", "es": "Latn", "fr": "Latn",
    "it": "Latn", "ja": "Jpan", "nl": "Latn", "pt": "Latn",
    "ru": "Cyrl", "uk": "Cyrl", "zh": "Hans"  
}

def calculate_split_stats(dataset):
    """Helper function to calculate statistics for a specific split."""
    stats = {}
    for sample in dataset:
        lang = sample["lang"]
        if lang not in stats:
            stats[lang] = {
                "samples": 0,
                "labels_per_sample": [],
                "unique_labels_per_sample": [],
                "unique_labels": set(),
                "words_per_sample": []
            }
        
        ner_list = sample["ner"]
        words = sample["tokenized_text"]
        
        stats[lang]["samples"] += 1
        stats[lang]["labels_per_sample"].append(len(ner_list))
        stats[lang]["words_per_sample"].append(len(words))
        
        # Calculate how many UNIQUE labels exist in this specific sample
        sample_unique_labels = len(set(ner[2] for ner in ner_list))
        stats[lang]["unique_labels_per_sample"].append(sample_unique_labels)
        
        # Add to the global set of unique labels for this language
        for ner in ner_list:
            stats[lang]["unique_labels"].add(ner[2])
            
    return stats

def process_labeled_data(input_dir="./x_swl_output", train_json="train_pre_training_data.json", val_json="val_pre_training_data.json", stats_txt="pre_training_data_statistics.txt"):
    # Group samples by language
    samples_by_lang = {lang: [] for lang in LANG_TO_SCRIPT}
    
    # Global parsing issues
    global_issues = {lang: {
        "missing_annotations": 0, 
        "orphaned_annotations": 0, 
        "empty_ner_excluded": 0, 
        "discarded_for_balance": 0
    } for lang in LANG_TO_SCRIPT}
    
    file_patterns = os.path.join(input_dir, "*_labeled.jsonl")
    # sorted() is crucial here for cross-platform OS file-reading reproducibility
    filepaths = sorted(glob.glob(file_patterns))
    
    if not filepaths:
        print(f"No labeled JSONL files found in '{input_dir}'.")
        return

    # 1. Pre-calculate total lines for the tqdm progress bar
    total_lines = 0
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines += sum(1 for line in f if line.strip())

    # 2. Parse all files and group by language
    with tqdm(total=total_lines, desc="Processing samples", unit="sample") as pbar:
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            lang_code = filename.split("_")[0]
            script_code = LANG_TO_SCRIPT[lang_code] 
                
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    data = json.loads(line)
                    
                    try:
                        annotations = data["annotations"]
                        
                        if annotations is None:
                            global_issues[lang_code]["missing_annotations"] += 1
                            pbar.update(1)
                            continue

                        tokenized_text = [tok["text"] for tok in data["tokenized_text"]]
                        char_start_to_token_idx = {tok["start"]: tok["idx"] for tok in data["tokenized_text"]}
                        char_end_to_token_idx = {tok["end"]: tok["idx"] for tok in data["tokenized_text"]}
                        
                        ner_list = []
                        uid_to_tags = data["uid_to_tags"]
                        id_map = data["id_map"]
                        
                        for ann in annotations:
                            label = ann["hypernym"]
                            if not label:
                                continue
                                
                            ann_id = str(ann["ID"])
                            if ann_id not in uid_to_tags:
                                global_issues[lang_code]["orphaned_annotations"] += 1
                                continue
                            
                            tags = uid_to_tags[ann_id] 
                            formatted_label = label[:1].upper() + label[1:]
                            
                            for tag in tags:
                                if tag not in id_map:
                                    global_issues[lang_code]["orphaned_annotations"] += 1
                                    continue
                                    
                                entity_info = id_map[tag] 
                                char_start = entity_info["start"]
                                char_end = entity_info["end"]
                                
                                start_word_idx = char_start_to_token_idx[char_start]
                                end_word_idx = char_end_to_token_idx[char_end]
                                
                                ner_list.append([start_word_idx, end_word_idx, formatted_label])
                        
                        # NEW REQUIREMENT: Skip the entire sample if the resulting NER list is empty
                        if not ner_list:
                            global_issues[lang_code]["empty_ner_excluded"] += 1
                            pbar.update(1)
                            continue
                        
                        ner_list.sort(key=lambda x: x[0])
                        
                        processed_sample = {
                            "lang": lang_code,
                            "script": script_code,
                            "tokenized_text": tokenized_text,
                            "ner": ner_list
                        }
                        
                        samples_by_lang[lang_code].append(processed_sample)
                        
                    except KeyError as e:
                        print(f"\nKeyError encountered in file {filename} at line {line_num}: missing key {e}")
                        raise 
                    
                    pbar.update(1)

    # 3. Determine strict split sizes
    min_valid_samples = min(len(samples) for samples in samples_by_lang.values())
    
    if min_valid_samples < TRAIN_SIZE_PER_LANG:
        raise ValueError(f"Insufficient data. Minimum valid samples across all languages is {min_valid_samples}, which is less than the required train size of {TRAIN_SIZE_PER_LANG}.")
        
    val_size_per_lang = min_valid_samples - TRAIN_SIZE_PER_LANG
    print(f"\nBalancing Dataset: Train size = {TRAIN_SIZE_PER_LANG}/lang, Val size = {val_size_per_lang}/lang")

    train_data = []
    val_data = []

    # 4. Perform balanced splitting
    random.seed(RANDOM_SEED)
    
    for lang, samples in samples_by_lang.items():
        random.shuffle(samples)
        
        train_slice = samples[:TRAIN_SIZE_PER_LANG]
        val_slice = samples[TRAIN_SIZE_PER_LANG:TRAIN_SIZE_PER_LANG + val_size_per_lang]
        
        # Track how many we threw away to maintain exact balance
        discarded = len(samples) - (TRAIN_SIZE_PER_LANG + val_size_per_lang)
        global_issues[lang]["discarded_for_balance"] = discarded
        
        train_data.extend(train_slice)
        val_data.extend(val_slice)

    # Global shuffle of the final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # 5. Write JSON outputs
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully written {len(train_data)} train samples to '{train_json}'.")
    print(f"Successfully written {len(val_data)} val samples to '{val_json}'.")
    
    # 6. Calculate separate stats
    train_stats = calculate_split_stats(train_data)
    val_stats = calculate_split_stats(val_data)
    
    # 7. Write stratified statistics to TXT file
    with open(stats_txt, 'w', encoding='utf-8') as sf:
        sf.write("=" * 60 + "\n")
        sf.write("GLOBAL PARSING & EXCLUSION ISSUES\n")
        sf.write("=" * 60 + "\n")
        for lang, issues in sorted(global_issues.items()):
            sf.write(f"Language: [{lang.upper()}]\n")
            sf.write(f"  Samples excluded (null):      {issues['missing_annotations']:,}\n")
            sf.write(f"  Orphaned IDs skipped:         {issues['orphaned_annotations']:,}\n")
            sf.write(f"  Empty NER lists excluded:     {issues['empty_ner_excluded']:,}\n")
            sf.write(f"  Discarded for exact balance:  {issues['discarded_for_balance']:,}\n\n")

        def write_split_stats(split_name, split_stats):
            sf.write("=" * 60 + "\n")
            sf.write(f"{split_name.upper()} SPLIT STATISTICS\n")
            sf.write("=" * 60 + "\n")
            for lang, metrics in sorted(split_stats.items()):
                labels_arr = metrics["labels_per_sample"]
                uniq_labels_arr = metrics["unique_labels_per_sample"]
                words_arr = metrics["words_per_sample"]
                
                sf.write(f"Language: [{lang.upper()}]\n")
                sf.write(f"  Samples:                 {metrics['samples']:,}\n")
                sf.write(f"  Total valid labels:      {sum(labels_arr):,}\n")
                sf.write(f"  Unique valid labels:     {len(metrics['unique_labels']):,}\n")
                sf.write(f"  Labels per sample:       Avg: {statistics.mean(labels_arr):.2f} | Min: {min(labels_arr)} | Max: {max(labels_arr)}\n")
                sf.write(f"  Unique labels/sample:    Avg: {statistics.mean(uniq_labels_arr):.2f} | Min: {min(uniq_labels_arr)} | Max: {max(uniq_labels_arr)}\n")
                sf.write(f"  Words per sample:        Avg: {statistics.mean(words_arr):.1f} | Min: {min(words_arr)} | Max: {max(words_arr)}\n\n")

        write_split_stats("Train", train_stats)
        write_split_stats("Validation", val_stats)
            
    print(f"Stratified statistics successfully saved to '{stats_txt}'.")

if __name__ == "__main__":
    process_labeled_data()