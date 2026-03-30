import json
import glob
import random
import os

# Set the seed for reproducibility
random.seed(42)

LANG_TO_SCRIPT = {
    "de": "Latn", "en": "Latn", "es": "Latn", "fr": "Latn",
    "it": "Latn", "ja": "Jpan", "nl": "Latn", "pt": "Latn",
    "ru": "Cyrl", "uk": "Cyrl", "zh": "Hans"  
}

def process_mtsamples():
    input_pattern = "outputs-mtsamples/mtsamples_*_ner.jsonl"
    output_file = "./mtsamples.json"
    
    all_processed_data = []

    # Iterate through all jsonl files matching the pattern
    for filepath in glob.glob(input_pattern):
        # Extract the language code from the filename (e.g., 'en' from 'mtsamples_en_ner.jsonl')
        filename = os.path.basename(filepath)
        lang_code = filename.split('_')[1] 
        
        # Fallback if an unexpected file is caught
        if lang_code not in LANG_TO_SCRIPT:
            print(f"Warning: Language code '{lang_code}' not found in LANG_TO_SCRIPT mapping. Skipping file.")
            continue
            
        script = LANG_TO_SCRIPT[lang_code]
        
        # Read the jsonl file
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                
                # Extract the required fields from the nested 'gliner_format'
                gliner_data = data.get("gliner_format", {})
                tokenized_text = gliner_data.get("tokenized_text", [])
                ner = gliner_data.get("ner", [])
                
                # Append the newly formatted dictionary
                all_processed_data.append({
                    "lang": lang_code,
                    "script": script,
                    "tokenized_text": tokenized_text,
                    "ner": ner
                })

    # Shuffle the list of dictionaries
    random.shuffle(all_processed_data)

    # Write the output with the requested JSON formatting
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(all_processed_data, out_f, ensure_ascii=False, indent=2)

    print(f"Successfully processed and merged records.")
    print(f"Total entries: {len(all_processed_data)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    process_mtsamples()