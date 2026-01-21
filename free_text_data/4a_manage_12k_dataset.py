import pandas as pd
import glob
import os
import argparse

def get_csv_language(filepath):
    """Reads the first row to check the language code."""
    try:
        df = pd.read_csv(filepath, nrows=1)
        if 'language' in df.columns and not df.empty:
            return df['language'].iloc[0]
    except Exception:
        pass
    return None

def manage_dataset(dataset_path, source_dir, replace_lang=None, seed=42):
    # 1. Load Existing Dataset
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset: {dataset_path}")
        df_main = pd.read_csv(dataset_path)
        existing_languages = set(df_main['language'].unique())
        print(f" -> Found {len(df_main)} rows covering: {sorted(list(existing_languages))}")
    else:
        print("Dataset not found. Creating a new one.")
        df_main = pd.DataFrame(columns=['language', 'text'])
        existing_languages = set()

    # 2. Handle Replacement (Remove old data)
    if replace_lang:
        if replace_lang in existing_languages:
            print(f"\n[REPLACE] Removing existing rows for '{replace_lang}'...")
            df_main = df_main[df_main['language'] != replace_lang]
            existing_languages.discard(replace_lang) # Mark as "missing" so it gets re-added below
        else:
            print(f"\n[WARNING] Requested to replace '{replace_lang}', but it was not in the dataset.")

    # 3. Scan Source Folder and Add Missing/Replaced Languages
    file_pattern = os.path.join(source_dir, '*_chunked_cleaned_deduplicated.csv')
    source_files = glob.glob(file_pattern)
    
    if not source_files:
        print(f"Error: No source files found in {source_dir}")
        return

    changes_made = False

    for f_path in source_files:
        # Determine language of the source file
        lang_code = get_csv_language(f_path)
        if not lang_code:
            continue

        # Logic: Only process if we don't have this language yet
        if lang_code not in existing_languages:
            print(f"[ADD] Processing '{lang_code}' from {os.path.basename(f_path)}...")
            
            # Read full source file
            df_source = pd.read_csv(f_path)
            
            # Sample 12k
            if len(df_source) >= 12000:
                df_sample = df_source.sample(n=12000, random_state=seed)
            else:
                print(f"   ⚠️ Warning: Only found {len(df_source)} rows (requested 12,000). Taking all.")
                df_sample = df_source.copy()
            
            # Add to main dataframe
            df_main = pd.concat([df_main, df_sample], ignore_index=True)
            existing_languages.add(lang_code)
            changes_made = True
        else:
            # Language already exists and wasn't marked for replacement
            pass

    # 4. Shuffle and Save (Only if needed)
    if changes_made or replace_lang:
        print("\nShuffling all rows...")
        df_final = df_main.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        df_final.to_csv(dataset_path, index=False)
        print(f"✔ Success! Saved {len(df_final)} rows to {dataset_path}")
        print(f"Final Languages: {sorted(list(df_final['language'].unique()))}")
    else:
        print("\nNo changes needed. Dataset is already up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add or replace languages in the 12k dataset.")
    parser.add_argument('--dataset', type=str, default='./data/final/balanced_12k.csv', help="Path to the 12k CSV.")
    parser.add_argument('--source_dir', type=str, default='./data/deduplicated', help="Folder with source CSVs.")
    parser.add_argument('--replace', type=str, default=None, help="Language code to replace (e.g., 'nl'). If not set, only adds new languages.")
    
    args = parser.parse_args()
    
    manage_dataset(args.dataset, args.source_dir, args.replace)