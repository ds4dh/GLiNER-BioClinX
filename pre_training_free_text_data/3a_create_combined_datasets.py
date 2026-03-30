import pandas as pd
import glob
import os

def create_combined_datasets(seed=42):
    """
    Finds all deduplicated CSVs, combines them, and saves two versions:
    1. A simple concatenation of all files, shuffled.
    2. A balanced concatenation where each language is sampled to match
       the size of the smallest dataset.
    """
    # Define file paths
    source_directory = './data/deduplicated/'
    output_directory = './data/final/'
    file_pattern = os.path.join(source_directory, '*_chunked_cleaned_deduplicated.csv')

    # Ensure the final output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Find all the deduplicated files
    deduplicated_files = glob.glob(file_pattern)
    if not deduplicated_files:
        print(f"Error: No files found matching the pattern '{file_pattern}'.")
        print("Please make sure your deduplicated files are in the correct directory.")
        return

    print(f"Found {len(deduplicated_files)} files to process:")
    for f in deduplicated_files:
        print(f" - {os.path.basename(f)}")
    print("-" * 30)

    # --- Task 1: Create a simple, shuffled rowbind of all files ---
    print("\n[Task 1] Creating a simple shuffled combination of all data...")
    
    # Read and collect all dataframes
    list_of_dfs = [pd.read_csv(f) for f in deduplicated_files]
    
    # Concatenate into a single dataframe
    full_df = pd.concat(list_of_dfs, ignore_index=True)
    
    # Shuffle the combined dataframe for randomness
    shuffled_full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the result
    full_output_path = os.path.join(output_directory, 'all.csv')
    shuffled_full_df.to_csv(full_output_path, index=False)
    
    print(f"✔ Successfully saved the fully combined dataset with {len(shuffled_full_df)} rows to:")
    print(f"  {full_output_path}")
    print("-" * 30)

    # --- Task 2: Create a balanced, sampled rowbind ---
    print("\n[Task 2] Creating a balanced combination of all data...")

    # First, determine the size of the smallest dataset
    counts = {os.path.basename(f): len(pd.read_csv(f)) for f in deduplicated_files}
    min_count = min(counts.values())
    
    print(f"The smallest dataset has {min_count} rows. Sampling this amount from each language.")

    # Sample from each dataframe and collect them
    list_of_sampled_dfs = []
    for f in deduplicated_files:
        df = pd.read_csv(f)
        sampled_df = df.sample(n=min_count, random_state=seed)
        list_of_sampled_dfs.append(sampled_df)
        
    # Concatenate the sampled dataframes
    balanced_df = pd.concat(list_of_sampled_dfs, ignore_index=True)
    
    # Shuffle the final balanced dataframe to mix the languages
    shuffled_balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the result
    balanced_output_path = os.path.join(output_directory, 'balanced.csv')
    shuffled_balanced_df.to_csv(balanced_output_path, index=False)

    print(f"✔ Successfully saved the balanced dataset with {len(shuffled_balanced_df)} rows to:")
    print(f"  {balanced_output_path}")
    print("-" * 30)

    # --- Task 3: Create a balanced sample of 12,000 rows per language from in‑memory balanced_df ---
    print("\n[Task 3] Creating a 12k-per-language sample from the balanced dataset...")

    sample_size = 12000

    # Ensure there's a 'language' column to group by
    if 'language' not in shuffled_balanced_df.columns:
        raise KeyError("Expected a 'language' column in the balanced DataFrame to group by language.")

    # Check that each language group has at least sample_size rows
    counts_bal = shuffled_balanced_df['language'].value_counts()
    too_small = counts_bal[counts_bal < sample_size]
    if not too_small.empty:
        for lang, cnt in too_small.items():
            print(f"Error: Language '{lang}' has only {cnt} rows in the balanced dataset, fewer than {sample_size}.")
        raise ValueError("Cannot sample 12,000 rows per language because some groups are too small.")

    # Perform the sampling directly on the in-memory balanced DataFrame
    sampled_12k_df = (
        shuffled_balanced_df
        .groupby('language', group_keys=False)
        .apply(lambda grp: grp.sample(n=sample_size, random_state=seed)) # Corrected: argument removed
    )

    # Shuffle the combined 12k-per-language sample
    shuffled_12k = sampled_12k_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the 12k-sampled balanced dataset
    balanced_12k_output_path = os.path.join(output_directory, 'balanced_12k.csv')
    shuffled_12k.to_csv(balanced_12k_output_path, index=False)

    print(f"✔ Successfully saved the 12k balanced dataset with {len(shuffled_12k)} rows to:")
    print(f"  {balanced_12k_output_path}")
    print("-" * 30)


if __name__ == "__main__":
    create_combined_datasets(seed=42)