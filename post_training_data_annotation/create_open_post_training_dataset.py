import json
import random

# Set the seed for exact reproducibility
random.seed(42)

def create_open_post_training_dataset():
    # Define the input files and the final output file
    input_files = ["medmentions.json", "mtsamples.json", "pii_health.json"]
    output_file = "open_post_training_data.json"
    
    combined_data = []

    # 1. Load and concatenate
    for filename in input_files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Verify it's a list before extending to prevent errors
                if isinstance(data, list):
                    combined_data.extend(data)
                    print(f"Successfully loaded {len(data)} records from '{filename}'.")
                else:
                    print(f"Warning: '{filename}' does not contain a top-level JSON list. Skipping.")
                    
        except FileNotFoundError:
            print(f"Error: Could not find '{filename}'. Please ensure it exists in the directory.")
        except json.JSONDecodeError:
            print(f"Error: '{filename}' contains invalid JSON formatting.")

    # 2. Seeded Shuffle
    # Because we set random.seed(42) at the top, this will shuffle identically on every run
    print("\nShuffling the combined dataset...")
    random.shuffle(combined_data)

    # 3. Save with specific formatting
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(combined_data, out_f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\nProcess complete!")
    print(f"Total combined entries: {len(combined_data)}")
    print(f"Final dataset saved to: '{output_file}'")

if __name__ == "__main__":
    create_open_post_training_dataset()
