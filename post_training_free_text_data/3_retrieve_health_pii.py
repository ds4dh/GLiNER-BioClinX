import pandas as pd
from pathlib import Path

INPUT_FILE = "pii_500k_english_with_domain_probs.parquet"
OUTPUT_DIR = Path("./data/ai4privacy-pii/")
OUTPUT_FILE = OUTPUT_DIR / "pii_health.parquet"
PROB_THRESHOLD = 0.90

def main():
    print(f"Loading '{INPUT_FILE}'...")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Filtering for 'Health' domain with probability > {PROB_THRESHOLD}...")
    mask = (df["predicted_domain"] == "Health") & (df["prob_Health"] > PROB_THRESHOLD)
    df_filtered = df[mask].reset_index(drop=True)

    print(f"Ensuring directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(df_filtered)} rows. Saving to '{OUTPUT_FILE}'...")
    df_filtered.to_parquet(OUTPUT_FILE, index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()