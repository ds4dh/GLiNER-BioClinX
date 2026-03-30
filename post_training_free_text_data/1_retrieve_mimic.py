import pandas as pd
import json
output_path = "./data/MIMIC/mimic_en.json"
df = pd.read_csv('./data/MIMIC/NOTEEVENTS.csv.gz', usecols=['TEXT'])
unique_texts = df['TEXT'].astype(str).unique()
sampled_notes = pd.Series(unique_texts).sample(n=400, random_state=42).tolist()
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sampled_notes, f, ensure_ascii=False, indent=2)