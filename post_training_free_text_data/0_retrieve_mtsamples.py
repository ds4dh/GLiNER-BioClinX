import pandas as pd
import os
import json
output_path = "./data/MTSamples/mtsamples_en.json"
path = "data/MTSamples/"
df_dev = pd.read_csv(os.path.join(path, "medical_transcripts_dev.csv"))
df_test = pd.read_csv(os.path.join(path, "medical_transcripts_test.csv"))
df_combined = pd.concat([df_dev, df_test], axis=0, ignore_index=True)
unique_mtsamples = df_combined.eng_report.unique().tolist()
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(unique_mtsamples, f, ensure_ascii=False, indent=2)