import pandas as pd
import os
# Ensure directory exists
os.makedirs("data/new", exist_ok=True)
df = pd.read_csv("data/raw/Titanic-Dataset.csv")
new_batch = df.sample(100, random_state=42)  # simulate new incoming data
new_batch.to_csv("data/new/X_new.csv", index=False)

# Run pipeline
# !python src/retrain_pipeline.py
