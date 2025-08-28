import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_path="data/processed/X_train.csv",
                 new_data_path="data/new/X_new.csv",
                 threshold: float = 0.05):
    """
    Simple drift detection using KS-test
    """
    ref = pd.read_csv(reference_path)
    new = pd.read_csv(new_data_path)

    drifted_features = []

    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            stat, pval = ks_2samp(ref[col].dropna(), new[col].dropna())
            if pval < threshold:
                drifted_features.append(col)

    if drifted_features:
        print(f"⚠️ Drift detected in {len(drifted_features)} features: {drifted_features}")
        return True
    else:
        print("✅ No significant drift.")
        return False
