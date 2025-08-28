import os
import subprocess
from drift import detect_drift

def run_pipeline():
    print("🚀 Checking for drift...")
    drift_detected = detect_drift()

    if drift_detected:
        print("🔄 Drift detected — retraining pipeline triggered...")

        # Run Spark preprocessing again (simulating new data arrival)
        subprocess.run(["python", "src/preprocess_spark1.py"])

        # Retrain model with MLflow logging
        subprocess.run(["python", "src/train3.py"])

        print("✅ Retraining complete. Model updated and tracked in MLflow.")
    else:
        print("👌 No retraining needed.")

if __name__ == "__main__":
    run_pipeline()
