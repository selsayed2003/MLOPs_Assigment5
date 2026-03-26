import sys
import mlflow
import os

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

# 1. Read the Run ID saved by train.py
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking MLflow Run ID: {run_id}")
except FileNotFoundError:
    print("Error: model_info.txt not found. Did the validate job upload the artifact?")
    sys.exit(1)

# 2. Fetch the run data from MLflow
# 2. Fetch the run data from MLflow
try:
    run = mlflow.get_run(run_id)
    
    # Get the raw value from DagsHub
    raw_acc = run.data.metrics.get("accuracy_Real", 0.0)
    
    # Smart check: if the number is > 1.0, it's a percentage (like 85.0)
    # If it's <= 1.0, it's already a decimal (like 0.85)
    if raw_acc > 1.0:
        accuracy = raw_acc / 100.0
    else:
        accuracy = raw_acc
    
    print(f"Validation Accuracy: {accuracy:.4f}")

    # 3. Check the threshold
    if accuracy < 0.85:
        print(f"❌ Pipeline Failed: Accuracy {accuracy:.2f} is below the 0.85 threshold.")
        sys.exit(1)
    else:
        print(f"✅ Pipeline Passed: Accuracy {accuracy:.2f} meets the threshold.")
        sys.exit(0)

except Exception as e:
    print(f"Error connecting to MLflow: {e}")
    sys.exit(1)