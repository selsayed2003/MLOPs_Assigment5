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
try:
    run = mlflow.get_run(run_id)
    
    # Fetch the 'accuracy_Real' metric from the last epoch.
    # Note: Your train.py calculates accuracy as a percentage (e.g., 85.0 instead of 0.85).
    # We divide by 100 to get the 0.0 to 1.0 format required by your assignment.
    acc_real_percentage = run.data.metrics.get("accuracy_Real", 0.0)
    accuracy = acc_real_percentage / 100.0
    
    print(f"Validation Accuracy: {accuracy:.2f}")

    # 3. Check the threshold
    if accuracy < 0.85:
        print("❌ Pipeline Failed: Accuracy is below the 0.85 threshold.")
        sys.exit(1) # This specific exit code tells GitHub Actions to fail the job and stop
    else:
        print("✅ Pipeline Passed: Accuracy meets the threshold.")
        sys.exit(0) # This tells GitHub Actions to proceed to the Docker build

except Exception as e:
    print(f"Error connecting to MLflow: {e}")
    sys.exit(1)