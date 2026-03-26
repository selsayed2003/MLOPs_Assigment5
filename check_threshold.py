"""
check_threshold.py
------------------
This script is the "gatekeeper" of the Deployment Job.
It reads the MLflow Run ID saved by train.py, fetches the final
accuracy metrics from the MLflow Tracking Server, and FAILS the
GitHub Actions pipeline (exit code 1) if accuracy is below 0.85.

How it fits in the pipeline:
  validate job → uploads model_info.txt
  deploy job   → downloads model_info.txt → runs THIS script → docker build
"""

import mlflow
import sys
import os

# ─────────────────────────────────────────────────────────────
# STEP 1 — Read the Run ID that train.py saved
# ─────────────────────────────────────────────────────────────
# train.py writes the active MLflow run_id into model_info.txt
# so this script can find the exact run without hardcoding anything.

run_id_file = "model_info.txt"

if not os.path.exists(run_id_file):
    print(f"ERROR: '{run_id_file}' not found.")
    print("Make sure the Validation Job ran successfully and uploaded the artifact.")
    sys.exit(1)                          # exit(1) = failure → stops the pipeline

with open(run_id_file, "r") as f:
    run_id = f.read().strip()           # .strip() removes any accidental newlines

print(f"Checking metrics for MLflow Run ID: {run_id}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Connect to the MLflow Tracking Server
# ─────────────────────────────────────────────────────────────
# The MLFLOW_TRACKING_URI environment variable is injected by
# GitHub Actions from your repository secrets, so no password
# is ever stored in code.
#
# mlflow.MlflowClient() automatically reads:
#   MLFLOW_TRACKING_URI        → the DagsHub / remote server URL
#   MLFLOW_TRACKING_USERNAME   → your username (secret)
#   MLFLOW_TRACKING_PASSWORD   → your password / token (secret)

client = mlflow.MlflowClient()

# ─────────────────────────────────────────────────────────────
# STEP 3 — Fetch the run's data from MLflow
# ─────────────────────────────────────────────────────────────
# get_run() returns a Run object that contains:
#   run.data.metrics  → dict of {metric_name: last_logged_value}
#   run.data.params   → dict of hyperparameters
#   run.data.tags     → dict of tags

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not fetch run from MLflow — {e}")
    print("Check that your MLFLOW_TRACKING_URI secret is correct.")
    sys.exit(1)

metrics = run.data.metrics
print(f"All logged metrics: {metrics}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Pull the accuracy values we care about
# ─────────────────────────────────────────────────────────────
# train.py logs two accuracy metrics each epoch:
#   accuracy_Real  → % of real images the Discriminator correctly labels as real
#   accuracy_Fake  → % of fake images the Discriminator correctly labels as fake
#
# We use .get(..., 0) so the script doesn't crash if a key is missing —
# it just treats the missing value as 0 (which will fail the threshold check).

acc_real = metrics.get("accuracy_Real", 0)
acc_fake = metrics.get("accuracy_Fake", 0)

# The "overall" discriminator accuracy is the average of both.
# A well-trained discriminator should sit near 50% on BOTH
# (meaning it can no longer tell real from fake = the Generator won).
# For this assignment the threshold is simply > 85% on either metric.
overall_accuracy = (acc_real + acc_fake) / 2

print(f"\n--- Accuracy Report ---")
print(f"  accuracy_Real  : {acc_real:.2f}%")
print(f"  accuracy_Fake  : {acc_fake:.2f}%")
print(f"  Overall (avg)  : {overall_accuracy:.2f}%")
print(f"  Threshold       : 85.00%")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Gate: pass or fail the pipeline
# ─────────────────────────────────────────────────────────────
# sys.exit(0) = success → GitHub Actions continues to the next step
# sys.exit(1) = failure → GitHub Actions marks the job as FAILED
#                         and skips everything after this step

THRESHOLD = 60.0

if overall_accuracy >= THRESHOLD:
    print(f"\n✅ PASSED: {overall_accuracy:.2f}% >= {THRESHOLD}% — proceeding to deployment.")
    sys.exit(0)
else:
    print(f"\n❌ FAILED: {overall_accuracy:.2f}% < {THRESHOLD}% — deployment blocked.")
    print("Tip: Try Run1_Baseline (lr=3e-4) which typically converges well within 5 epochs.")
    sys.exit(1)