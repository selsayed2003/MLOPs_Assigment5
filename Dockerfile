# Dockerfile
# ─────────────────────────────────────────────────────────────────────
# This Dockerfile packages the trained GAN model into a container image.
# It's a "mock" build — it simulates downloading the model without
# actually pulling from MLflow (since our server may not be public).
# ─────────────────────────────────────────────────────────────────────

# ── Base image ────────────────────────────────────────────────────────
# python:3.10-slim is a minimal Python image (~50 MB vs ~900 MB for full).
# "slim" omits compilers and test tools we don't need at runtime.
FROM python:3.10-slim

# ── Build argument ────────────────────────────────────────────────────
# ARG declares a build-time variable.
# Passed in via: docker build --build-arg RUN_ID=abc123 ...
# This lets the pipeline inject the exact MLflow Run ID into the image.
ARG RUN_ID

# ── Environment variable ──────────────────────────────────────────────
# ENV makes the ARG available INSIDE the running container (not just build time).
# Containers started from this image will always know which MLflow run
# they were built from — useful for auditing and debugging.
ENV MLFLOW_RUN_ID=${RUN_ID}

# ── Working directory ─────────────────────────────────────────────────
# All subsequent commands (RUN, COPY, CMD) happen relative to /app.
# Creates the folder if it doesn't exist.
WORKDIR /app

# ── Simulate model download ───────────────────────────────────────────
# In a real pipeline you would run:
#   RUN pip install mlflow && mlflow artifacts download -r ${MLFLOW_RUN_ID} ...
# Here we echo a message to a status file to mock the download step
# without needing network access to a private MLflow server.
RUN echo "Model downloaded for MLflow Run ID: ${MLFLOW_RUN_ID}" > model_status.txt

# ── Default command ───────────────────────────────────────────────────
# When someone runs "docker run my-gan-model:latest", this command executes.
# It prints the status file, confirming the image was built correctly
# with the right Run ID embedded.
CMD ["cat", "model_status.txt"]