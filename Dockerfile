# Use the specified base image
FROM python:3.10-slim

# Accept the MLflow Run ID as an argument during the build process
ARG RUN_ID
ENV MLFLOW_RUN_ID=${RUN_ID}

WORKDIR /app

# Mock the "model download" process
RUN echo "Downloading MLflow model for Run ID: ${MLFLOW_RUN_ID}..." > model_status.txt

# Default command to verify it worked
CMD ["cat", "model_status.txt"]