#!/usr/bin/env sh
# One-shot start-up job:
#   1. seed the database from the bundled synthetic CSVs (idempotent)
#   2. (optionally) re-train the model, logging runs + registered model to MLflow
#   3. produce a monitoring/drift report
# Training failures never block the stack: the bundled, pre-trained model is kept.
set -u
echo "[bootstrap] seeding database"
python -m src.data.repository seed || { echo "[bootstrap] database seeding failed"; exit 1; }

if [ "${TRAIN_ON_STARTUP:-true}" = "true" ]; then
  echo "[bootstrap] training (MLflow: ${MLFLOW_TRACKING_URI:-local})"
  if python -m training.train; then
    echo "[bootstrap] training finished"
  else
    echo "[bootstrap] training failed – keeping the bundled pre-trained model"
  fi
else
  echo "[bootstrap] TRAIN_ON_STARTUP=false – using the bundled pre-trained model"
fi

python -m src.monitoring.drift || echo "[bootstrap] drift report failed (non-fatal)"
echo "[bootstrap] done"
