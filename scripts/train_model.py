"""Train churn model and write models/churn_pipeline.joblib."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.modeling import DEFAULT_DATA_PATH, MODEL_PATH, save_model_artifact, train_and_evaluate


def main() -> None:
    """Purpose: CLI entrypoint to train the churn model from disk and save the artifact used by the app.

    Steps:
        1. Load labeled rows from `DEFAULT_DATA_PATH` (customer_churn.csv).
        2. Call `train_and_evaluate` to fit the pipeline, tune the probability threshold, and compute metrics.
        3. Save pipeline + threshold to `MODEL_PATH` via `save_model_artifact`.
        4. Print the output path, threshold, ROC-AUC, and classification report for a quick sanity check.
    """
    df = pd.read_csv(DEFAULT_DATA_PATH)
    pipe, threshold, metrics = train_and_evaluate(df)
    out = save_model_artifact(pipe, threshold)
    print(f"Saved model artifact to {out}")
    print(f"Probability threshold (tuned on validation): {threshold:.4f}")
    print(f"ROC-AUC (holdout): {metrics['roc_auc']:.4f}")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
