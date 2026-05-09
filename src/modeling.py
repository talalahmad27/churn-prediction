"""Training and inference helpers for the churn classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

ID_COL = "customer_id"
TARGET = "churn"


def fill_complaint_type_na(df: pd.DataFrame) -> pd.DataFrame:
    """Purpose: Ensure `complaint_type` has no missing values so encoders and the model never see NaNs.

    Steps:
        1. Copy the dataframe so the caller's data is unchanged.
        2. If `complaint_type` exists, replace NaNs with the sentinel label 'Unknown'.
        3. Return the cleaned frame.
    """
    out = df.copy()
    if "complaint_type" in out.columns:
        out["complaint_type"] = out["complaint_type"].fillna("Unknown")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Purpose: List columns used as model inputs (exclude identifier and label).

    Steps:
        1. Iterate all column names in the frame.
        2. Omit `customer_id` (not predictive) and `churn` (target).
        3. Return the remaining names in table order.
    """
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def infer_numeric_categorical(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Purpose: Split feature names into numeric vs categorical columns for preprocessing.

    Steps:
        1. From `feature_cols`, select columns whose dtype is numeric → passthrough branch.
        2. Treat every other listed column as categorical → one-hot branch.
        3. Return `(numeric_cols, categorical_cols)`.
    """
    numeric = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Purpose: Build an sklearn `Pipeline` that preprocesses features then fits a churn classifier.

    Steps:
        1. Create a `ColumnTransformer`: numeric columns pass through unchanged; categoricals are one-hot encoded (unknown categories ignored at inference).
        2. Attach a `RandomForestClassifier` with settings suited to imbalanced churn data.
        3. Return the chained sklearn Pipeline (preprocessing step named 'prep', classifier named 'model').
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("model", clf)])


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Purpose: Pick a probability cutoff that maximizes F1 on the positive (churn) class.

    Steps:
        1. Generate a grid of candidate thresholds between 0.05 and 0.95.
        2. For each threshold, convert probabilities to 0/1 predictions and compute F1 for churn.
        3. Return the threshold with highest F1 (if tied, keep the first threshold that achieved it).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline, float, dict[str, Any]]:
    """Purpose: Train the churn pipeline on labeled data, tune a decision threshold, and report holdout metrics.

    Steps:
        1. Clean inputs (`complaint_type` NaNs) and build X/y from feature columns vs `churn`.
        2. Stratified split: hold out a final test set; from the remainder, split again into fit vs validation.
        3. Fit the pipeline on the fit subset, search validation probabilities for the best F1 threshold (churn).
        4. Refit the pipeline on the full training split (fit + validation rows) for deployment.
        5. Score the held-out test set using probabilities and the tuned threshold; collect ROC-AUC and classification report.
        6. Return the fitted pipeline, the threshold, and a metrics dictionary.
    """
    df = fill_complaint_type_na(df)
    feature_cols = feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET]
    numeric, categorical = infer_numeric_categorical(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state + 1, stratify=y_train
    )
    pipe = build_pipeline(numeric, categorical)
    pipe.fit(X_fit, y_fit)
    threshold = _best_f1_threshold(y_val.values, pipe.predict_proba(X_val)[:, 1])
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_tuned = (proba_test >= threshold).astype(int)
    metrics = {
        "classification_report": classification_report(y_test, y_pred_tuned, digits=3, zero_division=0),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "probability_threshold": threshold,
    }
    return pipe, threshold, metrics


def save_model_artifact(pipe: Pipeline, threshold: float, path: Path | None = None) -> Path:
    """Purpose: Persist the trained estimator together with the churn probability cutoff used at inference.

    Steps:
        1. Resolve output path (default `models/churn_pipeline.joblib`).
        2. Create parent directories if missing.
        3. Serialize a dict {'pipeline': pipe, 'threshold': threshold} via joblib.
        4. Return the path written.
    """
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold}, path)
    return path


def load_model_artifact(path: Path | None = None) -> tuple[Pipeline, float]:
    """Purpose: Load a saved pipeline and its decision threshold for the Streamlit app or batch scoring.

    Steps:
        1. Resolve path and verify the file exists (otherwise instruct to run the training script).
        2. Load the joblib blob.
        3. If it is a dict with key 'pipeline', return pipeline + threshold (default 0.5 if absent).
        4. If it is a bare fitted estimator, return it with threshold 0.5 for backward compatibility.
        5. Otherwise raise a clear error about an unknown file format.
    """
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python scripts/train_model.py"
        )
    blob = joblib.load(path)
    if isinstance(blob, dict) and "pipeline" in blob:
        return blob["pipeline"], float(blob.get("threshold", 0.5))
    if hasattr(blob, "predict"):
        return blob, 0.5
    raise ValueError(f"Unrecognized model file format at {path}")


def predict_churn(pipe: Pipeline, row: pd.DataFrame, threshold: float = 0.5) -> tuple[int, float]:
    """Purpose: Score a single customer row as stay vs churn using the saved threshold and report calibrated churn risk.

    Steps:
        1. Align preprocessing with training by filling `complaint_type` NaNs if present.
        2. Run `predict_proba` and take P(churn) as the positive class probability.
        3. Assign class 1 if P(churn) >= `threshold`, else 0.
        4. Return `(predicted_label, probability_of_churn)`.
    """
    row = fill_complaint_type_na(row)
    proba_churn = float(pipe.predict_proba(row)[0, 1])
    pred = int(proba_churn >= threshold)
    return pred, proba_churn


def permutation_importance_by_feature(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "f1",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Purpose: Estimate how much each original input column impacts model performance (feature impact).

    This uses permutation importance on the *raw input columns* (before one-hot encoding),
    which makes the results easy to interpret as “which columns matter most”.

    Steps:
        1. Permute each column in `X` independently `n_repeats` times (breaking its relationship to `y`).
        2. Re-score the pipeline using `scoring` (default: F1 for churn).
        3. Compute the drop in score vs the unpermuted baseline.
        4. Return a sorted DataFrame with mean and std importance per feature.
    """
    r = permutation_importance(
        pipe,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    return out


def aggregated_model_importance(pipe: Pipeline) -> pd.DataFrame:
    """Purpose: Provide model-based importances aggregated back to the original columns.

    Random forests expose `feature_importances_` for the *expanded* feature space (after one-hot encoding).
    This helper sums one-hot importances back to their source column so you can compare columns directly.

    Steps:
        1. Pull numeric + categorical column lists from the fitted `ColumnTransformer`.
        2. Get expanded feature names (`get_feature_names_out`) and the forest importances.
        3. Map each expanded name back to its original source column.
        4. Sum importances per source column and return them sorted descending.
    """
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not expose feature_importances_.")
    if not hasattr(prep, "get_feature_names_out"):
        raise ValueError("Preprocessor does not expose get_feature_names_out().")

    expanded_names = prep.get_feature_names_out()
    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.shape[0] != len(expanded_names):
        raise ValueError("Feature importance length does not match expanded feature names.")

    num_cols: list[str] = []
    cat_cols: list[str] = []
    for name, _transformer, cols in getattr(prep, "transformers_", []):
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

    def origin_from_expanded(expanded: str) -> str:
        cleaned = expanded
        if cleaned.startswith("num__"):
            cleaned = cleaned[len("num__") :]
        elif cleaned.startswith("cat__"):
            cleaned = cleaned[len("cat__") :]

        if cleaned in num_cols:
            return cleaned

        # OneHotEncoder names look like "<col>_<category...>".
        # We recover the original column by checking which known categorical column
        # is the longest prefix match.
        for col in sorted(cat_cols, key=len, reverse=True):
            if cleaned.startswith(col + "_") or cleaned == col:
                return col
        return cleaned

    agg: dict[str, float] = {}
    # Python 3.9 does not support zip(..., strict=...).
    for name, imp in zip(expanded_names, importances):
        src = origin_from_expanded(str(name))
        agg[src] = agg.get(src, 0.0) + float(imp)

    out = (
        pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    return out
