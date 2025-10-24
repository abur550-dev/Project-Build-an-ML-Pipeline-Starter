#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Core ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

# W&B (optional/defensive)
try:
    import wandb  # noqa: F401
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False


LOG = logging.getLogger("train_random_forest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ---------- helpers (no lambdas so the pipeline is picklable) -----------------
def fillna_text(series: pd.Series) -> pd.Series:
    """
    Replace NaN with empty string for text fields.
    This must be a top-level function (not a lambda) so the pipeline is picklable.
    """
    return series.fillna("")


def load_dataframe_from_source(source: str, wb_run=None) -> Path:
    """
    Resolve a train/val CSV either from a local path or a Weights & Biases artifact name.
    Returns a local file path.
    """
    p = Path(source)
    if p.exists() and p.is_file():
        LOG.info("Loading train/val CSV from local path: %s", p)
        return p

    if wb_run is None:
        raise FileNotFoundError(
            f"Could not find local file {source}. "
            "If you intended a W&B artifact name (e.g., 'trainval_data.csv:latest'), "
            "you must run online with wandb and pass a valid artifact."
        )

    # W&B fallback: use artifact name
    LOG.info("Fetching artifact %s via W&B...", source)
    art = wb_run.use_artifact(source)
    local_dir = Path(art.download())
    # first CSV we find in the artifact
    csvs = sorted(local_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found inside artifact {source} ({local_dir})")
    LOG.info("Loaded CSV from artifact: %s", csvs[0])
    return csvs[0]


def split_data(
    df: pd.DataFrame,
    target_col: str,
    stratify_by: Optional[str],
    val_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/val. If stratify_by is provided and present, use it for stratification.
    """
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    stratify = None
    if stratify_by and stratify_by in df.columns:
        stratify = df[stratify_by]
        LOG.info("Stratifying by column: %s", stratify_by)
    else:
        if stratify_by:
            LOG.warning("Requested stratify_by='%s' but column not found; continuing without.", stratify_by)

    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify if stratify is not None else None,
    )


def detect_columns(X: pd.DataFrame) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Heuristically select numeric, categorical, and a single text column ('name' if present).
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # common categoricals present in this dataset
    likely_cat = [c for c in ["neighbourhood_group", "room_type"] if c in X.columns]
    # include other 'object' dtypes as categoricals (except the text 'name')
    other_cat = [c for c in X.select_dtypes(include=["object"]).columns if c not in likely_cat and c != "name"]
    cat_cols = list(dict.fromkeys(likely_cat + other_cat))  # unique, preserve order
    text_col = "name" if "name" in X.columns else None

    LOG.info("Detected columns -> numeric: %s", num_cols)
    LOG.info("Detected columns -> categorical: %s", cat_cols)
    LOG.info("Detected columns -> text: %s", text_col)
    return num_cols, cat_cols, text_col


def build_preprocess(
    X: pd.DataFrame,
    max_tfidf_features: int = 200,
) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer that imputes and encodes features.
    - numeric: median impute
    - categorical: most_frequent impute + one-hot
    - text ("name"): fillna + TF-IDF (max_features)
    """
    num_cols, cat_cols, text_col = detect_columns(X)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))  # list -> 2D as expected
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if text_col is not None:
        text_pipe = Pipeline(
            steps=[
                ("fillna", FunctionTransformer(fillna_text, feature_names_out="one-to-one", validate=False)),
                ("tfidf", TfidfVectorizer(max_features=max_tfidf_features, stop_words="english")),
            ]
        )
        # Using the text column name directly is OK: ColumnTransformer will pass 1D iterable of strings
        transformers.append(("text", text_pipe, text_col))

    if not transformers:
        raise RuntimeError("No transformers configured; input feature space appears empty.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=None)
    return pre


def build_model(rf_config_path: str) -> RandomForestRegressor:
    with open(rf_config_path) as f:
        rf_params = json.load(f)
    # Provide sane defaults if missing
    rf_params = {
        "n_estimators": rf_params.get("n_estimators", 100),
        "max_depth": rf_params.get("max_depth"),
        "max_features": rf_params.get("max_features", "auto"),
        "random_state": rf_params.get("random_state"),
        "n_jobs": rf_params.get("n_jobs", -1),
    }
    LOG.info("RandomForest params: %s", rf_params)
    return RandomForestRegressor(**rf_params)


def build_pipeline(pre: ColumnTransformer, rf: RandomForestRegressor) -> Pipeline:
    """
    The final inference pipeline.
    """
    return Pipeline(steps=[("preprocess", pre), ("model", rf)])


# ------------------------------- main -----------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest on NYC Airbnb")
    parser.add_argument("--trainval_artifact", type=str, required=True,
                        help="Local CSV path or W&B artifact name (e.g., trainval_data.csv:latest)")
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=False, default="")
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="Name for the logged MLflow model artifact, e.g., random_forest_export")
    return parser.parse_args()


def main():
    args = parse_args()

    # W&B run (best-effort; OK if offline)
    wb_run = None
    if _HAVE_WANDB:
        try:
            wb_run = wandb.init(job_type="train_random_forest")
        except Exception as e:
            LOG.warning("W&B init failed/disabled: %s", e)

    # Resolve / load the CSV
    csv_path = load_dataframe_from_source(args.trainval_artifact, wb_run)
    df = pd.read_csv(csv_path)

    # Expect target column named 'price' (per project)
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in the train/val CSV.")

    X_train, X_val, y_train, y_val = split_data(
        df=df,
        target_col="price",
        stratify_by=args.stratify_by or None,
        val_size=args.val_size,
        random_seed=args.random_seed,
    )

    # Build preprocess + model
    pre = build_preprocess(X_train, max_tfidf_features=args.max_tfidf_features)
    rf = build_model(args.rf_config)
    sk_pipe = build_pipeline(pre, rf)

    # Train
    sk_pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = sk_pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
    LOG.info("Validation MAE=%.5f  R2=%.5f", mae, r2)

    # Log metrics (MLflow)
    mlflow.log_metrics({"mae": mae, "r2": r2})

    # Log params (RF)
    try:
        with open(args.rf_config) as f:
            rf_params_for_logging = json.load(f)
        for k, v in rf_params_for_logging.items():
            mlflow.log_param(f"rf__{k}", v)
    except Exception:
        pass

    # Log feature columns used (handy for inspection)
    feature_cols = pd.Series(X_train.columns)
    tmp_cols = Path("feature_columns.csv")
    feature_cols.to_csv(tmp_cols, index=False, header=False)
    mlflow.log_artifact(str(tmp_cols))

    # Log the full sklearn pipeline as an MLflow model
    input_example = X_train.head(1)
    mlflow.sklearn.log_model(
        sk_pipe,
        artifact_path=args.output_artifact,
        input_example=input_example,
    )

    # Also log to W&B if available
    if wb_run is not None:
        try:
            wandb.log({"mae": mae, "r2": r2})
        except Exception:
            pass
        finally:
            try:
                wb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
