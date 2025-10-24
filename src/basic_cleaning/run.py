#!/usr/bin/env python
"""
Basic cleaning: read input (local CSV path OR W&B artifact), drop outliers & clip by lat/long,
save cleaned CSV, and log it as an MLflow artifact.
"""
import argparse
import os
import logging
from pathlib import Path

import pandas as pd
import mlflow

def _maybe_import_wandb():
    try:
        import wandb
        return wandb
    except Exception:
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger("basic_cleaning")

def read_input(input_artifact: str) -> pd.DataFrame:
    if (input_artifact.endswith(".csv") or "/" in input_artifact) and os.path.exists(input_artifact):
        logger.info(f"Reading local CSV: {input_artifact}")
        return pd.read_csv(input_artifact)

    if ":" in input_artifact:
        wandb = _maybe_import_wandb()
        if wandb is None:
            raise RuntimeError(
                f"Input '{input_artifact}' looks like a W&B artifact, "
                "but wandb is not available. Install wandb or pass a local CSV path."
            )
        logger.info(f"Fetching W&B artifact: {input_artifact}")
        run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning", group="cleaning", save_code=True)
        path = run.use_artifact(input_artifact).file()
        df = pd.read_csv(path)
        wandb.finish()
        return df

    raise FileNotFoundError(
        f"Could not interpret input_artifact='{input_artifact}'. "
        "Pass an existing local CSV path or a W&B artifact like 'raw_data:latest'."
    )

def go(args):
    df = read_input(args.input_artifact)

    # Price filter
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Parse last_review
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Geo filter (NYC bbox)
    if {"longitude", "latitude"}.issubset(df.columns):
        idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
        df = df[idx].copy()

    # Save cleaned file
    out_path = Path(args.output_artifact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned CSV to: {out_path.resolve()}")

    # Log as MLflow artifact (Projects already started a run)
    mlflow.log_artifact(str(out_path), artifact_path="basic_cleaning")
    logger.info("✅ Logged MLflow artifact: basic_cleaning/%s", out_path.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")
    parser.add_argument("--input_artifact", type=str, required=True,
                        help="Local CSV path OR a W&B artifact like 'raw_data:latest'")
    parser.add_argument("--output_artifact", type=str, required=True,
                        help="Filename for the cleaned CSV to create (e.g., 'outputs/clean/clean_sample.csv')")
    parser.add_argument("--output_type", type=str, required=True,
                        help="(Ignored for MLflow) kept for compatibility")
    parser.add_argument("--output_description", type=str, required=True,
                        help="(Ignored for MLflow) kept for compatibility")
    parser.add_argument("--min_price", type=float, required=True,
                        help="Minimum price threshold to keep; rows below are dropped")
    parser.add_argument("--max_price", type=float, required=True,
                        help="Maximum price threshold to keep; rows above are dropped")
    args = parser.parse_args()
    go(args)
