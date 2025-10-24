#!/usr/bin/env python3
import os
import sys
import logging
import mlflow

# Hydra / config
from omegaconf import DictConfig, OmegaConf
import hydra

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -------- Paths --------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
COMPONENTS_DIR = os.path.join(REPO_ROOT, "components")

# Small helper for safe config access
def _get(cfg: DictConfig, dotted: str, default=None):
    """
    Read a dotted key from an OmegaConf DictConfig with a default.
    """
    try:
        val = OmegaConf.select(cfg, dotted)
        return default if val is None else val
    except Exception:
        return default


@hydra.main(config_path="src", config_name="config", version_base=None)
def go(cfg: DictConfig):
    """
    Orchestrates the ML pipeline via MLflow Projects.
    You can select steps with:  -P steps="step1,step2"
    Use "all" to run the whole pipeline.
    """
    # Ensure mlflow runs from repo root (so relative paths work)
    os.chdir(REPO_ROOT)

    log.info("Active pipeline steps selection: %s", _get(cfg, "main.steps", "all"))

    # Define the known steps in order
    default_steps = [
        "download",
        "basic_cleaning",
        "data_check",
        "data_split",
        "train_random_forest",
        "register_model",
        "test_regression_model",  # not in default "all" run unless specified
    ]

    steps_param = _get(cfg, "main.steps", "all")
    if steps_param == "all":
        active_steps = default_steps[:-1]  # exclude test step by default
    else:
        # Allow comma or space separated
        parts = [s.strip() for s in steps_param.replace(" ", ",").split(",") if s.strip()]
        active_steps = parts

    # --- 1) DOWNLOAD ---
    if "download" in active_steps:
        log.info("Running step: download")
        _ = mlflow.run(
            os.path.join(SRC_DIR, "download_file"),
            entry_point="main",
            parameters={
                "file_url": _get(cfg, "etl.file_url"),
                "artifact_name": _get(cfg, "etl.artifact_name"),
                "artifact_type": _get(cfg, "etl.artifact_type"),
                "artifact_description": _get(cfg, "etl.artifact_description"),
            },
            env_manager="local",
        )

    # --- 2) BASIC CLEANING ---
    if "basic_cleaning" in active_steps:
        log.info("Running step: basic_cleaning")
        _ = mlflow.run(
            os.path.join(SRC_DIR, "basic_cleaning"),
            entry_point="main",
            parameters={
                "input_artifact": _get(cfg, "cleaning.input_artifact", "sample.csv:latest"),
                "output_artifact": _get(cfg, "cleaning.output_artifact", "clean_sample.csv"),
                "output_type": _get(cfg, "cleaning.output_type", "clean_sample"),
                "output_description": _get(cfg, "cleaning.output_description", "Cleaned data"),
                "min_price": str(_get(cfg, "cleaning.min_price")),
                "max_price": str(_get(cfg, "cleaning.max_price")),
            },
            env_manager="local",
        )

    # --- 3) DATA CHECK ---
    if "data_check" in active_steps:
        log.info("Running step: data_check")
        _ = mlflow.run(
            os.path.join(SRC_DIR, "data_check"),
            entry_point="main",
            parameters={
                "csv": _get(cfg, "data_check.csv", "clean_sample.csv:latest"),
                "ref": _get(cfg, "data_check.ref", "clean_sample.csv:latest"),
                "kl_threshold": str(_get(cfg, "data_check.kl_threshold", 0.2)),
            },
            env_manager="local",
        )

    # --- 4) DATA SPLIT ---
    if "data_split" in active_steps:
        log.info("Running step: data_split")
        _ = mlflow.run(
            os.path.join(SRC_DIR, "data_split"),
            entry_point="main",
            parameters={
                "input_artifact": _get(cfg, "data_split.input_artifact", "clean_sample.csv:latest"),
                "test_size": str(_get(cfg, "data_split.test_size", 0.2)),
                "random_seed": str(_get(cfg, "data_split.random_seed", 42)),
                "stratify_by": _get(cfg, "data_split.stratify_by", "neighbourhood_group"),
            },
            env_manager="local",
        )

    # --- 5) TRAIN RANDOM FOREST ---
    if "train_random_forest" in active_steps:
        log.info("Running step: train_random_forest")

        # Path to rf_config.json (already provided in the repo)
        rf_config = os.path.join(REPO_ROOT, "rf_config.json")

        _ = mlflow.run(
            os.path.join(SRC_DIR, "train_random_forest"),
            entry_point="main",
            parameters={
                # per instructions
                "trainval_artifact": _get(cfg, "modeling.trainval_artifact", "trainval_data.csv:latest"),
                "val_size": str(_get(cfg, "modeling.val_size", 0.2)),
                "random_seed": str(_get(cfg, "modeling.random_seed", 42)),
                "stratify_by": _get(cfg, "modeling.stratify_by", "neighbourhood_group"),
                "rf_config": rf_config,
                "max_tfidf_features": str(_get(cfg, "modeling.max_tfidf_features", 5)),
                "output_artifact": _get(cfg, "modeling.output_artifact", "random_forest_export"),
            },
            env_manager="local",
        )

    # --- 6) REGISTER MODEL (optional utility step) ---
    # If your training step already registers via MLflow, you can skip this.
    if "register_model" in active_steps:
        log.info("Running step: register_model")
        _ = mlflow.run(
            os.path.join(SRC_DIR, "register_model"),
            entry_point="main",
            parameters={
                "model_export": _get(cfg, "register.model_export", "random_forest_export"),
                "model_name": _get(cfg, "register.model_name", "nyc_airbnb_price_model"),
                # Optionally promote/alias inside that component if supported
            },
            env_manager="local",
        )

    # --- 7) TEST REGRESSION MODEL (run manually after promotion to prod) ---
    if "test_regression_model" in active_steps:
        log.info("Running step: test_regression_model")
        _ = mlflow.run(
            os.path.join(COMPONENTS_DIR, "test_regression_model"),  # IMPORTANT: components, not src
            entry_point="main",
            parameters={
                "mlflow_model": _get(cfg, "testing.mlflow_model", "random_forest_export:prod"),
                "test_artifact": _get(cfg, "testing.test_artifact", "test_data.csv:latest"),
            },
            env_manager="local",
        )

    log.info("Pipeline execution completed.")


if __name__ == "__main__":
    try:
        go()
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)
