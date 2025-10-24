#!/usr/bin/env python3
import os
import json
import logging
from typing import List

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)


def _get(cfg: DictConfig, path: str, default=None):
    """
    Safe nested getter for OmegaConf DictConfig.
    Example: _get(cfg, "modeling.random_forest.max_depth", 10)
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, DictConfig) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_steps(steps_str: str) -> List[str]:
    """
    Turn comma-separated steps string into a list.
    Accepts 'all' or blank to run everything in order.
    """
    if not steps_str or steps_str.strip() in ("", "all"):
        return [
            "download",
            "basic_cleaning",
            "data_check",
            "data_split",
            "train_random_forest",
            "test_regression_model",
        ]
    return [s.strip() for s in steps_str.split(",") if s.strip()]


@hydra.main(config_path="src", config_name="config", version_base=None)
def go(cfg: DictConfig) -> None:
    """
    Orchestrates the pipeline via MLflow Projects.
    Expects src/config.yaml (Hydra) to exist.
    """
    # Show the active config for debugging
    log.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    # Determine which steps to run
    active_steps = _parse_steps(_get(cfg, "main.steps", "all"))
    log.info("Active pipeline steps: %s", active_steps)

    # Common paths
    repo_root = get_original_cwd()
    components_dir = os.path.join(repo_root, "components")

    # -----------------------------
    # 1) DOWNLOAD
    # -----------------------------
    if "download" in active_steps:
        log.info("Running step: download")
        _ = mlflow.run(
            os.path.join(components_dir, "get_data"),
            "main",
            parameters={
                "file_url": _get(cfg, "data.file_url"),
                "artifact_name": _get(cfg, "data.raw_artifact", "raw_data.csv"),
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded",
            },
            env_manager="local",
        )

    # -----------------------------
    # 2) BASIC CLEANING
    # -----------------------------
    if "basic_cleaning" in active_steps:
        log.info("Running step: basic_cleaning")
        _ = mlflow.run(
            os.path.join(components_dir, "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": _get(cfg, "data.raw_artifact", "raw_data.csv:latest"),
                "output_artifact": _get(cfg, "data.cleaned_artifact", "clean_sample.csv"),
                "output_type": "cleaned_data",
                "output_description": "Data after basic cleaning",
                "min_price": str(_get(cfg, "data.min_price", 10)),
                "max_price": str(_get(cfg, "data.max_price", 350)),
            },
            env_manager="local",
        )

    # -----------------------------
    # 3) DATA CHECK
    # -----------------------------
    if "data_check" in active_steps:
        log.info("Running step: data_check")
        _ = mlflow.run(
            os.path.join(components_dir, "data_check"),
            "main",
            parameters={
                "csv": _get(cfg, "data.cleaned_artifact", "clean_sample.csv:latest"),
                "ref": _get(cfg, "data.ref_artifact", "clean_sample.csv:reference"),
                "kl_threshold": str(_get(cfg, "data.kl_threshold", 0.2)),
            },
            env_manager="local",
        )

    # -----------------------------
    # 4) DATA SPLIT
    # -----------------------------
    if "data_split" in active_steps:
        log.info("Running step: data_split")
        _ = mlflow.run(
            os.path.join(components_dir, "train_val_test_split"),
            "main",
            parameters={
                "csv": _get(cfg, "data.cleaned_artifact", "clean_sample.csv:latest"),
                "test_size": str(_get(cfg, "modeling.test_size", 0.2)),
                "random_seed": str(_get(cfg, "modeling.random_seed", 42)),
                "stratify_by": _get(cfg, "modeling.stratify_by", "neighbourhood_group"),
            },
            env_manager="local",
        )

    # -----------------------------
    # 5) TRAIN RANDOM FOREST
    # -----------------------------
    if "train_random_forest" in active_steps:
        log.info("Running step: train_random_forest")

        # Build/override RF config into a JSON file for the step
        rf_cfg = {
            "n_estimators": _get(cfg, "modeling.random_forest.n_estimators", 100),
            "max_depth": _get(cfg, "modeling.random_forest.max_depth", 15),
            "max_features": _get(cfg, "modeling.random_forest.max_features", 0.5),
            "min_samples_split": _get(cfg, "modeling.random_forest.min_samples_split", 2),
            "min_samples_leaf": _get(cfg, "modeling.random_forest.min_samples_leaf", 1),
            "n_jobs": _get(cfg, "modeling.random_forest.n_jobs", -1),
            "random_state": _get(cfg, "modeling.random_forest.random_state", 42),
        }

        rf_config_path = os.path.join(os.getcwd(), "rf_config.json")
        with open(rf_config_path, "w") as f:
            json.dump(rf_cfg, f)

        _ = mlflow.run(
            os.path.join(repo_root, "src", "train_random_forest"),
            "main",
            parameters={
                # Per lesson hint, use this exact artifact name
                "trainval_artifact": _get(cfg, "data.trainval_artifact", "trainval_data.csv:latest"),
                "val_size": str(_get(cfg, "modeling.val_size", 0.2)),
                "random_seed": str(_get(cfg, "modeling.random_seed", 42)),
                "stratify_by": _get(cfg, "modeling.stratify_by", "neighbourhood_group"),
                "rf_config": rf_config_path,
                "max_tfidf_features": str(_get(cfg, "modeling.max_tfidf_features", 5)),
                # Per lesson hint, use this exact output name
                "output_artifact": _get(cfg, "modeling.output_artifact", "random_forest_export"),
            },
            env_manager="local",
        )

    # -----------------------------
    # 6) TEST REGRESSION MODEL (after promotion to prod)
    # -----------------------------
    if "test_regression_model" in active_steps:
        log.info("Running step: test_regression_model")
        _ = mlflow.run(
            os.path.join(components_dir, "test_regression_model"),
            "main",
            parameters={
                # Use the model that was promoted to prod
                "mlflow_model": _get(cfg, "testing.mlflow_model", "random_forest_export:prod"),
                # Use the latest test split
                "test_artifact": _get(cfg, "testing.test_artifact", "test_data.csv:latest"),
            },
            env_manager="local",
        )

    log.info("Pipeline completed.")


if __name__ == "__main__":
    go()
