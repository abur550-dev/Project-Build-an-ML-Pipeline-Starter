import os
import json
import logging
from typing import List

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)

# Default pipeline steps
STEPS_DEFAULT: List[str] = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model",  # run explicitly after promoting a model to prod
]


@hydra.main(config_path="src", config_name="config", version_base=None)
def go(config: DictConfig) -> None:
    """
    Orchestrates the ML pipeline via mlflow.run() calls.
    Use `-P steps=...` to select specific steps (comma-separated).
    """

    # Normalize steps into a list
    steps_val = config["main"].get("steps", STEPS_DEFAULT)
    if isinstance(steps_val, (list, tuple)):
        active_steps = list(steps_val)
    else:
        active_steps = [s.strip() for s in str(steps_val).split(",") if s.strip()]

    log.info("Active pipeline steps: %s", active_steps)

    # --- download ------------------------------------------------------------
    if "download" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "get_data"),
            entry_point="main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            },
            env_manager="local",
        )

    # --- basic_cleaning ------------------------------------------------------
    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "basic_cleaning"),
            entry_point="main",
            parameters={
                "input_artifact": f"{config['data']['artifact_name']}:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Cleaned sample with price/geo filters",
                "min_price": config["basic_cleaning"]["min_price"],
                "max_price": config["basic_cleaning"]["max_price"],
            },
            env_manager="local",
        )

    # --- data_check ----------------------------------------------------------
    if "data_check" in active_steps:
        clean_ref = "clean_sample.csv:latest"
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "data_check"),
            entry_point="main",
            parameters={
                "csv": clean_ref,
                "ref": clean_ref,
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["basic_cleaning"]["min_price"],
                "max_price": config["basic_cleaning"]["max_price"],
                "neigh_col": config["data_check"]["neigh_col"],
            },
            env_manager="local",
        )

    # --- data_split: split into train/val and test ---------------------------
    if "data_split" in active_steps:
        comp_repo = config["main"]["components_repository"]
        _ = mlflow.run(
            f"{comp_repo}/train_val_test_split",
            entry_point="main",
            parameters={
                "input_artifact": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
            env_manager="local",
        )

    # --- train_random_forest -------------------------------------------------
    if "train_random_forest" in active_steps:
        # Serialize RF config to JSON
        rf_cfg_obj = OmegaConf.to_object(config["modeling"]["random_forest"])
        rf_config_path = os.path.abspath("rf_config.json")
        with open(rf_config_path, "w") as fp:
            json.dump(rf_cfg_obj, fp)

        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "train_random_forest"),
            entry_point="main",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
                "rf_config": rf_config_path,
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "output_artifact": "random_forest_export",
            },
            env_manager="local",
        )

    # --- test_regression_model (run explicitly after promoting a model) ------
    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "components", "test_regression_model"),
            entry_point="main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_artifact": "test_data.csv:latest",
            },
            env_manager="local",
        )


if __name__ == "__main__":
    go()
