import os
import json
import logging
from typing import List

import mlflow
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Default steps for a normal end-to-end run
STEPS_DEFAULT: List[str] = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model",  # run explicitly after promoting a model to prod
]


@hydra.main(config_path="src", config_name="config")
def go(config: DictConfig) -> None:
    """
    Orchestrates the pipeline via MLflow Projects.
    Hydra looks for config at src/config.yaml.
    """

    # Normalize steps from Hydra config into a list
    steps_val = config["main"].get("steps", STEPS_DEFAULT)
    if isinstance(steps_val, (list, tuple)):
        active_steps = list(steps_val)
    else:
        active_steps = [st.strip() for st in str(steps_val).split(",") if st.strip()]
    log.info("Active pipeline steps: %s", active_steps)

    # So that relative paths inside components resolve correctly
    repo_root = get_original_cwd()

    # -------------------------------------------------------------------------
    # 1) DOWNLOAD
    # -------------------------------------------------------------------------
    if "download" in active_steps:
        _ = mlflow.run(
            os.path.join(repo_root, "src", "get_data"),
            entry_point="main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            },
            env_manager="local",
        )

    # -------------------------------------------------------------------------
    # 2) BASIC CLEANING
    # -------------------------------------------------------------------------
    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            os.path.join(repo_root, "src", "basic_cleaning"),
            entry_point="main",
            parameters={
                "input_artifact": f"{config['data']['artifact_name']}:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Cleaned data with price/geo filters",
                "min_price": config["basic_cleaning"]["min_price"],
                "max_price": config["basic_cleaning"]["max_price"],
            },
            env_manager="local",
        )

    # -------------------------------------------------------------------------
    # 3) DATA CHECK
    # -------------------------------------------------------------------------
    if "data_check" in active_steps:
        clean_ref = "clean_sample.csv:latest"
        _ = mlflow.run(
            os.path.join(repo_root, "src", "data_check"),
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

    # -------------------------------------------------------------------------
    # 4) TRAIN/VAL/TEST SPLIT
    # -------------------------------------------------------------------------
    if "data_split" in active_steps:
        # This component usually lives under components/train_val_test_split
        split_comp_path = os.path.join(
            config["main"]["components_repository"], "train_val_test_split"
        )
        _ = mlflow.run(
            split_comp_path,
            entry_point="main",
            parameters={
                "input_artifact": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
            env_manager="local",
        )

    # -------------------------------------------------------------------------
    # 5) TRAIN RANDOM FOREST
    # -------------------------------------------------------------------------
    if "train_random_forest" in active_steps:
        # Serialize RF config (the training step expects a JSON file path)
        rf_cfg_path = os.path.abspath("rf_config.json")
        with open(rf_cfg_path, "w") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        _ = mlflow.run(
            os.path.join(repo_root, "src", "train_random_forest"),
            entry_point="main",
            parameters={
                # Hints specify these exact values:
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
                "rf_config": rf_cfg_path,
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "output_artifact": "random_forest_export",
            },
            env_manager="local",
        )

    # -------------------------------------------------------------------------
    # 6) TEST REGRESSION MODEL  (Run manually after promoting model to prod)
    # -------------------------------------------------------------------------
    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            os.path.join(repo_root, "components", "test_regression_model"),
            entry_point="main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_artifact": "test_data.csv:latest",
            },
            env_manager="local",
        )


if __name__ == "__main__":
    go()
