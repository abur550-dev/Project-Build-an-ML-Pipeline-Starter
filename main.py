import os
import json
import logging
from typing import List

import mlflow
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)

STEPS_DEFAULT: List[str] = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model",  # run explicitly after promoting a model to prod
]

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig):

    steps_val = config["main"].get("steps", STEPS_DEFAULT)
    active_steps = list(steps_val) if isinstance(steps_val, (list, tuple)) else [
        st.strip() for st in str(steps_val).split(",") if st.strip()
    ]
    log.info("Active pipeline steps: %s", active_steps)

    # --- download ---
    if "download" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "get_data"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            },
            env_manager="local",
        )

    # --- basic_cleaning ---
    if "basic_cleaning" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": config["data"]["artifact_name"] + ":latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Cleaned sample with price/geo filters",
                "min_price": config["basic_cleaning"]["min_price"],
                "max_price": config["basic_cleaning"]["max_price"],
            },
            env_manager="local",
        )

    # --- data_check ---
    if "data_check" in active_steps:
        clean_ref = "clean_sample.csv:latest"
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "data_check"),
            "main",
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

    # --- data_split ---
    if "data_split" in active_steps:
        _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            "main",
            parameters={
                "input_artifact": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
            env_manager="local",
        )

    # --- train_random_forest ---
    if "train_random_forest" in active_steps:
        rf_config_path = os.path.abspath("rf_config.json")
        with open(rf_config_path, "w") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        _ = mlflow.run(
            os.path.join(get_original_cwd(), "src", "train_random_forest"),
            "main",
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

    # --- test_regression_model (run after promoting a model to prod) ---
    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            os.path.join(get_original_cwd(), "components", "test_regression_model"),
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_artifact": "test_data.csv:latest",
            },
            env_manager="local",
        )

if __name__ == "__main__":
    go()
