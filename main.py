# main.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlflow
from mlflow.utils.file_utils import path_to_local_file_uri

import hydra
from omegaconf import OmegaConf

# ----------------------------
# Config
# ----------------------------
@dataclass
class MainConfig:
    # Comma-separated list of steps to run, or "all"
    steps: str = "all"

    # Optional MLflow experiment name; if blank we'll use the folder name
    experiment_name: str = ""

    # Where your step subprojects live
    steps_root: str = "src"

    # Default ordered pipeline when steps == "all"
    default_steps: List[str] = None  # set in __post_init__

    def __post_init__(self):
        if self.default_steps is None:
            # Adjust this to your repo’s steps if needed
            self.default_steps = [
                # "download",
                # "preprocess",
                # "check_data",
                # "segregate",
                # "train_random_forest",
                "test_regression_model",
            ]


# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)


# ----------------------------
# Helpers
# ----------------------------
def _as_list(steps_str: str, default_steps: List[str]) -> List[str]:
    s = (steps_str or "").strip()
    if s.lower() == "all" or s == "":
        return list(default_steps)
    return [p.strip() for p in s.split(",") if p.strip()]


def _ensure_mlproject_exists(step_dir: Path):
    mlp = step_dir / "MLproject"
    if not mlp.exists():
        raise FileNotFoundError(
            f"No MLproject file found at: {mlp}\n"
            f"Each step directory must be an MLflow Project."
        )


def _file_uri(p: Path) -> str:
    # Force MLflow to treat this as a local project, never a git repo
    return path_to_local_file_uri(str(p.resolve()))


# ----------------------------
# Main
# ----------------------------
@hydra.main(config_name="main", version_base=None)
def go(cfg: MainConfig) -> None:
    # Resolve paths relative to the original working dir (repo root when invoked by MLflow)
    repo_root = Path(hydra.utils.get_original_cwd()).resolve()
    steps_root = (repo_root / cfg.steps_root).resolve()

    # Choose experiment name
    exp_name = cfg.experiment_name or repo_root.name
    mlflow.set_experiment(exp_name)

    # Determine which steps to run
    steps_to_run = _as_list(cfg.steps, cfg.default_steps)
    logger.info("Active pipeline steps: %s", steps_to_run)

    for step in steps_to_run:
        step_dir = (steps_root / step).resolve()
        _ensure_mlproject_exists(step_dir)

        project_uri = _file_uri(step_dir)  # <-- key fix: use file:// URI

        logger.info("Running step '%s' from %s", step, project_uri)

        # If your step defines parameters in its MLproject entry point,
        # pass them via 'parameters={...}' here.
        #
        # By default we call the "main" entry point.
        #
        # Note: We deliberately do NOT pass 'version' so MLflow won't try to do any git ops.
        try:
            _ = mlflow.run(
                uri=project_uri,
                entry_point="main",
                parameters={},        # add step-specific params if needed
                use_conda=True,       # or False if you prefer your active env
                env_manager=None,     # let MLflow decide based on use_conda
                synchronous=True,     # wait for completion before next step
            )
        except Exception as e:
            logger.error(
                "Step '%s' failed. If you previously saw a 'not a git repository' error, "
                "this file-URI approach prevents MLflow from calling git in subdirs. "
                "Original error: %s",
                step,
                str(e),
            )
            raise


if __name__ == "__main__":
    go()
