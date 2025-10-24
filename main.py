# main.py
from __future__ import annotations
import logging, os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse, urlunparse, quote_from_bytes
import hydra
import mlflow

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="[%(asctime)s][%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

@dataclass
class MainConfig:
    steps: str = "all"
    steps_root: str = "src"
    default_steps: List[str] = None
    def __post_init__(self):
        if self.default_steps is None:
            self.default_steps = ["eda", "basic_cleaning", "data_check", "train_random_forest"]

def _as_list(s: str, default: List[str]) -> List[str]:
    s = (s or "").strip()
    return list(default) if s.lower() in ("", "all") else [p.strip() for p in s.split(",") if p.strip()]

def _to_file_uri(p: Path) -> str:
    path = p.resolve().as_posix()
    if not path.startswith("/"): path = "/" + path
    quoted = quote_from_bytes(path.encode("utf-8"), safe="/-_.~")
    return urlunparse(("file", "", quoted, "", "", ""))  # file:///...

@hydra.main(config_name="main", version_base=None)
def go(cfg: MainConfig) -> None:
    repo = Path(hydra.utils.get_original_cwd()).resolve()
    steps_dir = (repo / cfg.steps_root).resolve()
    steps = _as_list(cfg.steps, cfg.default_steps)
    log.info("Active pipeline steps: %s", steps)

    for s in steps:
        step_dir = (steps_dir / s)
        if not (step_dir / "MLproject").exists():
            raise FileNotFoundError(f"Missing MLproject for step '{s}': {step_dir/'MLproject'}")
        uri = _to_file_uri(step_dir)
        log.info("Running step '%s' from %s", s, uri)
        assert urlparse(uri).scheme == "file"
        mlflow.run(
            uri=uri,
            entry_point="main",
            parameters={},
            env_manager="local",   # <— use your current env, avoids pyenv/conda issues
            synchronous=True,
        )

if __name__ == "__main__":
    go()
