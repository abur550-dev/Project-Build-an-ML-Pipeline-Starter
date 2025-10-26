from __future__ import annotations
import os, sys, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlunparse, quote_from_bytes
import mlflow

logging.basicConfig(level=os.environ.get("LOGLEVEL","INFO"),
                    format="[%(asctime)s][%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

@dataclass
class Cfg:
    steps: str = "all"
    steps_root: str = "src"
    default_steps: List[str] = None
    def __post_init__(self):
        if self.default_steps is None:
            self.default_steps = ["eda","basic_cleaning","data_check","train_random_forest","test_regression_model"]

def parse_known(argv: List[str]) -> Tuple[str, Dict[str,str]]:
    # Parse main.steps=..., --steps X, and collect any --key value or --key=value options.
    steps = None
    params: Dict[str,str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("main.steps="):
            steps = a.split("=",1)[1].strip("'\"")
        elif a == "--steps" and i+1 < len(argv):
            steps = argv[i+1]; i += 1
        elif a.startswith("--steps="):
            steps = a.split("=",1)[1]
        elif a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, v = key.split("=",1)
                params[k] = v
            else:
                if i+1 < len(argv) and not argv[i+1].startswith("-"):
                    params[key] = argv[i+1]; i += 1
                else:
                    params[key] = "true"
        i += 1
    return (steps or "all", params)

def as_list(s: str, default: List[str]) -> List[str]:
    s = (s or "").strip()
    if s.lower() in ("","all"): return list(default)
    return [p.strip() for p in s.split(",") if p.strip()]

def to_file_uri(p: Path) -> str:
    path = p.resolve().as_posix()
    if not path.startswith("/"): path = "/" + path
    quoted = quote_from_bytes(path.encode("utf-8"), safe="/-_.~")
    return urlunparse(("file","",quoted,"","",""))

def main():
    cfg = Cfg()
    steps_str, extra_params = parse_known(sys.argv[1:])
    repo = Path.cwd().resolve()
    steps_dir = (repo / cfg.steps_root).resolve()

    steps = as_list(steps_str, cfg.default_steps)
    log.info("Active pipeline steps: %s", steps)
    if extra_params:
        log.info("Extra step parameters detected: %s", extra_params)

    for s in steps:
        step_dir = steps_dir / s
        mlp = step_dir / "MLproject"
        if not mlp.exists():
            raise FileNotFoundError(f"Missing MLproject for step '{s}': {mlp}")
        uri = to_file_uri(step_dir)
        log.info("Step '%s' dir: %s", s, step_dir)
        log.info("Step '%s' uri: %s", s, uri)

        # 🔑 Forward ALL extra params to every step; each step's MLproject will validate what's needed.
        params = dict(extra_params)

        log.info("Calling mlflow.run(entry_point='main', env_manager='local', parameters=%s)", params)
        mlflow.run(
            uri=uri,
            entry_point="main",
            parameters=params,
            env_manager="local",
            synchronous=True,
        )
        log.info("✅ Step '%s' completed", s)

if __name__ == "__main__":
    main()
