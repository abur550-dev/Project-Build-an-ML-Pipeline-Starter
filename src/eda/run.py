import json
from pathlib import Path
import mlflow

def main():
    # create a local file
    out_dir = Path("outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "summary.json"
    f.write_text(json.dumps({"status": "ok", "note": "EDA step ran headlessly"}))

    # log it as an MLflow artifact so it persists
    mlflow.log_artifact(str(f), artifact_path="eda")

    print("✅ EDA ran headlessly and logged artifact: eda/summary.json")

if __name__ == "__main__":
    # Projects already create a run, but this is safe either way
    with mlflow.start_run(nested=True):
        main()
