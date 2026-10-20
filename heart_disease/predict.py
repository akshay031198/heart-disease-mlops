"""Inference helpers shared by the CLI and the API.

CLI usage:
    python -m heart_disease.predict --input samples/sample_request.json
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import joblib
import pandas as pd

from heart_disease.config import FEATURES, METADATA_PATH, MODEL_PATH


@lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m heart_disease.train` first.")
    return joblib.load(path)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def predict(model, records: Union[Dict, List[Dict]]) -> List[Dict]:
    """Predict for one or more patient records. Returns prediction + confidence."""
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    df = df[FEATURES]
    proba = model.predict_proba(df)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        results.append(
            {
                "prediction": pred,
                "label": "Heart disease present" if pred == 1 else "No heart disease",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if pred == 1 else 1 - p), 4),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from a JSON file")
    parser.add_argument("--input", required=True, help="JSON file with one record or a list of records")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(predict(load_model(args.model), records), indent=2))


if __name__ == "__main__":
    main()
