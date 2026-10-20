"""Shared pytest fixtures. Tests use synthetic data so they run without network access."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_disease.config import COLUMN_NAMES, FEATURES, TARGET
from heart_disease.features import build_pipeline


def _synthetic_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(94, 200, n).astype(float),
        "chol": rng.integers(126, 564, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(71, 202, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    # make the target loosely depend on features so models can learn something
    score = (df["cp"] == 4) * 1.5 + df["exang"] + df["oldpeak"] / 2 - (df["thalach"] - 150) / 30
    df["num"] = np.where(score + rng.normal(0, 0.5, n) > 1.2, rng.integers(1, 5, n), 0)
    return df[COLUMN_NAMES]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    df = _synthetic_raw().astype(object)
    # inject UCI-style missing markers
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    df.loc[2, "chol"] = np.nan
    return df


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from heart_disease.data import clean_data

    return clean_data(raw_df)


@pytest.fixture
def sample_record() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
            "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture
def trained_model(clean_df):
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(clean_df[FEATURES], clean_df[TARGET])
    return model


@pytest.fixture
def model_files(tmp_path, trained_model):
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metadata.json"
    joblib.dump(trained_model, model_path)
    meta_path.write_text(json.dumps({"model_name": "test_lr", "model_version": "test"}))
    return model_path, meta_path
