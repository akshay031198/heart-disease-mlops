"""Data loading and cleaning utilities."""
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from heart_disease.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw UCI file (no header, '?' marks missing values)."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    logger.info("Loaded raw data with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Steps:
      1. Replace '?' placeholders with NaN and coerce every column to numeric.
      2. Drop exact duplicate rows and rows with a missing target.
      3. Convert multi-class 'num' (0-4) into a binary target (0 = absent, 1 = present).
      4. Impute missing values: median for numeric, mode for categorical.
      5. Cast categorical columns to int.
    """
    df = df.copy()
    df = df.replace("?", pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info("Dropped %d duplicate rows", before - len(df))

    if "num" in df.columns:
        df = df.dropna(subset=["num"])
        df[TARGET] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])
    elif TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df[TARGET] = (df[TARGET] > 0).astype(int)

    missing = df[FEATURES].isna().sum()
    if missing.any():
        logger.info("Missing values before imputation:\n%s", missing[missing > 0])

    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype(float).fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else 0
        df[col] = df[col].fillna(fill_value).astype(int)

    return df[FEATURES + [TARGET]].reset_index(drop=True)


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned CSV produced by `heart_disease.download_data`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run `python -m heart_disease.download_data` first."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into X (features) and y (target)."""
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()
