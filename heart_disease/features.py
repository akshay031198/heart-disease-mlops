"""Feature engineering: a reusable sklearn preprocessing pipeline.

The preprocessor is bundled *inside* the saved model pipeline, so the exact same
transformations are applied at training time and at inference time (API).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute + standard-scale. Categorical: mode-impute + one-hot."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline = preprocessing + classifier."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)])
