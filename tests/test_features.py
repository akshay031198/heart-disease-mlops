"""Unit tests for the preprocessing pipeline."""
import numpy as np
import pandas as pd

from heart_disease.config import FEATURES, NUMERIC_FEATURES
from heart_disease.features import build_preprocessor


def test_preprocessor_output_has_no_nans(clean_df):
    X = clean_df[FEATURES].copy()
    X.loc[0, "chol"] = np.nan  # imputer must handle missing values at inference time
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    assert out.shape[0] == len(X)


def test_numeric_features_are_scaled(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    out = pre.transform(clean_df[FEATURES])[:, : len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-6)


def test_one_hot_encoding_expands_columns(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    names = pre.get_feature_names_out()
    assert len(names) > len(FEATURES)
    assert any(n.startswith("cat__cp_") for n in names)


def test_unknown_category_is_ignored(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    row = clean_df[FEATURES].iloc[[0]].copy()
    row["restecg"] = 99  # unseen category must not crash inference
    out = pre.transform(pd.DataFrame(row))
    assert out.shape[1] == len(pre.get_feature_names_out())
