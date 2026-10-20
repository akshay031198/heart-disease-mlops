"""Unit tests for data loading and cleaning."""
import pandas as pd

from heart_disease.config import CATEGORICAL_FEATURES, FEATURES, TARGET
from heart_disease.data import clean_data, load_raw_data, split_features_target


def test_clean_data_removes_missing_values(clean_df):
    assert clean_df[FEATURES].isna().sum().sum() == 0


def test_clean_data_binarizes_target(clean_df):
    assert TARGET in clean_df.columns
    assert "num" not in clean_df.columns
    assert set(clean_df[TARGET].unique()) <= {0, 1}


def test_clean_data_output_columns(clean_df):
    assert list(clean_df.columns) == FEATURES + [TARGET]


def test_categorical_columns_are_integers(clean_df):
    for col in CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(clean_df[col]), col


def test_clean_data_drops_duplicates(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[5, 6]]], ignore_index=True)
    assert len(clean_data(duplicated)) == len(clean_data(raw_df))


def test_clean_data_drops_rows_without_target(raw_df):
    raw_df.loc[10, "num"] = None
    assert len(clean_data(raw_df)) == len(raw_df) - 1


def test_clean_data_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_load_raw_data_parses_question_marks(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
                 "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,?,2\n")
    df = load_raw_data(f)
    assert df.shape == (2, 14)
    assert df["ca"].isna().sum() == 1 and df["thal"].isna().sum() == 1


def test_split_features_target(clean_df):
    X, y = split_features_target(clean_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
