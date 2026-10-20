"""Unit tests for model training, evaluation and inference."""
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from heart_disease.config import FEATURES, TARGET
from heart_disease.features import build_pipeline
from heart_disease.predict import predict
from heart_disease.train import cross_validation_metrics, evaluate, get_model_candidates


def test_model_candidates_include_two_models():
    candidates = get_model_candidates(quick=True)
    assert {"logistic_regression", "random_forest"} <= set(candidates)
    assert len(get_model_candidates()) >= 2


def test_pipeline_predict_proba_range(trained_model, clean_df):
    proba = trained_model.predict_proba(clean_df[FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_all_metrics(trained_model, clean_df):
    metrics = evaluate(trained_model, clean_df[FEATURES], clean_df[TARGET])
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_model_learns_better_than_chance(trained_model, clean_df):
    metrics = evaluate(trained_model, clean_df[FEATURES], clean_df[TARGET])
    assert metrics["test_roc_auc"] > 0.6


def test_cross_validation_metrics(clean_df):
    model = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    m = cross_validation_metrics(model, clean_df[FEATURES], clean_df[TARGET], cv)
    assert "cv_roc_auc_mean" in m and "cv_accuracy_std" in m


def test_predict_returns_prediction_and_confidence(trained_model, sample_record):
    result = predict(trained_model, sample_record)[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0


def test_predict_batch(trained_model, sample_record):
    assert len(predict(trained_model, [sample_record, sample_record])) == 2


def test_predict_missing_feature_raises(trained_model, sample_record):
    sample_record.pop("age")
    with pytest.raises(ValueError):
        predict(trained_model, sample_record)


def test_model_serialization_roundtrip(trained_model, clean_df, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(trained_model, path)
    loaded = joblib.load(path)
    X = clean_df[FEATURES]
    assert (loaded.predict(X) == trained_model.predict(X)).all()
