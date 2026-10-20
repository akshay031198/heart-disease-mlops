"""Model training with hyper-parameter tuning, cross-validation and MLflow tracking.

Usage:
    python -m heart_disease.train            # full grid search
    python -m heart_disease.train --quick    # smaller grids (used in CI)

Every model family gets a parent MLflow run (best params, CV + test metrics, plots,
serialized model). Each hyper-parameter candidate is logged as a nested child run.
The best model (by mean CV ROC-AUC) is saved to models/model.joblib for serving.
"""
import argparse
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from heart_disease import __version__  # noqa: E402
from heart_disease.config import (  # noqa: E402
    CV_FOLDS,
    FEATURES,
    METADATA_PATH,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from heart_disease.data import load_clean_data, split_features_target  # noqa: E402
from heart_disease.features import build_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_model_candidates(quick: bool = False) -> dict:
    """Model families and their hyper-parameter grids (keys use pipeline step prefix)."""
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                {"classifier__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                {"classifier__n_estimators": [100], "classifier__max_depth": [None, 5]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [None, 4, 8],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [2, 3],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
    }


def evaluate(model, X, y) -> dict:
    """Hold-out test-set metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_precision": precision_score(y, y_pred, zero_division=0),
        "test_recall": recall_score(y, y_pred, zero_division=0),
        "test_f1": f1_score(y, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, y_proba),
    }


def cross_validation_metrics(model, X, y, cv) -> dict:
    """Mean/std of each metric across stratified K folds."""
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    out = {}
    for metric in SCORING:
        vals = scores[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(vals))
        out[f"cv_{metric}_std"] = float(np.std(vals))
    return out


def get_feature_names(pipeline) -> list:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def save_plots(model, X_test, y_test, name: str, out_dir: Path) -> list:
    """Confusion matrix, ROC, PR and feature-importance plots. Returns file paths."""
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {name}")
    p = out_dir / f"{name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve - {name}")
    p = out_dir / f"{name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Precision-Recall - {name}")
    p = out_dir / f"{name}_pr_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        label = "Feature importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        label = "|Coefficient|"
    else:
        importances = None
    if importances is not None:
        fi = pd.Series(importances, index=get_feature_names(model)).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        fi.plot(kind="barh", ax=ax, color="#4c72b0")
        ax.set_title(f"Top features - {name}")
        ax.set_xlabel(label)
        p = out_dir / f"{name}_feature_importance.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        kwargs = {}
        if MLFLOW_TRACKING_URI.startswith(("sqlite", "file")):
            kwargs["artifact_location"] = MLFLOW_ARTIFACT_ROOT
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, **kwargs)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s | experiment: %s", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)


def train_and_log_model(name, estimator, grid, X_train, y_train, X_test, y_test, cv) -> dict:
    """Grid-search one model family and log everything to MLflow."""
    logger.info("=== Training %s (%d candidates) ===", name, len(list(_iter_grid(grid))))
    pipeline = build_pipeline(estimator)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "experiment", "project_version": __version__})
        search.fit(X_train, y_train)

        # Child runs: one per hyper-parameter candidate
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}_cand_{i:02d}", nested=True):
                mlflow.log_params({k.replace("classifier__", ""): v for k, v in params.items()})
                mlflow.log_metrics(
                    {
                        "cv_roc_auc_mean": float(res["mean_test_score"][i]),
                        "cv_roc_auc_std": float(res["std_test_score"][i]),
                        "rank": int(res["rank_test_score"][i]),
                    }
                )

        best = search.best_estimator_
        best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        cv_metrics = cross_validation_metrics(best, X_train, y_train, cv)
        test_metrics = evaluate(best, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_params({"model_family": name, "cv_folds": CV_FOLDS, "test_size": TEST_SIZE,
                           "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for p in save_plots(best, X_test, y_test, name, tmp):
                mlflow.log_artifact(str(p), artifact_path="plots")
            report = classification_report(y_test, best.predict(X_test), digits=3)
            (tmp / "classification_report.txt").write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(tmp / "classification_report.txt"), artifact_path="reports")
            pd.DataFrame(res).drop(columns=["params"]).to_csv(tmp / "cv_results.csv", index=False)
            mlflow.log_artifact(str(tmp / "cv_results.csv"), artifact_path="reports")

        # Declare all inputs as float64 so MLflow's schema tolerates missing values at inference
        example = X_train.head(3).astype("float64")
        signature = infer_signature(X_train.astype("float64"), best.predict_proba(X_train))
        model_info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=example,
            # MLflow 3.x serialises sklearn models with skops; these are our own trusted objects
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        logger.info("%s | best params: %s", name, best_params)
        logger.info("%s | CV ROC-AUC %.4f | test ROC-AUC %.4f | test acc %.4f", name,
                    cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"],
                    test_metrics["test_accuracy"])
        return {
            "name": name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model": best,
            "best_params": best_params,
            **cv_metrics,
            **test_metrics,
        }


def _iter_grid(grid):
    from sklearn.model_selection import ParameterGrid

    return ParameterGrid(grid)


def save_comparison_plot(results: list) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).set_index("name")
    cols = ["cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean", "cv_f1_mean", "cv_roc_auc_mean"]
    fig, ax = plt.subplots(figsize=(10, 5))
    df[cols].T.plot(kind="bar", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model comparison (5-fold CV means)")
    ax.set_xticklabels([c.replace("cv_", "").replace("_mean", "") for c in cols], rotation=0)
    fig.tight_layout()
    path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(quick: bool = False) -> dict:
    df = load_clean_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %s | Test: %s | positive rate: %.2f", X_train.shape, X_test.shape, y.mean())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    setup_mlflow()
    results = [
        train_and_log_model(name, est, grid, X_train, y_train, X_test, y_test, cv)
        for name, (est, grid) in get_model_candidates(quick).items()
    ]

    # ----- Model selection: highest mean CV ROC-AUC (test set kept for unbiased reporting)
    best = max(results, key=lambda r: r["cv_roc_auc_mean"])
    logger.info("Best model: %s (CV ROC-AUC %.4f)", best["name"], best["cv_roc_auc_mean"])

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("model",)} for r in results])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    comparison_plot = save_comparison_plot(results)
    print("\n" + table[["name", "cv_roc_auc_mean", "cv_accuracy_mean", "test_roc_auc",
                        "test_accuracy", "test_precision", "test_recall"]].round(4).to_string(index=False))

    # ----- Persist final model + metadata for the serving layer
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    metadata = {
        "model_name": best["name"],
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": best["run_id"],
        "features": FEATURES,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in best["best_params"].items()},
        "metrics": {k: round(float(best[k]), 4) for k in best
                    if k.startswith(("cv_", "test_"))},
        "sklearn_version": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved final model to %s and metadata to %s", MODEL_PATH, METADATA_PATH)

    # ----- Log the champion summary + register the model in the MLflow Model Registry
    with mlflow.start_run(run_name="champion_selection"):
        mlflow.set_tag("stage", "selection")
        mlflow.log_param("champion", best["name"])
        mlflow.log_metric("champion_cv_roc_auc", best["cv_roc_auc_mean"])
        mlflow.log_artifact(str(REPORTS_DIR / "model_comparison.csv"))
        mlflow.log_artifact(str(comparison_plot))
        mlflow.log_artifact(str(METADATA_PATH))
    try:
        mlflow.register_model(best["model_uri"], "heart-disease-classifier")
    except Exception as exc:  # registry is optional (e.g. remote server without registry)
        logger.warning("Model registration skipped: %s", exc)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart-disease classifiers")
    parser.add_argument("--quick", action="store_true", help="Use small grids (fast, for CI)")
    args = parser.parse_args()
    main(quick=args.quick)
