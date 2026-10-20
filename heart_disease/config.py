"""Central configuration: paths, column names, and constants used across the project."""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "heart_clean.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: a local SQLite backend works out-of-the-box on Windows/Linux/macOS.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "heart-disease-classification")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG results (0-2)
    "thalach",   # max heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # original target: 0 (no disease) .. 4
]

TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
