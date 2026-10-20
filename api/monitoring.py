"""Logging configuration and Prometheus metrics for the API."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prometheus metrics (scraped from GET /metrics)
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total", "Number of predictions by predicted class", ["prediction"]
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of model predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
PREDICTION_ERRORS = Counter("model_prediction_errors_total", "Failed prediction attempts")
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded and ready, else 0")


def setup_logging() -> logging.Logger:
    """Log to stdout (for docker/kubectl logs) and to a rotating file (logs/api.log)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("heart_api")
    logger.setLevel(level)
    if logger.handlers:  # avoid duplicate handlers on reload
        return logger

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_dir / "api.log", maxBytes=5_000_000, backupCount=3,
                                           encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log directory %s; logging to stdout only", log_dir)
    logger.propagate = False
    return logger
