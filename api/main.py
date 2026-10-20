"""FastAPI model-serving application.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /          service info
    GET  /health    liveness / readiness
    POST /predict   single prediction (JSON) -> prediction + confidence
    POST /predict/batch  many predictions
    GET  /metrics   Prometheus metrics
    GET  /docs      Swagger UI
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.monitoring import (
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_ERRORS,
    PREDICTIONS_TOTAL,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    setup_logging,
)
from api.schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse
from heart_disease.config import METADATA_PATH, MODEL_PATH
from heart_disease.predict import load_metadata, load_model, predict

logger = setup_logging()
STATE = {"model": None, "metadata": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    metadata_path = Path(os.getenv("METADATA_PATH", str(METADATA_PATH)))
    try:
        STATE["model"] = load_model(str(model_path))
        STATE["metadata"] = load_metadata(metadata_path)
        MODEL_LOADED.set(1)
        logger.info("Model loaded from %s (%s)", model_path, STATE["metadata"].get("model_name", "unknown"))
    except Exception as exc:  # keep the API up so /health can report the problem
        MODEL_LOADED.set(0)
        logger.error("Failed to load model from %s: %s", model_path, exc)
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart-disease risk from patient health data (UCI dataset).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Log every request (id, method, path, status, latency) and record Prometheus metrics."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        latency = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        if endpoint != "/metrics":
            REQUEST_COUNT.labels(request.method, endpoint, str(status_code)).inc()
            REQUEST_LATENCY.labels(endpoint).observe(latency)
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "latency_ms": round(latency * 1000, 2),
            "client": request.client.host if request.client else None,
        }))
    response.headers["X-Request-ID"] = request_id
    return response


def _get_model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return STATE["model"]


def _to_response(result: dict) -> PredictionResponse:
    md = STATE["metadata"]
    PREDICTIONS_TOTAL.labels(str(result["prediction"])).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    return PredictionResponse(
        **result,
        model_name=md.get("model_name", "unknown"),
        model_version=md.get("model_version", "unknown"),
    )


@app.get("/")
def root():
    return {
        "service": "heart-disease-api",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict/batch", "/metrics"],
    }


@app.get("/health")
def health():
    if STATE["model"] is None:
        return Response(content=json.dumps({"status": "unhealthy", "model_loaded": False}),
                        status_code=503, media_type="application/json")
    md = STATE["metadata"]
    return {"status": "ok", "model_loaded": True, "model_name": md.get("model_name"),
            "trained_at": md.get("trained_at")}


@app.post("/predict", response_model=PredictionResponse)
def predict_one(patient: PatientFeatures):
    model = _get_model()
    try:
        result = predict(model, patient.model_dump())[0]
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    logger.info("prediction=%s confidence=%.3f input=%s", result["prediction"], result["confidence"],
                patient.model_dump())
    return _to_response(result)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    model = _get_model()
    try:
        results = predict(model, [r.model_dump() for r in batch.records])
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    preds = [_to_response(r) for r in results]
    return BatchResponse(predictions=preds, count=len(preds))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
