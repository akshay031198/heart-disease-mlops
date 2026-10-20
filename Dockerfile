# ---------- Heart Disease prediction API ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    LOG_DIR=/app/logs

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Application code + trained model (run `python -m heart_disease.train` before building)
COPY heart_disease/ ./heart_disease/
COPY api/ ./api/
COPY models/ ./models/

# Run as non-root user
RUN useradd --create-home --uid 1000 appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
