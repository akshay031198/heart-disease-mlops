#!/usr/bin/env bash
# End-to-end local pipeline (Linux / macOS / Git-Bash). Run from the project root.
set -euo pipefail
python -m heart_disease.download_data
python -m heart_disease.eda
flake8 heart_disease api tests scripts
pytest
python -m heart_disease.train
python -m heart_disease.predict --input samples/sample_request.json
echo "Pipeline finished. Start the API with: uvicorn api.main:app --host 0.0.0.0 --port 8000"
