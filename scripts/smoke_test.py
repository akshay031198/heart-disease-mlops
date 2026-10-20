"""Smoke-test a running API instance (local uvicorn, Docker or Kubernetes).

Usage:
    python scripts/smoke_test.py                       # defaults to http://localhost:8000
    python scripts/smoke_test.py --url http://localhost:80
    python scripts/smoke_test.py --requests 50         # generate traffic for Grafana
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def call(url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def wait_for_health(base, retries=30):
    for _ in range(retries):
        try:
            status, body = call(f"{base}/health")
            if status == 200:
                return body
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(f"API at {base} did not become healthy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1, help="number of random /predict calls")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print("Health:", wait_for_health(base))
    sample = json.loads((ROOT / "samples" / "sample_request.json").read_text())
    status, body = call(f"{base}/predict", sample)
    print("Single prediction:", json.dumps(body, indent=2))
    assert status == 200 and body["prediction"] in (0, 1) and 0 <= body["confidence"] <= 1

    batch = json.loads((ROOT / "samples" / "sample_batch.json").read_text())
    status, body = call(f"{base}/predict/batch", batch)
    print(f"Batch prediction: {body['count']} results")
    assert status == 200

    for _ in range(max(args.requests - 1, 0)):
        rec = dict(sample)
        rec.update(age=random.randint(30, 77), chol=random.randint(150, 400),
                   thalach=random.randint(90, 200), cp=random.randint(1, 4),
                   oldpeak=round(random.uniform(0, 5), 1), exang=random.randint(0, 1))
        call(f"{base}/predict", rec)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # non-zero exit fails the CI job
        print(f"SMOKE TEST FAILED: {exc}")
        sys.exit(1)
