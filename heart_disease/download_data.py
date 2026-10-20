"""Download the UCI Heart Disease (Cleveland) dataset and produce the cleaned CSV.

Usage:
    python -m heart_disease.download_data
"""
import argparse
import logging
import urllib.request
from pathlib import Path

from heart_disease.config import CLEAN_DATA_PATH, DATA_URL, RAW_DATA_PATH
from heart_disease.data import clean_data, load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def download(url: str = DATA_URL, dest: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """Download the raw data file if it does not already exist."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("Raw data already present at %s (use --force to re-download)", dest)
        return dest
    logger.info("Downloading dataset from %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # nosec - trusted public URL
        content = resp.read()
    dest.write_bytes(content)
    logger.info("Saved raw data to %s (%d bytes)", dest, len(content))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean the UCI Heart Disease dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    raw_path = download(force=args.force)
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info("Raw shape: %s -> clean shape: %s", raw_df.shape, clean_df.shape)
    logger.info("Cleaned dataset written to %s", CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()
