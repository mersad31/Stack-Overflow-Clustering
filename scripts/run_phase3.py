from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stackoverflow_clustering.config import load_config, ensure_output_directories
from stackoverflow_clustering.phase3 import run_phase3_analysis
from scripts.build_phase3_outputs import figures


def run(path: str = "config.yaml"):
    config = load_config(path)
    ensure_output_directories(config)

    logging.info("Phase 3 started")
    result = run_phase3_analysis(config)

    logging.info("Building Phase 3 figures")
    figures()

    logging.info("Phase 3 complete")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run(args.config)
