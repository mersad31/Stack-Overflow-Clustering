from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.config import ensure_output_directories, load_config
from stackoverflow_clustering.io_utils import utc_now_iso, write_json
from stackoverflow_clustering.partitioning import run_partitioning_analysis
from stackoverflow_clustering.phase2_visualization import create_partitioning_figures
from stackoverflow_clustering.model_families import run_model_family_analysis
from stackoverflow_clustering.family_visualization import create_model_family_figures


def run(config_path: str | Path = "config.yaml") -> dict:
    config = load_config(config_path)
    ensure_output_directories(config)
    started = time.perf_counter()
    logging.info("Phase 2A partitioning analysis started")
    partitioning = run_partitioning_analysis(config)
    create_partitioning_figures(config, partitioning)
    logging.info("Phase 2B model-family comparison started")
    model_families = run_model_family_analysis(config)
    create_model_family_figures(config, model_families)
    elapsed = time.perf_counter() - started
    result = {
        "phase": 2,
        "status": "completed",
        "scope": "partitioning, hierarchical, density, model-based, and spectral comparison",
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "partitioning": partitioning,
        "model_families": model_families,
    }
    artifacts = Path(config["_project_root"]) / config["paths"]["artifacts_dir"]
    write_json(result, artifacts / "phase2_run_summary.json")
    logging.info("Phase 2 completed in %.1f seconds", elapsed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 clustering analysis.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
