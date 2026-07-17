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

from stackoverflow_clustering.config import ensure_output_directories, load_config, project_path
from stackoverflow_clustering.data import clean_selected, ingest_selected
from stackoverflow_clustering.features import build_feature_matrices
from stackoverflow_clustering.io_utils import utc_now_iso, write_json
from stackoverflow_clustering.reduction import run_dimensionality_reduction
from stackoverflow_clustering.tendency import run_tendency_assessment
from stackoverflow_clustering.visualization import create_phase1_figures, create_phase1_tables


def run(config_path: str | Path = "config.yaml") -> dict:
    config = load_config(config_path)
    ensure_output_directories(config)
    started = time.perf_counter()
    logging.info("Phase 1 started")

    raw_frame, ingestion = ingest_selected(config)
    logging.info("Ingested %s rows", f"{len(raw_frame):,}")

    clean_frame, decisions, cleaning = clean_selected(raw_frame, config)
    logging.info("Clean cohort: %s rows; %s decisions", f"{len(clean_frame):,}", len(decisions))

    features = build_feature_matrices(clean_frame, config)
    logging.info("Feature matrices created: %s", features["representations"])

    reduction = run_dimensionality_reduction(config)
    logging.info("PCA/UMAP completed: %s", reduction["umap_status"])

    tendency = run_tendency_assessment(config)
    logging.info("Hopkins and VAT completed")

    create_phase1_tables(clean_frame, config)
    create_phase1_figures(raw_frame, clean_frame, config)
    logging.info("Phase 1 report tables and figures created")

    elapsed = time.perf_counter() - started
    result = {
        "phase": 1,
        "status": "completed",
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "ingestion": ingestion,
        "cleaning": cleaning,
        "features": features,
        "reduction": reduction,
        "tendency": tendency,
    }
    write_json(result, project_path(config, "artifacts_dir") / "phase1_run_summary.json")
    logging.info("Phase 1 completed in %.1f seconds", elapsed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 end to end.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args.config)


if __name__ == "__main__":
    main()

