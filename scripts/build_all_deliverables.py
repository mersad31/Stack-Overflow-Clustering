from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.build_final_report import build as build_final_report
from scripts.build_phase1_notebook import build as build_phase1_notebook
from scripts.build_phase2_notebook import build as build_phase2_notebook
from scripts.build_phase2_report import build as build_phase2_report
from scripts.build_phase3_outputs import figures as build_phase3_figures
from scripts.build_phase3_outputs import notebook as build_phase3_notebook
from scripts.build_phase3_report import build as build_phase3_report
from scripts.build_topic_report import build as build_topic_report
from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.reporting import build_phase1_report


def main() -> None:
    config = load_config("config.yaml")
    build_phase1_notebook((ROOT / "notebooks/01_phase1_eda_executed.ipynb").resolve())
    build_phase2_notebook()
    build_phase3_figures()
    build_phase3_notebook()
    build_phase1_report(config)
    build_phase2_report()
    build_phase3_report()
    build_final_report()
    build_topic_report()
    print("Built all notebooks, figures, and DOCX reports.")
    print("Export refreshed PDFs with: powershell -ExecutionPolicy Bypass -File scripts/export_reports_pdf.ps1")


if __name__ == "__main__":
    main()
