from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.reporting import build_phase1_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    output = build_phase1_report(load_config(args.config))
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()

