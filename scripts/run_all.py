from __future__ import annotations

import argparse
import logging

from .run_phase1 import run as run_phase1
from .run_phase2 import run as run_phase2
from .run_phase3 import run as run_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every implemented project phase.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_phase1(args.config)
    run_phase2(args.config)
    run_phase3(args.config)


if __name__ == "__main__":
    main()
