from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.bonus import paired_sign_flip_pvalue, percentile_interval


def test_percentile_interval_contains_central_values() -> None:
    lower, upper = percentile_interval(range(100))
    assert 1.0 < lower < 5.0
    assert 94.0 < upper < 98.0


def test_sign_flip_detects_consistent_paired_improvement() -> None:
    differences = np.linspace(0.1, 0.3, 80)
    pvalue = paired_sign_flip_pvalue(differences, repeats=999, random_state=42)
    assert pvalue <= 0.01


def test_sign_flip_is_reproducible() -> None:
    differences = np.array([-0.2, 0.1, 0.3, -0.1, 0.05])
    first = paired_sign_flip_pvalue(differences, repeats=199, random_state=7)
    second = paired_sign_flip_pvalue(differences, repeats=199, random_state=7)
    assert first == second
