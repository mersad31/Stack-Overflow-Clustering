from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.phase3 import _psi, mean_absolute_shap_values


def test_psi_is_zero_for_identical_distributions() -> None:
    values = np.linspace(-2, 2, 500)
    assert _psi(values, values) == 0.0


def test_psi_detects_large_distribution_shift() -> None:
    reference = np.linspace(0, 1, 1000)
    current = np.linspace(2, 3, 1000)
    assert _psi(reference, current) > 0.2


def test_shap_reduction_supports_samples_features_classes() -> None:
    values = np.zeros((4, 3, 2))
    values[:, 1, :] = 2.0
    result = mean_absolute_shap_values(values, 3)
    assert result.tolist() == [0.0, 2.0, 0.0]


def test_shap_reduction_rejects_unknown_layout() -> None:
    try:
        mean_absolute_shap_values(np.zeros((2, 2, 2, 2)), 2)
    except ValueError as exc:
        assert "Unexpected SHAP shape" in str(exc)
    else:
        raise AssertionError("Unknown SHAP layout should fail")
