from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.evaluation import (
    approximate_dunn_index,
    internal_metrics,
    pairwise_ari_summary,
)
from stackoverflow_clustering.partitioning import gap_statistic


def test_internal_metrics_reward_separated_blobs() -> None:
    matrix, labels = make_blobs(n_samples=300, centers=3, cluster_std=0.25, random_state=42)
    metrics = internal_metrics(matrix, labels, dunn_max_points=300)
    assert metrics["silhouette"] > 0.8
    assert metrics["davies_bouldin"] < 0.3
    assert metrics["calinski_harabasz"] > 100
    assert approximate_dunn_index(matrix, labels, max_points=300) > 0.5


def test_pairwise_ari_is_permutation_invariant() -> None:
    first = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    frame, summary = pairwise_ari_summary([first, permuted])
    assert frame.iloc[0]["ari"] == 1.0
    assert summary["mean"] == 1.0


def test_gap_statistic_has_one_row_per_k() -> None:
    matrix, _ = make_blobs(n_samples=180, centers=3, n_features=4, random_state=42)
    result = gap_statistic(
        matrix.astype(np.float32),
        [2, 3, 4],
        reference_repeats=2,
        random_state=42,
        n_init=2,
    )
    assert result["k"].tolist() == [2, 3, 4]
    assert np.isfinite(result["gap"]).all()
