from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def approximate_dunn_index(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 1200,
    random_state: int = 42,
) -> float:
    """Estimate Dunn's index on a deterministic stratified subsample.

    The exact statistic requires a quadratic distance matrix.  This version
    preserves every cluster while bounding the calculation for pipeline use.
    """
    matrix = np.asarray(matrix)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    per_cluster = max(2, max_points // len(unique))
    selected: list[np.ndarray] = []
    for label in unique:
        indices = np.flatnonzero(labels == label)
        take = min(len(indices), per_cluster)
        selected.append(rng.choice(indices, size=take, replace=False))
    sample_indices = np.concatenate(selected)
    sample = matrix[sample_indices]
    sample_labels = labels[sample_indices]

    max_diameter = 0.0
    for label in unique:
        points = sample[sample_labels == label]
        if len(points) > 1:
            max_diameter = max(max_diameter, float(pdist(points).max(initial=0.0)))
    if max_diameter <= 0:
        return float("nan")

    min_separation = float("inf")
    for left, right in combinations(unique, 2):
        distance = cdist(sample[sample_labels == left], sample[sample_labels == right])
        if distance.size:
            min_separation = min(min_separation, float(distance.min()))
    return min_separation / max_diameter


def internal_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    dunn_max_points: int = 1200,
) -> dict[str, float]:
    """Return the common internal metrics for a non-degenerate partition."""
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    if len(clusters) < 2 or len(clusters) >= len(labels):
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "dunn_approx": float("nan"),
        }
    return {
        "silhouette": float(silhouette_score(matrix, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "dunn_approx": float(
            approximate_dunn_index(
                matrix,
                labels,
                max_points=dunn_max_points,
                random_state=random_state,
            )
        ),
    }


def pairwise_ari_summary(labelings: list[np.ndarray]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = [
        {"run_a": left, "run_b": right, "ari": adjusted_rand_score(labelings[left], labelings[right])}
        for left, right in combinations(range(len(labelings)), 2)
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"mean": float("nan"), "std": float("nan"), "min": float("nan")}
    return frame, {
        "mean": float(frame["ari"].mean()),
        "std": float(frame["ari"].std(ddof=0)),
        "min": float(frame["ari"].min()),
    }


def proxy_label_metrics(cluster_labels: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """Measure association with interpretive labels; these are not ground truth."""
    rows: list[dict[str, Any]] = []
    for column in ("DevType", "EdLevel", "Employment", "RemoteWork", "AISelect"):
        values = metadata[column].fillna("Missing").astype(str).to_numpy()
        rows.append(
            {
                "proxy": column,
                "normalized_mutual_information": normalized_mutual_info_score(values, cluster_labels),
                "categories": int(pd.Series(values).nunique()),
                "interpretation": "association only; proxy is not ground truth",
            }
        )
    return pd.DataFrame(rows)
