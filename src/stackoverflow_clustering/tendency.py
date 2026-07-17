from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .config import project_path
from .io_utils import utc_now_iso, write_json


def _dense_rows(matrix: np.ndarray | sparse.spmatrix, indices: np.ndarray) -> np.ndarray:
    selected = matrix[indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float64)


def hopkins_statistic(
    matrix: np.ndarray | sparse.spmatrix,
    sample_size: int,
    random_state: int,
) -> float:
    """Compute Hopkins H using nearest-neighbour distances.

    H near 0.5 indicates spatial randomness; H above roughly 0.7 indicates
    appreciable clustering tendency. Uniform reference points use the observed
    axis-aligned bounding box of the representation.
    """
    rng = np.random.default_rng(random_state)
    n_rows, n_features = matrix.shape
    m = min(sample_size, max(2, n_rows - 1))
    sample_indices = rng.choice(n_rows, size=m, replace=False)
    real_sample = _dense_rows(matrix, sample_indices)

    if sparse.issparse(matrix):
        mins = np.asarray(matrix.min(axis=0).toarray()).ravel()
        maxs = np.asarray(matrix.max(axis=0).toarray()).ravel()
    else:
        array = np.asarray(matrix)
        mins = np.nanmin(array, axis=0)
        maxs = np.nanmax(array, axis=0)
    constant = maxs <= mins
    maxs = np.where(constant, mins + 1e-12, maxs)
    uniform = rng.uniform(mins, maxs, size=(m, n_features))

    model = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm="brute", n_jobs=1)
    model.fit(matrix)
    batch_size = 64
    real_parts: list[np.ndarray] = []
    uniform_parts: list[np.ndarray] = []
    for start in range(0, m, batch_size):
        stop = min(start + batch_size, m)
        real_parts.append(
            model.kneighbors(real_sample[start:stop], n_neighbors=2, return_distance=True)[0][
                :, 1
            ]
        )
        uniform_parts.append(
            model.kneighbors(uniform[start:stop], n_neighbors=1, return_distance=True)[0][
                :, 0
            ]
        )
    real_distances = np.concatenate(real_parts)
    uniform_distances = np.concatenate(uniform_parts)
    denominator = uniform_distances.sum() + real_distances.sum()
    return float(uniform_distances.sum() / denominator) if denominator > 0 else 0.5


def vat_order(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a Prim-style VAT permutation of a precomputed dissimilarity matrix."""
    n = distance_matrix.shape[0]
    if n < 2:
        return np.arange(n)
    start = int(np.argmax(distance_matrix.max(axis=1)))
    selected = [start]
    remaining = np.ones(n, dtype=bool)
    remaining[start] = False
    min_to_selected = distance_matrix[start].copy()
    for _ in range(n - 1):
        candidates = np.flatnonzero(remaining)
        next_index = int(candidates[np.argmin(min_to_selected[candidates])])
        selected.append(next_index)
        remaining[next_index] = False
        min_to_selected = np.minimum(min_to_selected, distance_matrix[next_index])
    return np.asarray(selected, dtype=np.int32)


def run_tendency_assessment(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    tables_dir = project_path(config, "tables_dir")
    random_state = int(config["project"]["random_state"])
    sample_size = int(config["tendency"]["hopkins_sample_size"])
    repeats = int(config["tendency"]["hopkins_repeats"])

    representations: dict[str, np.ndarray | sparse.spmatrix] = {
        "X_full_standard": sparse.load_npz(processed_dir / "X_full_standard.npz"),
        "X_full_robust": sparse.load_npz(processed_dir / "X_full_robust.npz"),
        "X_full_minmax": sparse.load_npz(processed_dir / "X_full_minmax.npz"),
        "X_tech_stack": sparse.load_npz(processed_dir / "X_tech_stack.npz"),
        "X_pca": np.load(processed_dir / "X_pca.npy", mmap_mode="r"),
    }
    rows: list[dict[str, Any]] = []
    for name, matrix in representations.items():
        for repeat in range(repeats):
            value = hopkins_statistic(matrix, sample_size, random_state + repeat)
            rows.append({"representation": name, "repeat": repeat + 1, "hopkins": value})
    raw = pd.DataFrame(rows)
    raw.to_csv(tables_dir / "hopkins_repeats.csv", index=False)
    summary = (
        raw.groupby("representation")["hopkins"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(tables_dir / "hopkins_summary.csv", index=False)

    x_pca = representations["X_pca"]
    rng = np.random.default_rng(random_state)
    vat_size = min(int(config["tendency"]["vat_sample_size"]), x_pca.shape[0])
    vat_indices = np.sort(rng.choice(x_pca.shape[0], size=vat_size, replace=False))
    vat_input = np.asarray(x_pca[vat_indices], dtype=np.float32)
    distances = pairwise_distances(vat_input, metric="euclidean", n_jobs=1)
    order = vat_order(distances)
    reordered = distances[np.ix_(order, order)].astype(np.float32)
    vat_matrix_path = processed_dir / "vat_reordered_distances.npy"
    np.save(vat_matrix_path, reordered, allow_pickle=False)
    np.save(processed_dir / "vat_sample_indices.npy", vat_indices, allow_pickle=False)
    np.save(processed_dir / "vat_order.npy", order, allow_pickle=False)

    result = {
        "created_at_utc": utc_now_iso(),
        "hopkins_sample_size": sample_size,
        "hopkins_repeats": repeats,
        "hopkins_summary": summary.to_dict(orient="records"),
        "vat_sample_size": vat_size,
        "vat_matrix_path": str(vat_matrix_path),
    }
    write_json(result, artifacts_dir / "tendency_results.json")
    return result
