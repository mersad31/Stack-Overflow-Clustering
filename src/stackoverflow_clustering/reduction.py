from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from .config import project_path
from .io_utils import sha256_file, utc_now_iso, write_json


def run_dimensionality_reduction(config: dict[str, Any]) -> dict[str, Any]:
    """Fit PCA, sparse SVD, and a sampled UMAP diagnostic embedding."""
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    tables_dir = project_path(config, "tables_dir")
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    random_state = int(config["project"]["random_state"])
    requested_components = int(config["features"]["pca_components"])
    x_full = sparse.load_npz(processed_dir / "X_full.npz").astype(np.float32)
    x_tech = sparse.load_npz(processed_dir / "X_tech_stack.npz").astype(np.float32)
    n_components = min(requested_components, x_full.shape[0] - 1, x_full.shape[1])

    dense_full = x_full.toarray().astype(np.float32, copy=False)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    x_pca = pca.fit_transform(dense_full).astype(np.float32)
    del dense_full
    pca_path = processed_dir / "X_pca.npy"
    np.save(pca_path, x_pca, allow_pickle=False)
    pca_model_path = model_dir / "pca.joblib"
    joblib.dump(pca, pca_model_path)

    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(tables_dir / "pca_explained_variance.csv", index=False)

    svd_components = min(n_components, x_tech.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    x_svd = svd.fit_transform(x_tech).astype(np.float32)
    svd_path = processed_dir / "X_tech_svd.npy"
    np.save(svd_path, x_svd, allow_pickle=False)
    svd_model_path = model_dir / "tech_truncated_svd.joblib"
    joblib.dump(svd, svd_model_path)
    pd.DataFrame(
        {
            "component": np.arange(1, svd_components + 1),
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(svd.explained_variance_ratio_),
        }
    ).to_csv(tables_dir / "tech_svd_explained_variance.csv", index=False)

    rng = np.random.default_rng(random_state)
    umap_size = min(int(config["features"]["umap_sample_size"]), len(x_pca))
    umap_indices = np.sort(rng.choice(len(x_pca), size=umap_size, replace=False))
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(config["features"]["umap_neighbors"]),
            min_dist=float(config["features"]["umap_min_dist"]),
            metric="euclidean",
            random_state=random_state,
            n_jobs=1,
        )
        x_umap = reducer.fit_transform(x_pca[umap_indices]).astype(np.float32)
        umap_model_path = model_dir / "umap_sample.joblib"
        joblib.dump(reducer, umap_model_path)
        umap_status = "completed"
    except Exception as exc:  # Keep linear results reproducible if optional UMAP fails.
        x_umap = np.empty((0, 2), dtype=np.float32)
        umap_model_path = model_dir / "umap_sample.joblib"
        umap_status = f"failed: {type(exc).__name__}: {exc}"

    umap_path = processed_dir / "X_umap_sample.npy"
    umap_indices_path = processed_dir / "umap_sample_indices.npy"
    np.save(umap_path, x_umap, allow_pickle=False)
    np.save(umap_indices_path, umap_indices, allow_pickle=False)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "pca_shape": list(map(int, x_pca.shape)),
        "pca_cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "pca_model_sha256": sha256_file(pca_model_path),
        "svd_shape": list(map(int, x_svd.shape)),
        "svd_cumulative_variance": float(np.cumsum(svd.explained_variance_ratio_)[-1]),
        "umap_sample_size": int(len(x_umap)),
        "umap_status": umap_status,
        "paths": {
            "X_pca": str(pca_path),
            "X_tech_svd": str(svd_path),
            "X_umap_sample": str(umap_path),
            "umap_sample_indices": str(umap_indices_path),
            "pca_model": str(pca_model_path),
            "svd_model": str(svd_model_path),
            "umap_model": str(umap_model_path) if umap_status == "completed" else None,
        },
    }
    write_json(manifest, artifacts_dir / "reduction_manifest.json")
    return manifest

