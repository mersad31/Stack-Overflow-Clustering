from __future__ import annotations

import time
from itertools import combinations
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

from .config import project_path
from .evaluation import internal_metrics
from .io_utils import utc_now_iso, write_json


def _balance(labels: np.ndarray, *, ignore_noise: bool = False) -> dict[str, float | int]:
    labels = np.asarray(labels)
    retained = labels >= 0 if ignore_noise else np.ones(len(labels), dtype=bool)
    _, counts = np.unique(labels[retained], return_counts=True)
    denominator = max(int(retained.sum()), 1)
    if not len(counts):
        return {"minimum_cluster_size": 0, "minimum_cluster_fraction": 0.0, "largest_cluster_fraction": 0.0}
    return {
        "minimum_cluster_size": int(counts.min()),
        "minimum_cluster_fraction": float(counts.min() / denominator),
        "largest_cluster_fraction": float(counts.max() / denominator),
    }


def _metrics_with_noise(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
    dunn_max_points: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    retained = labels >= 0
    clusters = np.unique(labels[retained])
    result: dict[str, Any] = {
        "clusters_found": int(len(clusters)),
        "noise_fraction": float(1 - retained.mean()),
        "evaluated_rows": int(retained.sum()),
        **_balance(labels, ignore_noise=True),
    }
    if len(clusters) >= 2 and retained.sum() > len(clusters):
        result.update(
            internal_metrics(
                matrix[retained],
                labels[retained],
                random_state=random_state,
                dunn_max_points=dunn_max_points,
            )
        )
    else:
        result.update({key: float("nan") for key in ("silhouette", "davies_bouldin", "calinski_harabasz", "dunn_approx")})
    return result


def _hierarchical(
    matrix: np.ndarray,
    k_values: list[int],
    random_state: int,
    dunn_max_points: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_labels: dict[str, np.ndarray] = {}
    linkages: dict[str, np.ndarray] = {}
    cuts: list[dict[str, Any]] = []
    distances = pdist(matrix, metric="euclidean")
    for method in ("single", "complete", "average", "ward"):
        started = time.perf_counter()
        tree = linkage(distances, method=method)
        fit_seconds = time.perf_counter() - started
        linkages[method] = tree
        correlation = float(cophenet(tree, distances)[0])
        method_rows: list[dict[str, Any]] = []
        method_labelings: dict[int, np.ndarray] = {}
        for k in k_values:
            labels = fcluster(tree, t=k, criterion="maxclust").astype(np.int16) - 1
            method_labelings[k] = labels
            metrics = internal_metrics(
                matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points
            )
            row = {
                "family": "hierarchical",
                "algorithm": f"agglomerative_{method}",
                "k_requested": k,
                "clusters_found": int(np.unique(labels).size),
                "noise_fraction": 0.0,
                "runtime_seconds": fit_seconds,
                "cophenetic_correlation": correlation,
                **_balance(labels),
                **metrics,
            }
            rows.append(row)
            method_rows.append(row)
        selected = max(method_rows, key=lambda item: item["silhouette"])
        selected_k = int(selected["k_requested"])
        best_labels[method] = method_labelings[selected_k]
        # A height threshold is a genuinely different cut specification and is
        # recorded beside maxclust even when both happen to return the same cut.
        threshold = float((tree[-selected_k, 2] + tree[-selected_k + 1, 2]) / 2)
        distance_labels = fcluster(tree, t=threshold, criterion="distance").astype(np.int16) - 1
        cuts.extend(
            [
                {"linkage": method, "strategy": "maxclust", "parameter": selected_k, "clusters": int(np.unique(best_labels[method]).size)},
                {"linkage": method, "strategy": "distance", "parameter": threshold, "clusters": int(np.unique(distance_labels).size)},
            ]
        )
    return rows, best_labels, linkages, pd.DataFrame(cuts)


def _density(
    matrix: np.ndarray,
    random_state: int,
    dunn_max_points: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    labelings: dict[str, np.ndarray] = {}
    neighbors = 20
    distances, _ = NearestNeighbors(n_neighbors=neighbors).fit(matrix).kneighbors(matrix)
    kth = np.sort(distances[:, -1])
    diagnostics = pd.DataFrame({"rank": np.arange(1, len(kth) + 1), "k_distance": kth})
    condensed = pd.DataFrame()
    reachability = pd.DataFrame()

    for quantile in (0.85, 0.90, 0.95):
        eps = float(np.quantile(kth, quantile))
        started = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=neighbors, metric="euclidean", n_jobs=-1).fit_predict(matrix)
        name = f"dbscan_q{int(quantile * 100)}"
        labelings[name] = labels.astype(np.int16)
        rows.append(
            {
                "family": "density",
                "algorithm": name,
                "k_requested": np.nan,
                "runtime_seconds": time.perf_counter() - started,
                "parameter": f"eps={eps:.6g};min_samples={neighbors}",
                **_metrics_with_noise(matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points),
            }
        )

    for min_cluster_size in (30, 75, 150):
        started = time.perf_counter()
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, metric="euclidean")
        labels = model.fit_predict(matrix)
        if condensed.empty:
            condensed = model.condensed_tree_.to_pandas()
        name = f"hdbscan_mcs{min_cluster_size}"
        labelings[name] = labels.astype(np.int16)
        rows.append(
            {
                "family": "density",
                "algorithm": name,
                "k_requested": np.nan,
                "runtime_seconds": time.perf_counter() - started,
                "parameter": f"min_cluster_size={min_cluster_size};min_samples=10",
                **_metrics_with_noise(matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points),
            }
        )

    for min_samples in (10, 30):
        started = time.perf_counter()
        model = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=0.02, n_jobs=-1)
        labels = model.fit_predict(matrix)
        if reachability.empty:
            ordering = model.ordering_
            reachability = pd.DataFrame(
                {
                    "order": np.arange(len(ordering)),
                    "sample_position": ordering,
                    "reachability": model.reachability_[ordering],
                    "cluster": labels[ordering],
                }
            )
        name = f"optics_ms{min_samples}"
        labelings[name] = labels.astype(np.int16)
        rows.append(
            {
                "family": "density",
                "algorithm": name,
                "k_requested": np.nan,
                "runtime_seconds": time.perf_counter() - started,
                "parameter": f"min_samples={min_samples};xi=0.05;min_cluster_size=0.02",
                **_metrics_with_noise(matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points),
            }
        )
    return rows, labelings, diagnostics, condensed, reachability


def _gmm(
    matrix: np.ndarray,
    k_values: list[int],
    random_state: int,
    dunn_max_points: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, GaussianMixture]]:
    rows: list[dict[str, Any]] = []
    labelings: dict[str, np.ndarray] = {}
    models: dict[str, GaussianMixture] = {}
    for covariance in ("spherical", "diag", "tied", "full"):
        for k in k_values:
            started = time.perf_counter()
            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance,
                n_init=3,
                max_iter=300,
                reg_covar=1e-6,
                random_state=random_state,
            ).fit(matrix)
            labels = model.predict(matrix).astype(np.int16)
            probabilities = model.predict_proba(matrix)
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1)
            name = f"gmm_{covariance}_k{k}"
            labelings[name] = labels
            models[name] = model
            rows.append(
                {
                    "family": "model_based",
                    "algorithm": f"gmm_{covariance}",
                    "k_requested": k,
                    "clusters_found": int(np.unique(labels).size),
                    "noise_fraction": 0.0,
                    "runtime_seconds": time.perf_counter() - started,
                    "aic": float(model.aic(matrix)),
                    "bic": float(model.bic(matrix)),
                    "converged": bool(model.converged_),
                    "iterations": int(model.n_iter_),
                    "lower_bound": float(model.lower_bound_),
                    "mean_assignment_entropy": float(entropy.mean()),
                    **_balance(labels),
                    **internal_metrics(matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points),
                }
            )
    return rows, labelings, models


def _spectral(
    matrix: np.ndarray,
    k_values: list[int],
    random_state: int,
    dunn_max_points: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    labelings: dict[str, np.ndarray] = {}
    neighbors = min(20, len(matrix) - 1)
    for k in k_values:
        started = time.perf_counter()
        model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            n_init=10,
            random_state=random_state,
            eigen_solver="arpack",
            n_jobs=-1,
        )
        labels = model.fit_predict(matrix).astype(np.int16)
        name = f"spectral_k{k}"
        labelings[name] = labels
        rows.append(
            {
                "family": "spectral",
                "algorithm": "spectral_nearest_neighbors",
                "k_requested": k,
                "clusters_found": int(np.unique(labels).size),
                "noise_fraction": 0.0,
                "runtime_seconds": time.perf_counter() - started,
                "parameter": f"n_neighbors={neighbors};assign_labels=kmeans",
                **_balance(labels),
                **internal_metrics(matrix, labels, random_state=random_state, dunn_max_points=dunn_max_points),
            }
        )
    return rows, labelings


def run_model_family_analysis(config: dict[str, Any]) -> dict[str, Any]:
    processed = project_path(config, "processed_dir")
    artifacts = project_path(config, "artifacts_dir")
    tables = project_path(config, "tables_dir")
    models_dir = artifacts / "models"
    settings = config["phase2"]
    random_state = int(config["project"]["random_state"])
    source = np.load(processed / "X_pca.npy", mmap_mode="r")
    sample_size = min(int(settings.get("family_sample_size", 3000)), len(source))
    rng = np.random.default_rng(random_state + 200)
    indices = np.sort(rng.choice(len(source), size=sample_size, replace=False))
    matrix = np.asarray(source[indices], dtype=np.float32)
    k_values = [int(k) for k in settings.get("family_k_values", [2, 3, 4, 5, 6, 7, 8])]
    dunn = int(settings.get("dunn_sample_size", 1200))

    hierarchical_rows, hierarchical_labels, trees, cuts = _hierarchical(matrix, k_values, random_state, dunn)
    density_rows, density_labels, kdist, condensed, reachability = _density(matrix, random_state, dunn)
    gmm_rows, gmm_labels, gmm_models = _gmm(matrix, k_values, random_state, dunn)
    spectral_rows, spectral_labels = _spectral(matrix, k_values, random_state, dunn)

    # Add K-Means on the exact same sample to keep cross-family comparisons fair.
    comparison_rows: list[dict[str, Any]] = []
    comparison_labels: dict[str, np.ndarray] = {}
    for k in k_values:
        started = time.perf_counter()
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(matrix)
        labels = model.labels_.astype(np.int16)
        comparison_labels[f"kmeans_k{k}"] = labels
        comparison_rows.append(
            {
                "family": "partitioning",
                "algorithm": "kmeans",
                "k_requested": k,
                "clusters_found": k,
                "noise_fraction": 0.0,
                "runtime_seconds": time.perf_counter() - started,
                "inertia": float(model.inertia_),
                **_balance(labels),
                **internal_metrics(matrix, labels, random_state=random_state, dunn_max_points=dunn),
            }
        )

    scores = pd.DataFrame([*comparison_rows, *hierarchical_rows, *density_rows, *gmm_rows, *spectral_rows])
    scores_path = tables / "phase2_model_family_scores.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    cuts.to_csv(tables / "phase2_hierarchical_cut_strategies.csv", index=False, encoding="utf-8-sig")
    kdist.to_csv(tables / "phase2_k_distance.csv", index=False, encoding="utf-8-sig")
    condensed.to_csv(tables / "phase2_hdbscan_condensed_tree.csv", index=False, encoding="utf-8-sig")
    reachability.to_csv(tables / "phase2_optics_reachability.csv", index=False, encoding="utf-8-sig")

    candidates: dict[str, np.ndarray] = {}
    candidates["kmeans"] = comparison_labels[f"kmeans_k{int(pd.DataFrame(comparison_rows).loc[pd.DataFrame(comparison_rows)['silhouette'].idxmax(), 'k_requested'])}"]
    min_fraction = float(settings.get("minimum_cluster_fraction", 0.02))
    eligible = scores[(scores["minimum_cluster_fraction"] >= min_fraction) & scores["silhouette"].notna()]
    best_h = eligible[eligible["family"] == "hierarchical"].sort_values("silhouette", ascending=False).iloc[0]
    h_method = str(best_h["algorithm"]).replace("agglomerative_", "")
    candidates[str(best_h["algorithm"])] = hierarchical_labels[h_method]
    valid_density = eligible[(eligible["family"] == "density") & (eligible["noise_fraction"] <= 0.5)].copy()
    valid_density["selection_score"] = valid_density["silhouette"] * (1 - valid_density["noise_fraction"])
    if len(valid_density):
        best_d = valid_density.sort_values("selection_score", ascending=False).iloc[0]
        candidates[str(best_d["algorithm"])] = density_labels[str(best_d["algorithm"])]
    best_g = scores[scores["family"] == "model_based"].sort_values("bic").iloc[0]
    g_name = f"{best_g['algorithm']}_k{int(best_g['k_requested'])}"
    candidates[g_name] = gmm_labels[g_name]
    best_s = eligible[eligible["family"] == "spectral"].sort_values("silhouette", ascending=False).iloc[0]
    s_name = f"spectral_k{int(best_s['k_requested'])}"
    candidates[s_name] = spectral_labels[s_name]

    agreement_rows = [
        {"candidate_a": left, "candidate_b": right, "adjusted_rand_index": adjusted_rand_score(candidates[left], candidates[right])}
        for left, right in combinations(candidates, 2)
    ]
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(tables / "phase2_algorithm_agreement.csv", index=False, encoding="utf-8-sig")
    labels_path = processed / "phase2_family_candidate_labels.npz"
    np.savez_compressed(labels_path, sample_indices=indices, **candidates)
    trees_path = processed / "phase2_hierarchical_linkages.npz"
    np.savez_compressed(trees_path, **trees)
    best_gmm_path = models_dir / "phase2_gmm_bic_winner.joblib"
    joblib.dump(gmm_models[g_name], best_gmm_path)

    family_best: dict[str, Any] = {}
    for family, group in scores.groupby("family"):
        valid = group[(group["silhouette"].notna()) & (group["minimum_cluster_fraction"] >= min_fraction)]
        if len(valid):
            row = valid.sort_values("silhouette", ascending=False).iloc[0]
            family_best[family] = {
                "algorithm": str(row["algorithm"]),
                "k": None if pd.isna(row["k_requested"]) else int(row["k_requested"]),
                "clusters_found": int(row["clusters_found"]),
                "silhouette": float(row["silhouette"]),
                "noise_fraction": float(row["noise_fraction"]),
            }
    manifest = {
        "created_at_utc": utc_now_iso(),
        "representation": "X_pca",
        "sample_size": sample_size,
        "sample_seed": random_state + 200,
        "k_values": k_values,
        "configurations_evaluated": int(len(scores)),
        "minimum_cluster_fraction_for_selection": min_fraction,
        "family_best_by_silhouette": family_best,
        "gmm_bic_winner": g_name,
        "agreement_candidates": list(candidates),
        "paths": {"scores": str(scores_path), "labels": str(labels_path), "linkages": str(trees_path), "gmm_model": str(best_gmm_path)},
    }
    write_json(manifest, artifacts / "phase2_model_families_manifest.json")
    return manifest
