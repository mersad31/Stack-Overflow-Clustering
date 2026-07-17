from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples

from .config import project_path
from .evaluation import internal_metrics, pairwise_ari_summary, proxy_label_metrics
from .io_utils import sha256_file, utc_now_iso, write_json


def _fit_partitioning(
    algorithm: str,
    matrix: np.ndarray,
    k: int,
    random_state: int,
    n_init: int,
) -> KMeans | MiniBatchKMeans:
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            tol=1e-4,
            random_state=random_state,
            algorithm="lloyd",
        )
    elif algorithm == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            batch_size=2048,
            max_iter=300,
            max_no_improvement=20,
            reassignment_ratio=0.01,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown partitioning algorithm: {algorithm}")
    model.fit(matrix)
    return model


def gap_statistic(
    matrix: np.ndarray,
    k_values: list[int],
    *,
    reference_repeats: int,
    random_state: int,
    n_init: int,
) -> pd.DataFrame:
    """Compute Tibshirani's Gap Statistic with axis-aligned uniform references."""
    rng = np.random.default_rng(random_state)
    lower = matrix.min(axis=0)
    upper = matrix.max(axis=0)
    rows: list[dict[str, float | int]] = []
    for k in k_values:
        observed = _fit_partitioning("kmeans", matrix, k, random_state, n_init)
        reference_logs: list[float] = []
        for repeat in range(reference_repeats):
            reference = rng.uniform(lower, upper, size=matrix.shape).astype(np.float32)
            model = _fit_partitioning(
                "kmeans", reference, k, random_state + 1000 + repeat, n_init
            )
            reference_logs.append(float(np.log(max(model.inertia_, np.finfo(float).tiny))))
        logs = np.asarray(reference_logs)
        rows.append(
            {
                "k": k,
                "gap": float(logs.mean() - np.log(max(observed.inertia_, np.finfo(float).tiny))),
                "gap_standard_error": float(logs.std(ddof=1) * np.sqrt(1 + 1 / reference_repeats))
                if reference_repeats > 1
                else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    next_gap = result["gap"].shift(-1)
    next_se = result["gap_standard_error"].shift(-1)
    result["one_se_rule_satisfied"] = result["gap"] >= (next_gap - next_se)
    return result


def _choose_k(scores: pd.DataFrame, gaps: pd.DataFrame) -> dict[str, int | None]:
    kneedle = KneeLocator(
        scores.loc[scores["algorithm"] == "kmeans", "k"],
        scores.loc[scores["algorithm"] == "kmeans", "inertia"],
        curve="convex",
        direction="decreasing",
    )
    eligible = gaps.loc[gaps["one_se_rule_satisfied"] & gaps["gap"].shift(-1).notna(), "k"]
    return {
        "elbow_kneedle": int(kneedle.knee) if kneedle.knee is not None else None,
        "silhouette_best": int(scores.loc[scores["silhouette"].idxmax(), "k"]),
        "gap_one_se": int(eligible.iloc[0]) if len(eligible) else int(gaps.loc[gaps["gap"].idxmax(), "k"]),
    }


def run_partitioning_analysis(config: dict[str, Any]) -> dict[str, Any]:
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    tables_dir = project_path(config, "tables_dir")
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    settings = config["phase2"]
    random_state = int(config["project"]["random_state"])
    k_values = [int(value) for value in settings["k_values"]]
    n_init = int(settings.get("partitioning_n_init", 10))
    matrix = np.load(processed_dir / "X_pca.npy", mmap_mode="r")
    rng = np.random.default_rng(random_state)
    evaluation_size = min(int(settings["evaluation_sample_size"]), len(matrix))
    evaluation_indices = np.sort(rng.choice(len(matrix), size=evaluation_size, replace=False))
    evaluation_matrix = np.asarray(matrix[evaluation_indices], dtype=np.float32)

    score_rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], KMeans | MiniBatchKMeans] = {}
    for algorithm in ("kmeans", "minibatch_kmeans"):
        for k in k_values:
            started = time.perf_counter()
            model = _fit_partitioning(algorithm, matrix, k, random_state, n_init)
            elapsed = time.perf_counter() - started
            labels = model.predict(evaluation_matrix)
            metrics = internal_metrics(
                evaluation_matrix,
                labels,
                random_state=random_state,
                dunn_max_points=int(settings.get("dunn_sample_size", 1200)),
            )
            score_rows.append(
                {
                    "algorithm": algorithm,
                    "representation": "X_pca",
                    "k": k,
                    "inertia": float(model.inertia_),
                    "iterations": int(model.n_iter_),
                    "converged": bool(model.n_iter_ < model.max_iter),
                    "runtime_seconds": elapsed,
                    **metrics,
                }
            )
            models[(algorithm, k)] = model
    scores = pd.DataFrame(score_rows)
    scores_path = tables_dir / "phase2_partitioning_scores.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")

    gap_size = min(int(settings.get("gap_sample_size", 4000)), len(matrix))
    gap_indices = np.sort(rng.choice(len(matrix), size=gap_size, replace=False))
    gaps = gap_statistic(
        np.asarray(matrix[gap_indices], dtype=np.float32),
        k_values,
        reference_repeats=int(settings.get("gap_reference_repeats", 3)),
        random_state=random_state,
        n_init=max(3, min(n_init, 5)),
    )
    gaps_path = tables_dir / "phase2_gap_statistic.csv"
    gaps.to_csv(gaps_path, index=False, encoding="utf-8-sig")
    recommendations = _choose_k(scores, gaps)

    # Bootstrap every k on a shared holdout so stability contributes to k selection,
    # rather than being reported only after a value has already been chosen.
    bootstrap_k_rows: list[dict[str, Any]] = []
    bootstrap_k_repeats = int(settings.get("bootstrap_k_repeats", 8))
    bootstrap_fit_size = min(int(settings.get("bootstrap_fit_size", 15000)), len(matrix))
    for k in k_values:
        k_labelings: list[np.ndarray] = []
        for repeat in range(bootstrap_k_repeats):
            boot = rng.choice(len(matrix), size=bootstrap_fit_size, replace=True)
            model = _fit_partitioning(
                "kmeans",
                np.asarray(matrix[boot], dtype=np.float32),
                k,
                random_state + 20_000 + 100 * k + repeat,
                max(3, min(n_init, 5)),
            )
            k_labelings.append(model.predict(evaluation_matrix))
        _, summary = pairwise_ari_summary(k_labelings)
        bootstrap_k_rows.append({"k": k, "repeats": bootstrap_k_repeats, **summary})
    bootstrap_k = pd.DataFrame(bootstrap_k_rows)
    bootstrap_k.to_csv(
        tables_dir / "phase2_bootstrap_k_stability.csv", index=False, encoding="utf-8-sig"
    )

    kmeans_scores = scores[scores["algorithm"] == "kmeans"].merge(bootstrap_k, on="k")
    kmeans_scores["rank_silhouette"] = kmeans_scores["silhouette"].rank(ascending=False)
    kmeans_scores["rank_davies_bouldin"] = kmeans_scores["davies_bouldin"].rank()
    kmeans_scores["rank_calinski_harabasz"] = kmeans_scores["calinski_harabasz"].rank(ascending=False)
    kmeans_scores["rank_bootstrap_stability"] = kmeans_scores["mean"].rank(ascending=False)
    kmeans_scores["selection_rank_sum"] = kmeans_scores[
        ["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz", "rank_bootstrap_stability"]
    ].sum(axis=1)
    kmeans_scores["elbow_candidate"] = kmeans_scores["k"] == recommendations["elbow_kneedle"]
    kmeans_scores["gap_candidate"] = kmeans_scores["k"] == recommendations["gap_one_se"]
    k_selection_path = tables_dir / "phase2_k_selection_synthesis.csv"
    kmeans_scores.to_csv(k_selection_path, index=False, encoding="utf-8-sig")
    recommendations["rank_synthesis"] = int(kmeans_scores.loc[kmeans_scores["selection_rank_sum"].idxmin(), "k"])

    # The main winner uses silhouette, with full K-Means preferred on exact ties.
    ranked = scores.sort_values(
        ["silhouette", "davies_bouldin", "algorithm"], ascending=[False, True, True]
    )
    winner = ranked.iloc[0]
    winner_algorithm = str(winner["algorithm"])
    winner_k = int(winner["k"])
    winner_model = models[(winner_algorithm, winner_k)]
    labels = winner_model.predict(matrix).astype(np.int16)
    labels_path = processed_dir / "phase2_partitioning_labels.npy"
    np.save(labels_path, labels, allow_pickle=False)
    evaluation_indices_path = processed_dir / "phase2_evaluation_indices.npy"
    np.save(evaluation_indices_path, evaluation_indices, allow_pickle=False)
    winner_model_path = model_dir / "phase2_partitioning_winner.joblib"
    joblib.dump(winner_model, winner_model_path)

    stability_labelings: list[np.ndarray] = []
    stability_rows: list[dict[str, Any]] = []
    seeds = int(settings["stability_seeds"])
    for seed in range(random_state, random_state + seeds):
        started = time.perf_counter()
        model = _fit_partitioning(winner_algorithm, matrix, winner_k, seed, n_init)
        stability_labelings.append(model.predict(evaluation_matrix))
        stability_rows.append(
            {"seed": seed, "inertia": float(model.inertia_), "runtime_seconds": time.perf_counter() - started}
        )
    stability_pairs, stability_summary = pairwise_ari_summary(stability_labelings)
    stability_pairs.to_csv(
        tables_dir / "phase2_seed_stability_pairwise_ari.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stability_rows).to_csv(
        tables_dir / "phase2_seed_stability_runs.csv", index=False, encoding="utf-8-sig"
    )

    bootstrap_repeats = int(settings["bootstrap_repeats"])
    bootstrap_fraction = float(settings.get("bootstrap_fraction", 0.7))
    bootstrap_labelings: list[np.ndarray] = []
    for repeat in range(bootstrap_repeats):
        base = rng.choice(len(matrix), size=bootstrap_fit_size, replace=False)
        boot = rng.choice(base, size=max(winner_k * 20, int(bootstrap_fraction * len(base))), replace=True)
        model = _fit_partitioning(
            winner_algorithm,
            np.asarray(matrix[boot], dtype=np.float32),
            winner_k,
            random_state + 10_000 + repeat,
            n_init,
        )
        bootstrap_labelings.append(model.predict(evaluation_matrix))
    bootstrap_pairs, bootstrap_summary = pairwise_ari_summary(bootstrap_labelings)
    bootstrap_pairs.to_csv(
        tables_dir / "phase2_bootstrap_stability_pairwise_ari.csv", index=False, encoding="utf-8-sig"
    )

    coassociation_size = min(int(settings.get("coassociation_sample_size", 400)), evaluation_size)
    evaluation_labels = labels[evaluation_indices]
    positions: list[np.ndarray] = []
    per_cluster = max(1, coassociation_size // len(np.unique(evaluation_labels)))
    for cluster in np.unique(evaluation_labels):
        available = np.flatnonzero(evaluation_labels == cluster)
        positions.append(rng.choice(available, size=min(per_cluster, len(available)), replace=False))
    coassociation_positions = np.concatenate(positions)[:coassociation_size]
    coassociation_size = len(coassociation_positions)
    coassociation = np.zeros((coassociation_size, coassociation_size), dtype=np.float32)
    for run_labels in bootstrap_labelings:
        subset = run_labels[coassociation_positions]
        coassociation += subset[:, None] == subset[None, :]
    coassociation /= max(len(bootstrap_labelings), 1)
    coassociation_path = processed_dir / "phase2_bootstrap_coassociation.npy"
    np.save(coassociation_path, coassociation, allow_pickle=False)

    metadata = pd.read_parquet(processed_dir / "respondent_metadata.parquet")
    proxies = proxy_label_metrics(labels, metadata)
    proxies.to_csv(tables_dir / "phase2_proxy_label_metrics.csv", index=False, encoding="utf-8-sig")

    sample_silhouette = silhouette_samples(evaluation_matrix, evaluation_labels)
    error_count = max(1, int(0.05 * evaluation_size))
    error_order = np.argsort(sample_silhouette)[:error_count]
    errors = metadata.iloc[evaluation_indices[error_order]].copy()
    errors.insert(1, "cluster", evaluation_labels[error_order])
    errors.insert(2, "silhouette", sample_silhouette[error_order])
    errors.to_csv(tables_dir / "phase2_low_silhouette_cases.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": utc_now_iso(),
        "representation": "X_pca",
        "rows": int(len(matrix)),
        "dimensions": int(matrix.shape[1]),
        "evaluation_sample_size": evaluation_size,
        "k_values": k_values,
        "recommendations": recommendations,
        "winner": {
            "algorithm": winner_algorithm,
            "k": winner_k,
            "silhouette": float(winner["silhouette"]),
            "davies_bouldin": float(winner["davies_bouldin"]),
            "calinski_harabasz": float(winner["calinski_harabasz"]),
            "dunn_approx": float(winner["dunn_approx"]),
        },
        "seed_stability": {"runs": seeds, **stability_summary},
        "bootstrap_stability": {"runs": bootstrap_repeats, **bootstrap_summary},
        "model_sha256": sha256_file(winner_model_path),
        "paths": {
            "scores": str(scores_path),
            "gap_statistic": str(gaps_path),
            "k_selection_synthesis": str(k_selection_path),
            "labels": str(labels_path),
            "evaluation_indices": str(evaluation_indices_path),
            "coassociation": str(coassociation_path),
            "model": str(winner_model_path),
        },
    }
    write_json(manifest, artifacts_dir / "phase2_partitioning_manifest.json")
    return manifest
