from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import MiniBatchNMF
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

from .config import project_path
from .io_utils import sha256_file, utc_now_iso, write_json


def percentile_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided percentile interval for finite bootstrap estimates."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    *,
    repeats: int,
    random_state: int,
) -> float:
    """Monte-Carlo paired randomisation test for a mean metric difference."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return float("nan")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(random_state)
    extreme = 0
    # Chunking avoids allocating repeats x observations for the full test.
    for start in range(0, repeats, 250):
        size = min(250, repeats - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
    return float((extreme + 1) / (repeats + 1))


def _technology_label(feature_name: str) -> str:
    parts = feature_name.split("__", 2)
    return parts[-1] if len(parts) == 3 else feature_name


def _fit_nmf_track(
    config: dict[str, Any],
    consensus_labels: np.ndarray,
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    processed = project_path(config, "processed_dir")
    artifacts = project_path(config, "artifacts_dir")
    tables = project_path(config, "tables_dir")
    models = artifacts / "models"
    seed = int(config["project"]["random_state"])
    settings = config["bonus"]

    feature_names = json.loads((processed / "feature_names.json").read_text(encoding="utf-8"))
    technology_names = feature_names["technology"]
    technology = sparse.load_npz(processed / "X_tech_stack.npz")[:, : len(technology_names)].tocsr()
    if technology.min() < 0:
        raise ValueError("NMF input must contain only non-negative technology indicators")

    scores: list[dict[str, Any]] = []
    fitted: dict[int, tuple[MiniBatchNMF, np.ndarray, np.ndarray]] = {}
    for k in settings["nmf_components"]:
        model = MiniBatchNMF(
            n_components=int(k),
            init="nndsvda",
            batch_size=int(settings["nmf_batch_size"]),
            max_iter=int(settings["nmf_max_iter"]),
            random_state=seed,
        )
        weights = model.fit_transform(technology).astype(np.float32)
        labels = weights.argmax(axis=1).astype(np.int16)
        counts = np.bincount(labels, minlength=int(k))
        valid = np.count_nonzero(counts) > 1 and counts.min() / len(labels) >= 0.02
        sample_labels = labels[sample_indices]
        if len(np.unique(sample_labels)) > 1:
            cosine_silhouette = float(
                silhouette_score(technology[sample_indices], sample_labels, metric="cosine")
            )
            latent_silhouette = float(silhouette_score(weights[sample_indices], sample_labels))
        else:
            cosine_silhouette = latent_silhouette = float("nan")
        scores.append(
            {
                "components": int(k),
                "technology_cosine_silhouette": cosine_silhouette,
                "latent_silhouette": latent_silhouette,
                "minimum_cluster_fraction": float(counts.min() / len(labels)),
                "reconstruction_error_per_row": float(model.reconstruction_err_ / len(labels)),
                "ari_vs_consensus": float(adjusted_rand_score(consensus_labels, labels)),
                "nmi_vs_consensus": float(normalized_mutual_info_score(consensus_labels, labels)),
                "iterations": int(model.n_iter_),
                "valid": bool(valid),
            }
        )
        fitted[int(k)] = (model, weights, labels)

    scores_frame = pd.DataFrame(scores)
    valid_scores = scores_frame.loc[scores_frame["valid"]]
    if valid_scores.empty:
        raise RuntimeError("NMF did not produce a valid multi-cluster solution")
    chosen_k = int(
        valid_scores.loc[valid_scores["technology_cosine_silhouette"].idxmax(), "components"]
    )
    chosen_model, chosen_weights, chosen_labels = fitted[chosen_k]
    scores_frame["selected"] = scores_frame["components"].eq(chosen_k)
    scores_frame.to_csv(tables / "bonus_nmf_scores.csv", index=False, encoding="utf-8-sig")
    np.save(processed / "X_nmf_technology.npy", chosen_weights, allow_pickle=False)
    np.save(processed / "bonus_nmf_labels.npy", chosen_labels, allow_pickle=False)
    model_path = models / "bonus_nmf_technology.joblib"
    joblib.dump(chosen_model, model_path)

    component_rows: list[dict[str, Any]] = []
    for component, loadings in enumerate(chosen_model.components_):
        top = np.argsort(loadings)[::-1][:15]
        for rank, feature_index in enumerate(top, 1):
            component_rows.append(
                {
                    "component": int(component),
                    "rank": rank,
                    "feature": technology_names[feature_index],
                    "technology": _technology_label(technology_names[feature_index]),
                    "loading": float(loadings[feature_index]),
                }
            )
    pd.DataFrame(component_rows).to_csv(
        tables / "bonus_nmf_components.csv", index=False, encoding="utf-8-sig"
    )
    selected_row = scores_frame.loc[scores_frame["selected"]].iloc[0]
    return {
        "selected_components": chosen_k,
        "technology_cosine_silhouette": float(selected_row["technology_cosine_silhouette"]),
        "ari_vs_consensus": float(selected_row["ari_vs_consensus"]),
        "nmi_vs_consensus": float(selected_row["nmi_vs_consensus"]),
        "minimum_cluster_fraction": float(selected_row["minimum_cluster_fraction"]),
        "model_sha256": sha256_file(model_path),
        "labels": chosen_labels,
    }


def _bootstrap_comparisons(
    config: dict[str, Any],
    x_pca: np.ndarray,
    sample_indices: np.ndarray,
    consensus_sample_labels: np.ndarray,
    candidate_names: list[str],
    candidate_labelings: list[np.ndarray],
    nmf_labels: np.ndarray,
) -> dict[str, Any]:
    tables = project_path(config, "tables_dir")
    seed = int(config["project"]["random_state"])
    settings = config["bonus"]
    rng = np.random.default_rng(seed + 700)
    requested = int(settings["bootstrap_sample_size"])
    local = np.sort(rng.choice(len(sample_indices), size=min(requested, len(sample_indices)), replace=False))
    data = np.asarray(x_pca[sample_indices[local]], dtype=np.float32)
    reference = np.asarray(consensus_sample_labels[local])
    methods: dict[str, np.ndarray] = {"consensus": reference}
    methods.update(
        {
            name: np.asarray(labels)[local]
            for name, labels in zip(candidate_names, candidate_labelings)
        }
    )
    methods["nmf_high_dimensional"] = np.asarray(nmf_labels)[sample_indices[local]]
    methods = {
        name: labels
        for name, labels in methods.items()
        if len(np.unique(labels)) > 1 and np.bincount(labels.astype(int)).min() >= 2
    }

    repeats = int(settings["bootstrap_repeats"])
    metric_rows: list[dict[str, Any]] = []
    point_silhouettes: dict[str, np.ndarray] = {}
    for method, labels in methods.items():
        point_silhouettes[method] = silhouette_samples(data, labels)
        estimates = {
            "silhouette": float(point_silhouettes[method].mean()),
            "calinski_harabasz_per_row": float(calinski_harabasz_score(data, labels) / len(data)),
            "davies_bouldin": float(davies_bouldin_score(data, labels)),
            "ari_vs_consensus": float(adjusted_rand_score(reference, labels)),
            "nmi_vs_consensus": float(normalized_mutual_info_score(reference, labels)),
        }
        distributions: dict[str, list[float]] = {name: [] for name in estimates}
        for _ in range(repeats):
            boot = rng.integers(0, len(data), size=len(data))
            boot_labels = labels[boot]
            boot_reference = reference[boot]
            if len(np.unique(boot_labels)) < 2:
                continue
            distributions["silhouette"].append(float(point_silhouettes[method][boot].mean()))
            distributions["calinski_harabasz_per_row"].append(
                float(calinski_harabasz_score(data[boot], boot_labels) / len(boot))
            )
            distributions["davies_bouldin"].append(float(davies_bouldin_score(data[boot], boot_labels)))
            distributions["ari_vs_consensus"].append(
                float(adjusted_rand_score(boot_reference, boot_labels))
            )
            distributions["nmi_vs_consensus"].append(
                float(normalized_mutual_info_score(boot_reference, boot_labels))
            )
        for metric, estimate in estimates.items():
            lower, upper = percentile_interval(distributions[metric])
            metric_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence": 0.95,
                    "bootstrap_repeats": repeats,
                    "sample_size": len(data),
                    "reference_is_not_ground_truth": metric.endswith("vs_consensus"),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        tables / "bonus_metric_confidence_intervals.csv", index=False, encoding="utf-8-sig"
    )

    ranking = sorted(point_silhouettes, key=lambda name: point_silhouettes[name].mean(), reverse=True)
    comparisons: list[tuple[str, str, str]] = [
        ("best_vs_runner_up", ranking[0], ranking[1]),
    ]
    individual = max(
        (name for name in ranking if name != "consensus"),
        key=lambda name: point_silhouettes[name].mean(),
    )
    if {"consensus", individual} != {ranking[0], ranking[1]}:
        comparisons.append(("consensus_vs_best_individual", "consensus", individual))
    permutation_rows: list[dict[str, Any]] = []
    for comparison, first, second in comparisons:
        differences = point_silhouettes[first] - point_silhouettes[second]
        boot_means = [
            float(differences[rng.integers(0, len(differences), len(differences))].mean())
            for _ in range(repeats)
        ]
        lower, upper = percentile_interval(boot_means)
        permutation_rows.append(
            {
                "comparison": comparison,
                "first_method": first,
                "second_method": second,
                "mean_silhouette_difference": float(differences.mean()),
                "difference_ci_lower": lower,
                "difference_ci_upper": upper,
                "permutation_pvalue_two_sided": paired_sign_flip_pvalue(
                    differences,
                    repeats=int(settings["permutation_repeats"]),
                    random_state=seed + 701,
                ),
                "permutation_repeats": int(settings["permutation_repeats"]),
                "sample_size": len(data),
            }
        )
    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(tables / "bonus_permutation_tests.csv", index=False, encoding="utf-8-sig")
    best_row = permutation.iloc[0]
    return {
        "best_method": ranking[0],
        "runner_up": ranking[1],
        "mean_silhouette_difference": float(best_row["mean_silhouette_difference"]),
        "permutation_pvalue": float(best_row["permutation_pvalue_two_sided"]),
        "bootstrap_repeats": repeats,
        "permutation_repeats": int(settings["permutation_repeats"]),
        "sample_size": len(data),
    }


def _split_stability(
    config: dict[str, Any],
    x_pca: np.ndarray,
) -> dict[str, Any]:
    processed = project_path(config, "processed_dir")
    artifacts = project_path(config, "artifacts_dir")
    tables = project_path(config, "tables_dir")
    seed = int(config["project"]["random_state"])
    k = int(config["bonus"]["split_clusters"])

    metadata = pd.read_parquet(processed / "respondent_metadata.parquet")
    experience = metadata["ExperienceConsensus"].to_numpy()
    threshold = float(np.nanmedian(experience))
    split_positions = [np.flatnonzero(experience <= threshold), np.flatnonzero(experience > threshold)]
    split_names = [f"experience_le_{threshold:g}", f"experience_gt_{threshold:g}"]
    technology = sparse.load_npz(processed / "X_tech_stack.npz")[:, :-1].tocsr()
    feature_names = json.loads((processed / "feature_names.json").read_text(encoding="utf-8"))[
        "technology"
    ]

    fitted: list[MiniBatchKMeans] = []
    labels_by_split: list[np.ndarray] = []
    top_features: list[dict[int, set[int]]] = []
    for split_index, positions in enumerate(split_positions):
        model = MiniBatchKMeans(
            n_clusters=k,
            n_init=20,
            batch_size=2048,
            random_state=seed + split_index,
        ).fit(np.asarray(x_pca[positions], dtype=np.float32))
        labels = model.labels_
        fitted.append(model)
        labels_by_split.append(labels)
        per_cluster: dict[int, set[int]] = {}
        for cluster in range(k):
            members = positions[labels == cluster]
            prevalence = np.asarray(technology[members].mean(axis=0)).ravel()
            per_cluster[cluster] = set(np.argsort(prevalence)[::-1][:20].tolist())
        top_features.append(per_cluster)

    first_clusters, second_clusters = linear_sum_assignment(
        cdist(fitted[0].cluster_centers_, fitted[1].cluster_centers_)
    )
    rows: list[dict[str, Any]] = []
    for first, second in zip(first_clusters, second_clusters):
        a = top_features[0][int(first)]
        b = top_features[1][int(second)]
        intersection = a & b
        union = a | b
        rows.append(
            {
                "split_a": split_names[0],
                "cluster_a": int(first),
                "split_b": split_names[1],
                "cluster_b": int(second),
                "rows_a": int(np.count_nonzero(labels_by_split[0] == first)),
                "rows_b": int(np.count_nonzero(labels_by_split[1] == second)),
                "prototype_distance_pca": float(
                    np.linalg.norm(
                        fitted[0].cluster_centers_[first] - fitted[1].cluster_centers_[second]
                    )
                ),
                "top20_technology_jaccard": float(len(intersection) / len(union)),
                "shared_top_technologies": "; ".join(
                    sorted(_technology_label(feature_names[index]) for index in intersection)
                ),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(tables / "bonus_split_stability.csv", index=False, encoding="utf-8-sig")
    joblib.dump(
        {"threshold": threshold, "split_names": split_names, "models": fitted},
        artifacts / "models" / "bonus_split_models.joblib",
    )
    return {
        "stratification": "ExperienceConsensus median split",
        "threshold": threshold,
        "split_rows": [int(len(positions)) for positions in split_positions],
        "mean_top20_technology_jaccard": float(stability["top20_technology_jaccard"].mean()),
        "minimum_top20_technology_jaccard": float(stability["top20_technology_jaccard"].min()),
    }


def _build_umap_3d(
    config: dict[str, Any],
    x_pca: np.ndarray,
    consensus_labels: np.ndarray,
) -> dict[str, Any]:
    import plotly.express as px
    import umap

    processed = project_path(config, "processed_dir")
    artifacts = project_path(config, "artifacts_dir")
    figures = project_path(config, "figures_dir")
    tables = project_path(config, "tables_dir")
    seed = int(config["project"]["random_state"])
    size = min(int(config["bonus"]["umap_3d_sample_size"]), len(x_pca))
    rng = np.random.default_rng(seed + 900)
    indices = np.sort(rng.choice(len(x_pca), size=size, replace=False))
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(config["features"]["umap_neighbors"]),
        min_dist=float(config["features"]["umap_min_dist"]),
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(np.asarray(x_pca[indices], dtype=np.float32)).astype(np.float32)
    np.save(processed / "X_umap_3d_sample.npy", embedding, allow_pickle=False)
    np.save(processed / "umap_3d_sample_indices.npy", indices, allow_pickle=False)
    model_path = artifacts / "models" / "umap_3d_sample.joblib"
    joblib.dump(reducer, model_path)

    metadata = pd.read_parquet(processed / "respondent_metadata.parquet")
    profiles = pd.read_csv(tables / "phase3_cluster_profiles.csv").set_index("cluster")["profile_name"]
    cluster = consensus_labels[indices].astype(int)
    frame = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "UMAP3": embedding[:, 2],
            "cluster": cluster.astype(str),
            "profile": [profiles.loc[value] for value in cluster],
            "ResponseId": metadata.iloc[indices]["ResponseId"].to_numpy(),
        }
    )
    frame.to_csv(tables / "bonus_umap_3d.csv", index=False, encoding="utf-8-sig")
    figure = px.scatter_3d(
        frame,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="cluster",
        hover_data=["profile", "ResponseId"],
        opacity=0.45,
        title="Interactive 3D UMAP of consensus developer profiles",
        color_discrete_sequence=["#087E8B", "#FF5A5F", "#F4B942", "#0B3954"],
    )
    figure.update_traces(marker={"size": 2.2})
    html_path = figures / "phase3_umap_3d_interactive.html"
    figure.write_html(html_path, include_plotlyjs=True, full_html=True, auto_open=False)
    return {
        "sample_size": size,
        "dimensions": 3,
        "html_path": str(html_path.relative_to(Path(config["_project_root"]))),
        "model_sha256": sha256_file(model_path),
    }


def run_bonus_analysis(
    config: dict[str, Any],
    *,
    x_pca: np.ndarray,
    consensus_labels: np.ndarray,
    consensus_sample_labels: np.ndarray,
    candidate_names: list[str],
    candidate_labelings: list[np.ndarray],
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    """Run the four extensions that make up the project's 15-point bonus target."""
    nmf = _fit_nmf_track(config, consensus_labels, sample_indices)
    nmf_labels = nmf.pop("labels")
    significance = _bootstrap_comparisons(
        config,
        x_pca,
        sample_indices,
        consensus_sample_labels,
        candidate_names,
        candidate_labelings,
        nmf_labels,
    )
    split_stability = _split_stability(config, x_pca)
    umap_3d = _build_umap_3d(config, x_pca, consensus_labels)
    summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "points_targeted": 15,
        "extensions": {
            "second_advanced_track_nmf": 5,
            "bootstrap_ci_and_permutation_testing": 3,
            "stability_across_meaningful_splits": 4,
            "interactive_3d_umap": 3,
        },
        "nmf": nmf,
        "significance": significance,
        "split_stability": split_stability,
        "umap_3d": umap_3d,
    }
    write_json(summary, project_path(config, "artifacts_dir") / "bonus_summary.json")
    return summary
