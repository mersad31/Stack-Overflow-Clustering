from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from .config import project_path


def _save(fig: plt.Figure, directory, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_model_family_figures(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    figures = project_path(config, "figures_dir")
    tables = project_path(config, "tables_dir")
    processed = project_path(config, "processed_dir")
    sns.set_theme(style="whitegrid", context="notebook")
    scores = pd.read_csv(tables / "phase2_model_family_scores.csv")

    minimum_fraction = float(manifest["minimum_cluster_fraction_for_selection"])
    comparable = scores[
        scores["silhouette"].notna() & (scores["minimum_cluster_fraction"] >= minimum_fraction)
    ].copy()
    comparable["configuration"] = comparable["algorithm"] + comparable["k_requested"].map(
        lambda value: "" if pd.isna(value) else f" (k={int(value)})"
    )
    top = comparable.sort_values("silhouette", ascending=False).head(18).sort_values("silhouette")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["configuration"], top["silhouette"], color="#087E8B")
    ax.set(title="Best cross-family configurations on a common sample", xlabel="Silhouette", ylabel="")
    _save(fig, figures, "phase2_family_silhouette_comparison")

    gmm = scores[scores["family"] == "model_based"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algorithm, group in gmm.groupby("algorithm"):
        axes[0].plot(group["k_requested"], group["bic"], marker="o", label=algorithm)
        axes[1].plot(group["k_requested"], group["aic"], marker="o", label=algorithm)
    axes[0].set(title="GMM Bayesian information criterion", xlabel="Components", ylabel="BIC")
    axes[1].set(title="GMM Akaike information criterion", xlabel="Components", ylabel="AIC")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, figures, "phase2_gmm_information_criteria")

    kdist = pd.read_csv(tables / "phase2_k_distance.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(kdist["rank"], kdist["k_distance"], color="#0B3954")
    for q in (0.85, 0.90, 0.95):
        ax.axhline(kdist["k_distance"].quantile(q), linestyle="--", linewidth=1, label=f"q={q:.2f}")
    ax.set(title="DBSCAN 20-nearest-neighbor distance diagnostic", xlabel="Ordered respondent", ylabel="20-NN distance")
    ax.legend(frameon=False)
    _save(fig, figures, "phase2_k_distance")

    density = scores[scores["family"] == "density"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(density["noise_fraction"], density["silhouette"], s=80, color="#FF5A5F")
    for _, row in density.iterrows():
        ax.annotate(row["algorithm"], (row["noise_fraction"], row["silhouette"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Density-model separation versus rejected noise", xlabel="Noise fraction", ylabel="Silhouette on non-noise rows")
    _save(fig, figures, "phase2_density_tradeoff")

    reachability = pd.read_csv(tables / "phase2_optics_reachability.csv")
    finite = np.isfinite(reachability["reachability"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(reachability.loc[finite, "order"], reachability.loc[finite, "reachability"], color="#0B3954", linewidth=0.8)
    ax.set(title="OPTICS reachability diagnostic", xlabel="OPTICS ordering", ylabel="Reachability distance")
    _save(fig, figures, "phase2_optics_reachability")

    condensed = pd.read_csv(tables / "phase2_hdbscan_condensed_tree.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    points = ax.scatter(condensed["lambda_val"], condensed["child_size"], c=np.log1p(condensed["child_size"]), cmap="viridis", s=10, alpha=0.55)
    ax.set_yscale("log")
    ax.set(title="HDBSCAN condensed-tree diagnostic", xlabel="Persistence level (lambda)", ylabel="Child size (log scale)")
    fig.colorbar(points, ax=ax, label="log(1 + child size)")
    _save(fig, figures, "phase2_hdbscan_condensed_tree")

    trees = np.load(processed / "phase2_hierarchical_linkages.npz")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, method in zip(axes.ravel(), ("single", "complete", "average", "ward")):
        dendrogram(trees[method], truncate_mode="lastp", p=35, no_labels=True, ax=ax)
        ax.set(title=f"{method.title()} linkage", xlabel="Truncated branches", ylabel="Merge distance")
    fig.suptitle("Hierarchical diagnostics (common sample)", fontweight="bold")
    fig.tight_layout()
    _save(fig, figures, "phase2_hierarchical_dendrograms")

    agreement = pd.read_csv(tables / "phase2_algorithm_agreement.csv")
    names = sorted(set(agreement["candidate_a"]) | set(agreement["candidate_b"]))
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for _, row in agreement.iterrows():
        matrix.loc[row["candidate_a"], row["candidate_b"]] = row["adjusted_rand_index"]
        matrix.loc[row["candidate_b"], row["candidate_a"]] = row["adjusted_rand_index"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, cmap="vlag", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Agreement of cross-family candidate partitions (ARI)")
    fig.tight_layout()
    _save(fig, figures, "phase2_algorithm_agreement")
