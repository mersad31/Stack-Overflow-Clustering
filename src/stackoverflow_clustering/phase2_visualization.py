from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import project_path


def _save(fig: plt.Figure, directory, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_partitioning_figures(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    figures = project_path(config, "figures_dir")
    tables = project_path(config, "tables_dir")
    processed = project_path(config, "processed_dir")
    sns.set_theme(style="whitegrid", context="notebook")

    scores = pd.read_csv(tables / "phase2_partitioning_scores.csv")
    kmeans = scores[scores["algorithm"] == "kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algorithm, group in scores.groupby("algorithm"):
        axes[0].plot(group["k"], group["inertia"], marker="o", label=algorithm)
        axes[1].plot(group["k"], group["silhouette"], marker="o", label=algorithm)
    knee = manifest["recommendations"]["elbow_kneedle"]
    if knee is not None:
        axes[0].axvline(knee, color="#C81D25", linestyle="--", label=f"Kneedle: k={knee}")
    axes[0].set(title="Elbow diagnostic", xlabel="k", ylabel="SSE / inertia")
    axes[1].set(title="Silhouette model selection", xlabel="k", ylabel="Silhouette")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, figures, "phase2_partitioning_selection")

    gaps = pd.read_csv(tables / "phase2_gap_statistic.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(gaps["k"], gaps["gap"], yerr=gaps["gap_standard_error"], marker="o", capsize=4)
    selected = manifest["recommendations"]["gap_one_se"]
    ax.axvline(selected, color="#C81D25", linestyle="--", label=f"1-SE selection: k={selected}")
    ax.set(title="Gap Statistic", xlabel="k", ylabel="Gap(k)")
    ax.legend(frameon=False)
    _save(fig, figures, "phase2_gap_statistic")

    stability = pd.read_csv(tables / "phase2_seed_stability_pairwise_ari.csv")
    bootstrap = pd.read_csv(tables / "phase2_bootstrap_stability_pairwise_ari.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(stability["ari"], color="#087E8B", label="20 seeds", kde=True, ax=ax)
    sns.histplot(bootstrap["ari"], color="#FF5A5F", label="bootstrap", kde=True, ax=ax)
    ax.set(title="Partition stability", xlabel="Pairwise adjusted Rand index", ylabel="Pairs")
    ax.legend(frameon=False)
    _save(fig, figures, "phase2_partitioning_stability")

    k_stability = pd.read_csv(tables / "phase2_bootstrap_k_stability.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(k_stability["k"], k_stability["mean"], yerr=k_stability["std"], marker="o", capsize=4, color="#087E8B")
    ax.set(title="Bootstrap stability across candidate k", xlabel="k", ylabel="Mean pairwise ARI")
    _save(fig, figures, "phase2_bootstrap_k_stability")

    coassociation = np.load(processed / "phase2_bootstrap_coassociation.npy")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(coassociation, cmap="mako", vmin=0, vmax=1, xticklabels=False, yticklabels=False, ax=ax)
    ax.set(title="Bootstrap co-clustering probability", xlabel="Evaluation sample", ylabel="Evaluation sample")
    _save(fig, figures, "phase2_bootstrap_coassociation")

    labels = np.load(processed / "phase2_partitioning_labels.npy")
    embedding = np.load(processed / "X_umap_sample.npy")
    indices = np.load(processed / "umap_sample_indices.npy")
    if len(embedding):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels[indices], cmap="tab10", s=4, alpha=0.35
        )
        ax.set(title="Winning partition on the Phase 1 UMAP sample", xlabel="UMAP-1", ylabel="UMAP-2")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        _save(fig, figures, "phase2_partitioning_umap")
