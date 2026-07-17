from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from .config import project_path
from .features import split_multilabel


COLORS = ["#0B3954", "#087E8B", "#5BC0BE", "#FF5A5F", "#C81D25", "#F4D35E"]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("Missing"), y.fillna("Missing"))
    if table.empty or min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    cols_corrected = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def create_phase1_tables(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    tables_dir = project_path(config, "tables_dir")
    tech_columns = config["data"]["technology_columns"]
    prevalence_rows: list[dict[str, Any]] = []
    for column in tech_columns:
        domain = column.replace("HaveWorkedWith", "")
        counts = frame[column].map(split_multilabel).explode().dropna().value_counts()
        for technology, count in counts.items():
            prevalence_rows.append(
                {
                    "domain": domain,
                    "technology": technology,
                    "respondents": int(count),
                    "pct_of_cohort": 100 * float(count) / len(frame),
                }
            )
    pd.DataFrame(prevalence_rows).sort_values(
        ["domain", "respondents"], ascending=[True, False]
    ).to_csv(tables_dir / "technology_prevalence.csv", index=False, encoding="utf-8-sig")

    numeric = [
        "YearsCodeNumeric",
        "YearsCodeProNumeric",
        "WorkExpNumeric",
        "ExperienceConsensus",
        "ExperienceSpread",
        "ProfessionalExperienceRatio",
        "OrgSizeLog",
        "TechnologyBreadth",
        "ConvertedCompYearly",
    ]
    frame[numeric].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_csv(
        tables_dir / "numeric_summary.csv", encoding="utf-8-sig"
    )

    categorical = ["RemoteWork", "Country", "DevType", "EdLevel", "AISelect"]
    category_rows: list[dict[str, Any]] = []
    for column in categorical:
        counts = frame[column].fillna("Missing").value_counts(dropna=False)
        for value, count in counts.items():
            category_rows.append(
                {
                    "column": column,
                    "category": value,
                    "count": int(count),
                    "pct": 100 * float(count) / len(frame),
                }
            )
    pd.DataFrame(category_rows).to_csv(
        tables_dir / "categorical_distributions.csv", index=False, encoding="utf-8-sig"
    )

    associations = pd.DataFrame(np.eye(len(categorical)), index=categorical, columns=categorical)
    for left, right in combinations(categorical, 2):
        value = _cramers_v(frame[left], frame[right])
        associations.loc[left, right] = value
        associations.loc[right, left] = value
    associations.to_csv(tables_dir / "categorical_cramers_v.csv", encoding="utf-8-sig")


def create_phase1_figures(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    _style()
    figures_dir = project_path(config, "figures_dir")
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    tables_dir = project_path(config, "tables_dir")
    random_state = int(config["project"]["random_state"])
    rng = np.random.default_rng(random_state)

    profile = pd.DataFrame(
        {
            "column": raw_frame.columns,
            "missing_pct": 100 * raw_frame.isna().mean().to_numpy(),
        }
    ).sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(profile["column"], profile["missing_pct"], color=COLORS[1])
    ax.set(xlabel="Missing values (%)", ylabel="", title="Missingness of selected raw survey fields")
    ax.axvline(30, color=COLORS[3], linestyle="--", linewidth=1, label="30%")
    ax.legend(frameon=False)
    _save(fig, figures_dir, "phase1_missingness_selected_fields")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, column, title in zip(
        axes,
        ["YearsCodeNumeric", "YearsCodeProNumeric", "WorkExpNumeric"],
        ["Years coding", "Professional coding", "Work experience"],
    ):
        sns.histplot(clean_frame[column].dropna(), bins=35, kde=True, color=COLORS[1], ax=ax)
        ax.set(title=title, xlabel="Years", ylabel="Respondents")
    fig.suptitle("Experience distributions before imputation", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, figures_dir, "phase1_experience_distributions")

    tech = pd.read_csv(tables_dir / "technology_prevalence.csv")
    top = tech.sort_values("respondents", ascending=False).head(25).sort_values("respondents")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["technology"], top["pct_of_cohort"], color=COLORS[0])
    ax.set(xlabel="Respondents in cleaned cohort (%)", ylabel="", title="Top 25 technologies used in 2024")
    _save(fig, figures_dir, "phase1_top_technologies")

    preprocessor = joblib.load(artifacts_dir / "models" / "phase1_preprocessor.joblib")
    numeric_columns = preprocessor["numeric_columns"]
    pair = ["ExperienceConsensus", "OrgSizeLog"]
    pair_indices = [numeric_columns.index(name) for name in pair]
    raw_numeric = clean_frame[numeric_columns].astype(float).to_numpy()
    sample_size = min(10000, len(clean_frame))
    sample = rng.choice(len(clean_frame), size=sample_size, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    axes[0].scatter(
        raw_numeric[sample, pair_indices[0]],
        raw_numeric[sample, pair_indices[1]],
        s=5,
        alpha=0.15,
        color=COLORS[0],
    )
    axes[0].set(title="Before scaling", xlabel=pair[0], ylabel=pair[1])
    for ax, scaler_name in zip(axes[1:], ["standard", "robust", "minmax"]):
        transformed = preprocessor["scalers"][scaler_name].transform(raw_numeric[sample])
        ax.scatter(
            transformed[:, pair_indices[0]],
            transformed[:, pair_indices[1]],
            s=5,
            alpha=0.15,
            color=COLORS[1],
        )
        ax.set(title=f"{scaler_name.title()} scaler", xlabel=pair[0], ylabel=pair[1])
    fig.suptitle("Geometry of two features before and after scaling", fontweight="bold")
    fig.tight_layout()
    _save(fig, figures_dir, "phase1_scaler_comparison")

    pca_variance = pd.read_csv(tables_dir / "pca_explained_variance.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        pca_variance["component"],
        pca_variance["cumulative_explained_variance"],
        marker="o",
        markersize=3,
        color=COLORS[0],
    )
    for level, color in [(0.80, COLORS[2]), (0.90, COLORS[3])]:
        ax.axhline(level, color=color, linestyle="--", linewidth=1, label=f"{int(level*100)}%")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance", title="PCA diagnostic")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    _save(fig, figures_dir, "phase1_pca_explained_variance")

    x_pca = np.load(processed_dir / "X_pca.npy", mmap_mode="r")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    density = ax.hexbin(x_pca[:, 0], x_pca[:, 1], gridsize=75, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(density, ax=ax, label="log10(count)")
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA projection — respondent density")
    _save(fig, figures_dir, "phase1_pca_density")

    x_umap = np.load(processed_dir / "X_umap_sample.npy")
    if len(x_umap):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        density = ax.hexbin(
            x_umap[:, 0], x_umap[:, 1], gridsize=70, bins="log", mincnt=1, cmap="magma"
        )
        fig.colorbar(density, ax=ax, label="log10(count)")
        ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="UMAP sample — respondent density")
        _save(fig, figures_dir, "phase1_umap_density")

    vat = np.load(processed_dir / "vat_reordered_distances.npy", mmap_mode="r")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(vat, cmap="mako", aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Euclidean dissimilarity")
    ax.set(title="VAT — reordered dissimilarity matrix", xlabel="Reordered respondents", ylabel="Reordered respondents")
    _save(fig, figures_dir, "phase1_vat_heatmap")

    numeric_corr_columns = [
        "YearsCodeNumericImputed",
        "YearsCodeProNumericImputed",
        "WorkExpNumericImputed",
        "ExperienceConsensus",
        "ExperienceSpread",
        "ProfessionalExperienceRatio",
        "OrgSizeLog",
        "TechnologyBreadth",
    ]
    correlation = clean_frame[numeric_corr_columns].corr(method="spearman")
    correlation.to_csv(tables_dir / "numeric_spearman_correlation.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Spearman correlation of engineered numeric features")
    fig.tight_layout()
    _save(fig, figures_dir, "phase1_numeric_correlation")

    categorical = pd.read_csv(tables_dir / "categorical_cramers_v.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(categorical, cmap="crest", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Categorical association — bias-corrected Cramer's V")
    fig.tight_layout()
    _save(fig, figures_dir, "phase1_categorical_cramers_v")
