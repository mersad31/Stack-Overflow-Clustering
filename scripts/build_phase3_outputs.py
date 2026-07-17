from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures() -> None:
    tables = ROOT / "reports" / "tables"
    output = ROOT / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    profiles = pd.read_csv(tables / "phase3_cluster_profiles.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(profiles["cluster"].astype(str), profiles["share_pct"], color=["#087E8B", "#FF5A5F"])
    ax.set(title="Consensus cluster sizes", xlabel="Cluster", ylabel="Respondents (%)")
    _save(fig, output / "phase3_cluster_sizes.png")

    sensitivity = pd.read_csv(tables / "phase3_sensitivity.csv").sort_values("ari_vs_consensus")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(sensitivity["variant"], sensitivity["ari_vs_consensus"], color="#0B3954")
    ax.set(xlim=(0, 1), title="Sensitivity to representation and scaling", xlabel="ARI versus consensus")
    _save(fig, output / "phase3_sensitivity.png")

    importance = (
        pd.read_csv(tables / "phase3_feature_importance_shap.csv")
        .sort_values("shap_mean_abs", ascending=False)
        .head(12)
        .sort_values("shap_mean_abs")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["shap_mean_abs"], color="#087E8B")
    ax.set(title="Consensus assignment explanation", xlabel="Mean absolute SHAP value")
    _save(fig, output / "phase3_shap_importance.png")

    selection = pd.read_csv(tables / "phase3_consensus_k_selection.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(selection["k"], selection["silhouette"], marker="o")
    ax.set(title="Consensus k selection", xlabel="k", ylabel="Silhouette")
    _save(fig, output / "phase3_consensus_selection.png")

    nmf = pd.read_csv(tables / "bonus_nmf_scores.csv")
    selected = nmf.loc[nmf["selected"]].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(nmf["components"], nmf["technology_cosine_silhouette"], marker="o", label="Technology cosine space")
    ax.plot(nmf["components"], nmf["latent_silhouette"], marker="s", label="NMF latent space")
    ax.axvline(
        selected["components"],
        color="#FF5A5F",
        linestyle="--",
        label=f"Selected k={int(selected['components'])}",
    )
    ax.set(
        title="Second advanced track: NMF component selection",
        xlabel="NMF components / hard clusters",
        ylabel="Silhouette",
    )
    ax.legend()
    _save(fig, output / "bonus_nmf_comparison.png")

    confidence = (
        pd.read_csv(tables / "bonus_metric_confidence_intervals.csv")
        .query("metric == 'silhouette'")
        .sort_values("estimate")
    )
    errors = np.vstack(
        [
            confidence["estimate"] - confidence["ci_lower"],
            confidence["ci_upper"] - confidence["estimate"],
        ]
    )
    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.errorbar(
        confidence["estimate"],
        confidence["method"],
        xerr=errors,
        fmt="o",
        color="#0B3954",
        ecolor="#087E8B",
        capsize=3,
    )
    ax.set(title="Bootstrap 95% confidence intervals", xlabel="Mean silhouette", ylabel="Method")
    _save(fig, output / "bonus_silhouette_confidence_intervals.png")

    split = pd.read_csv(tables / "bonus_split_stability.csv")
    labels = [f"{int(a)}↔{int(b)}" for a, b in zip(split["cluster_a"], split["cluster_b"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, split["top20_technology_jaccard"], color="#087E8B")
    ax.set(
        ylim=(0, 1),
        title="Prototype stability across experience splits",
        xlabel="Matched clusters (junior↔senior)",
        ylabel="Top-20 technology Jaccard",
    )
    _save(fig, output / "bonus_split_stability.png")


def notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell("# فاز سوم — Consensus، پروفایل، توضیح‌پذیری و افزونه‌ها"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, pandas as pd\n"
            "from IPython.display import display, Image\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "json.loads((ROOT / 'artifacts/phase3_run_summary.json').read_text())"
        ),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/phase3_cluster_profiles.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/phase3_cluster_sizes.png', width=700))"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/phase3_cluster_top_technologies.csv').groupby('cluster').head(15)"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/phase3_shap_importance.png', width=750))"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/phase3_downstream_compensation.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/phase3_sensitivity.png', width=750))"),
        nbf.v4.new_markdown_cell("## هدف امتیاز افزوده: ۱۵ امتیاز"),
        nbf.v4.new_code_cell("json.loads((ROOT / 'artifacts/bonus_summary.json').read_text())"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/bonus_nmf_scores.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/bonus_nmf_comparison.png', width=750))"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/bonus_metric_confidence_intervals.csv')"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/bonus_permutation_tests.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/bonus_silhouette_confidence_intervals.png', width=800))"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT / 'reports/tables/bonus_split_stability.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT / 'reports/figures/bonus_split_stability.png', width=700))"),
        nbf.v4.new_markdown_cell(
            "NMF چهار مؤلفه فناوری را برگزید، اما توافق آن با consensus ناچیز بود؛ بنابراین ساختار فناوری‌محور مکمل است نه تأیید همان دو پروفایل. "
            "آزمون permutation نیز برتری silhouette روش Ward بر consensus را نشان داد. پایداری واژگان فناوری میان دو نیمه تجربه متوسط است و نمایش UMAP سه‌بعدی فقط ابزار اکتشافی است."
        ),
    ]

    output = ROOT / "notebooks" / "03_phase3_consensus_executed.ipynb"
    executed = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbf.write(executed, output)


if __name__ == "__main__":
    figures()
    notebook()
    print("Phase 3 figures and notebook built")
