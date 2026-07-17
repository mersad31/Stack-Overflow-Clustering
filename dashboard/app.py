from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Developer Cluster Explorer", layout="wide")
page = st.sidebar.radio(
    "Page", ["Overview", "Cluster Explorer", "Evaluation", "Live Assignment", "3D Explorer"]
)
profiles = pd.read_csv(TABLES / "phase3_cluster_profiles.csv")
summary = json.loads((ARTIFACTS / "phase3_run_summary.json").read_text(encoding="utf-8"))


if page == "Overview":
    st.title("Stack Overflow 2024 — Developer Profiles")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{sum(profiles.respondents):,}")
    c2.metric("Consensus clusters", summary["consensus_k"])
    c3.metric("Input features", 328)
    c4.metric("Assignment accuracy", f"{100 * summary['validation_accuracy']:.1f}%")
    st.markdown(
        "**Chosen method:** average-linkage consensus over K-Means, Ward, GMM, and "
        "Spectral candidates on 50 PCA components. The result is exploratory, not ground truth."
    )
    left, right = st.columns([1, 1.5])
    with left:
        st.dataframe(
            profiles[["cluster", "profile_name", "respondents", "share_pct", "TechnologyBreadth_median"]],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.image(str(ROOT / "reports" / "figures" / "phase3_consensus_selection.png"))

elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    embedding = np.load(PROCESSED / "X_umap_sample.npy")
    indices = np.load(PROCESSED / "umap_sample_indices.npy")
    labels = np.load(PROCESSED / "phase3_consensus_labels.npy")[indices]
    metadata = pd.read_parquet(PROCESSED / "respondent_metadata.parquet").iloc[indices]
    embedding_frame = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "cluster": labels.astype(str),
            "Experience": metadata["ExperienceConsensus"].to_numpy(),
            "TechnologyBreadth": metadata["TechnologyBreadth"].to_numpy(),
            "ResponseId": metadata["ResponseId"].to_numpy(),
        }
    )
    available = sorted(embedding_frame.cluster.unique())
    selected_clusters = st.multiselect("Clusters shown", available, default=available)
    exp_min, exp_max = float(embedding_frame.Experience.min()), float(embedding_frame.Experience.max())
    experience_range = st.slider(
        "Experience range", min_value=exp_min, max_value=exp_max, value=(exp_min, exp_max)
    )
    filtered = embedding_frame[
        embedding_frame.cluster.isin(selected_clusters)
        & embedding_frame.Experience.between(*experience_range)
    ]
    figure = px.scatter(
        filtered,
        x="UMAP1",
        y="UMAP2",
        color="cluster",
        hover_data=["ResponseId", "Experience", "TechnologyBreadth"],
        opacity=0.4,
        color_discrete_sequence=["#087E8B", "#FF5A5F", "#F4B942", "#0B3954"],
    )
    figure.update_traces(marker={"size": 3})
    st.plotly_chart(figure, width="stretch")

    cluster = st.selectbox("Profile panel", profiles.cluster)
    st.dataframe(profiles.query("cluster==@cluster"), width="stretch", hide_index=True)
    tech = pd.read_csv(TABLES / "phase3_cluster_top_technologies.csv")
    domain = st.selectbox("Technology domain", sorted(tech.domain.unique()))
    st.dataframe(
        tech.query("cluster==@cluster and domain==@domain"), width="stretch", hide_index=True
    )
    st.dataframe(
        pd.read_csv(TABLES / "phase3_exemplars_boundaries.csv").query("cluster==@cluster"),
        width="stretch",
        hide_index=True,
    )

elif page == "Evaluation":
    st.title("Evaluation, Agreement, and Robustness")
    st.image(str(ROOT / "reports" / "figures" / "phase2_algorithm_agreement.png"))
    st.image(str(ROOT / "reports" / "figures" / "phase3_sensitivity.png"))
    internal, proxy, stability, bonus = st.tabs(
        ["Internal metrics", "Proxy-label metrics", "Stability", "Bonus inference"]
    )
    with internal:
        st.dataframe(pd.read_csv(TABLES / "phase2_model_family_scores.csv"), width="stretch")
    with proxy:
        st.caption("These survey fields are interpretive proxies, not cluster ground truth.")
        st.dataframe(pd.read_csv(TABLES / "phase2_proxy_label_metrics.csv"), width="stretch")
    with stability:
        st.dataframe(pd.read_csv(TABLES / "phase2_seed_stability_pairwise_ari.csv"), width="stretch")
        st.dataframe(pd.read_csv(TABLES / "phase3_downstream_compensation.csv"), width="stretch")
        st.dataframe(pd.read_csv(TABLES / "phase3_drift_baseline.csv"), width="stretch")
    with bonus:
        st.dataframe(pd.read_csv(TABLES / "bonus_metric_confidence_intervals.csv"), width="stretch")
        st.dataframe(pd.read_csv(TABLES / "bonus_permutation_tests.csv"), width="stretch")
        st.dataframe(pd.read_csv(TABLES / "bonus_split_stability.csv"), width="stretch")

elif page == "Live Assignment":
    st.title("Live Assignment")
    st.caption(
        "Enter a preprocessed PCA record or upload a CSV containing PC1…PC50. "
        "Assignments and Euclidean distances to every consensus centroid are returned."
    )
    model = joblib.load(ARTIFACTS / "models" / "phase3_consensus_assigner.joblib")
    centroids = np.load(PROCESSED / "phase3_consensus_centroids.npy")
    columns = [f"PC{i}" for i in range(1, 51)]

    def assign(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[columns].to_numpy(dtype=float)
        result = frame.copy()
        result["cluster"] = model.predict(values).astype(int)
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        for cluster_id in range(centroids.shape[0]):
            result[f"distance_cluster_{cluster_id}"] = distances[:, cluster_id]
        return result

    upload = st.file_uploader("CSV", type="csv")
    if upload:
        uploaded = pd.read_csv(upload)
        missing = [column for column in columns if column not in uploaded]
        if missing:
            st.error(f"Missing columns: {missing[:8]}")
        else:
            assigned = assign(uploaded)
            st.dataframe(assigned, width="stretch")
            st.download_button(
                "Download assignments",
                assigned.to_csv(index=False).encode(),
                "assignments.csv",
                "text/csv",
            )
    else:
        st.caption("Edit all 50 components; zeros represent the PCA origin, not missing values.")
        record = st.data_editor(
            pd.DataFrame([{column: 0.0 for column in columns}]), hide_index=True, width="stretch"
        )
        if st.button("Assign"):
            assigned = assign(record)
            st.success(f"Assigned cluster: {int(assigned.loc[0, 'cluster'])}")
            distance_columns = [column for column in assigned if column.startswith("distance_cluster_")]
            st.dataframe(assigned[distance_columns], width="stretch", hide_index=True)

else:
    st.title("Interactive 3D UMAP")
    st.caption(
        "A 10,000-response UMAP projection for exploration only; distances in this view are not "
        "used for clustering. A standalone HTML copy is included in reports/figures."
    )
    frame = pd.read_csv(TABLES / "bonus_umap_3d.csv", dtype={"cluster": str})
    shown = st.multiselect("Clusters shown", sorted(frame.cluster.unique()), default=sorted(frame.cluster.unique()))
    frame = frame[frame.cluster.isin(shown)]
    figure = px.scatter_3d(
        frame,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="cluster",
        hover_data=["profile", "ResponseId"],
        opacity=0.45,
        color_discrete_sequence=["#087E8B", "#FF5A5F", "#F4B942", "#0B3954"],
    )
    figure.update_traces(marker={"size": 2.2})
    st.plotly_chart(figure, width="stretch")
