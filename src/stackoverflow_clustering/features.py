from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from .config import project_path
from .io_utils import sha256_file, utc_now_iso, write_json


def split_multilabel(value: object) -> list[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    return sorted({part.strip() for part in str(value).split(";") if part.strip()})


def _make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


def _save_sparse(matrix: sparse.spmatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, sparse.csr_matrix(matrix, dtype=np.float32), compressed=True)


def build_feature_matrices(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Build leakage-aware full and technology-stack representations."""
    processed_dir = project_path(config, "processed_dir")
    artifacts_dir = project_path(config, "artifacts_dir")
    tables_dir = project_path(config, "tables_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)

    tech_columns = config["data"]["technology_columns"]
    tech_blocks: list[sparse.csr_matrix] = []
    tech_feature_names: list[str] = []
    tech_encoders: dict[str, MultiLabelBinarizer] = {}
    tech_breadth_columns: list[str] = []

    working = frame.copy()
    for column in tech_columns:
        parsed = working[column].map(split_multilabel)
        encoder = MultiLabelBinarizer(sparse_output=True)
        block = encoder.fit_transform(parsed).astype(np.float32)
        prefix = column.replace("HaveWorkedWith", "")
        names = [f"tech__{prefix}__{item}" for item in encoder.classes_]
        tech_blocks.append(sparse.csr_matrix(block))
        tech_feature_names.extend(names)
        tech_encoders[column] = encoder
        breadth_column = f"{prefix}Breadth"
        working[breadth_column] = np.asarray(block.sum(axis=1)).ravel().astype(np.float32)
        tech_breadth_columns.append(breadth_column)

    tech_binary = sparse.hstack(tech_blocks, format="csr", dtype=np.float32)
    working["TechnologyBreadth"] = np.asarray(tech_binary.sum(axis=1)).ravel().astype(np.float32)

    employment_encoder = MultiLabelBinarizer(sparse_output=True)
    employment = employment_encoder.fit_transform(working["Employment"].map(split_multilabel))
    employment = sparse.csr_matrix(employment, dtype=np.float32)
    employment_names = [f"employment__{item}" for item in employment_encoder.classes_]

    categorical_columns = ["RemoteWork", "Country"]
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    categorical = categorical_encoder.fit_transform(working[categorical_columns])
    categorical_names = categorical_encoder.get_feature_names_out(categorical_columns).tolist()

    numeric_columns = [
        "YearsCodeNumericImputed",
        "YearsCodeProNumericImputed",
        "WorkExpNumericImputed",
        "ExperienceConsensus",
        "ExperienceSpread",
        "ProfessionalExperienceRatio",
        "OrgSizeLog",
        "YearsCodeNumericMissing",
        "YearsCodeProNumericMissing",
        "WorkExpNumericMissing",
        "OrgSizeMissing",
        *[f"{column}Missing" for column in tech_columns],
        *tech_breadth_columns,
        "TechnologyBreadth",
    ]
    numeric = working[numeric_columns].astype(np.float64).to_numpy()
    if not np.isfinite(numeric).all():
        bad = np.argwhere(~np.isfinite(numeric))[:10].tolist()
        raise ValueError(f"Non-finite engineered numeric values at positions {bad}")

    scaler_models: dict[str, Any] = {}
    representation_paths: dict[str, str] = {}
    full_feature_names = [
        *[f"numeric__{name}" for name in numeric_columns],
        *categorical_names,
        *employment_names,
        *tech_feature_names,
    ]
    binary = sparse.hstack([categorical, employment, tech_binary], format="csr", dtype=np.float32)

    selected_scaler = config["features"]["selected_scaler"]
    for scaler_name in config["features"]["scalers"]:
        scaler = _make_scaler(scaler_name)
        scaled_numeric = scaler.fit_transform(numeric).astype(np.float32)
        full = sparse.hstack(
            [sparse.csr_matrix(scaled_numeric), binary], format="csr", dtype=np.float32
        )
        output_path = processed_dir / f"X_full_{scaler_name}.npz"
        _save_sparse(full, output_path)
        representation_paths[f"X_full_{scaler_name}"] = str(output_path)
        scaler_models[scaler_name] = scaler
        if scaler_name == selected_scaler:
            selected_path = processed_dir / "X_full.npz"
            _save_sparse(full, selected_path)
            representation_paths["X_full"] = str(selected_path)

    breadth_scaler = RobustScaler()
    breadth_scaled = breadth_scaler.fit_transform(working[["TechnologyBreadth"]]).astype(np.float32)
    tech_stack = sparse.hstack(
        [tech_binary, sparse.csr_matrix(breadth_scaled)], format="csr", dtype=np.float32
    )
    tech_stack_names = [*tech_feature_names, "numeric__TechnologyBreadth"]
    tech_path = processed_dir / "X_tech_stack.npz"
    _save_sparse(tech_stack, tech_path)
    representation_paths["X_tech_stack"] = str(tech_path)

    feature_names = {
        "X_full": full_feature_names,
        "X_tech_stack": tech_stack_names,
        "numeric": numeric_columns,
        "categorical": categorical_names,
        "employment": employment_names,
        "technology": tech_feature_names,
    }
    write_json(feature_names, processed_dir / "feature_names.json")

    metadata_columns = [
        "ResponseId",
        "MainBranch",
        "DevType",
        "EdLevel",
        "AISelect",
        "ConvertedCompYearly",
        "Employment",
        "RemoteWork",
        "CountryOriginal",
        "Country",
        "YearsCodeNumeric",
        "YearsCodeProNumeric",
        "WorkExpNumeric",
        "ExperienceConsensus",
        "TechnologyBreadth",
    ]
    metadata_path = processed_dir / "respondent_metadata.parquet"
    working[metadata_columns].to_parquet(metadata_path, index=False)

    preprocessor = {
        "technology_encoders": tech_encoders,
        "employment_encoder": employment_encoder,
        "categorical_encoder": categorical_encoder,
        "scalers": scaler_models,
        "technology_breadth_scaler": breadth_scaler,
        "selected_scaler": selected_scaler,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names": feature_names,
    }
    preprocessor_path = artifacts_dir / "models" / "phase1_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    inventory = pd.DataFrame(
        [
            {"block": "numeric", "features": len(numeric_columns)},
            {"block": "categorical_one_hot", "features": len(categorical_names)},
            {"block": "employment_multi_hot", "features": len(employment_names)},
            {"block": "technology_multi_hot", "features": len(tech_feature_names)},
            {"block": "X_full_total", "features": len(full_feature_names)},
            {"block": "X_tech_stack_total", "features": len(tech_stack_names)},
        ]
    )
    inventory.to_csv(tables_dir / "feature_inventory.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": utc_now_iso(),
        "rows": int(len(working)),
        "selected_scaler": selected_scaler,
        "representations": {
            "X_full": [int(len(working)), int(len(full_feature_names))],
            "X_tech_stack": [int(len(working)), int(len(tech_stack_names))],
        },
        "technology_features": len(tech_feature_names),
        "employment_features": len(employment_names),
        "categorical_features": len(categorical_names),
        "numeric_features": len(numeric_columns),
        "paths": representation_paths,
        "metadata_path": str(metadata_path),
        "preprocessor_path": str(preprocessor_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
    }
    write_json(manifest, artifacts_dir / "feature_manifest.json")
    return manifest


def load_feature_names(config: dict[str, Any]) -> dict[str, list[str]]:
    path = project_path(config, "processed_dir") / "feature_names.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

