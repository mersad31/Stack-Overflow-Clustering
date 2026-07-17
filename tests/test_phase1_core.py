from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.data import parse_experience, validate_source_schema
from stackoverflow_clustering.features import split_multilabel
from stackoverflow_clustering.tendency import hopkins_statistic, vat_order


def test_experience_parser_handles_boundary_labels() -> None:
    values = pd.Series(["Less than 1 year", "12", "More than 50 years", None, "bad"])
    result = parse_experience(values)
    assert result.iloc[0] == 0.5
    assert result.iloc[1] == 12.0
    assert result.iloc[2] == 51.0
    assert result.iloc[3:].isna().all()


def test_multilabel_parser_deduplicates_and_sorts() -> None:
    assert split_multilabel("Python; SQL;Python") == ["Python", "SQL"]
    assert split_multilabel("") == []
    assert split_multilabel(None) == []


def test_vat_order_is_a_permutation() -> None:
    points = np.array([[0.0], [0.1], [5.0], [5.1]])
    distances = np.abs(points - points.T)
    order = vat_order(distances)
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_hopkins_detects_separated_blobs() -> None:
    rng = np.random.default_rng(42)
    matrix = np.vstack(
        [rng.normal(-3, 0.15, size=(250, 2)), rng.normal(3, 0.15, size=(250, 2))]
    )
    assert hopkins_statistic(matrix, sample_size=100, random_state=42) > 0.75


def test_config_and_source_schema() -> None:
    config = load_config(ROOT / "config.yaml")
    info = validate_source_schema(config)
    assert info["source_columns"] == 114
    assert info["selected_columns"] == 21  # 18 analytical fields + 3 control/id fields
    assert len(config["data"]["clustering_columns"]) == 14
    assert len(config["data"]["supplementary_columns"]) == 4

