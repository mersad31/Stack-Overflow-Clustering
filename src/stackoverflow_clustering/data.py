from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_directories, project_path
from .io_utils import sha256_file, utc_now_iso, write_json


EXPERIENCE_SPECIAL = {
    "Less than 1 year": 0.5,
    "More than 50 years": 51.0,
}

ORG_SIZE_ORDINAL = {
    "Just me - I am a freelancer, sole proprietor, etc.": 1.0,
    "2 to 9 employees": 5.5,
    "10 to 19 employees": 14.5,
    "20 to 99 employees": 59.5,
    "100 to 499 employees": 299.5,
    "500 to 999 employees": 749.5,
    "1,000 to 4,999 employees": 2999.5,
    "5,000 to 9,999 employees": 7499.5,
    "10,000 or more employees": 10000.0,
}


def _root(config: dict[str, Any]) -> Path:
    return Path(config["_project_root"])


def _survey_path(config: dict[str, Any]) -> Path:
    return _root(config) / config["paths"]["survey_csv"]


def _schema_path(config: dict[str, Any]) -> Path:
    return _root(config) / config["paths"]["schema_csv"]


def validate_source_schema(config: dict[str, Any]) -> dict[str, Any]:
    """Validate source files and required columns before expensive processing."""
    survey_path = _survey_path(config)
    schema_path = _schema_path(config)
    if not survey_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError("The survey CSV and schema CSV must exist in the project root.")

    header = pd.read_csv(survey_path, nrows=0).columns.tolist()
    required = config["data"]["selected_columns"]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Required columns missing from survey CSV: {missing}")

    schema = pd.read_csv(schema_path)
    schema_required = {"qname", "question", "type", "selector"}
    if not schema_required.issubset(schema.columns):
        raise ValueError("survey_results_schema.csv has an unexpected schema.")

    return {
        "survey_path": str(survey_path),
        "schema_path": str(schema_path),
        "source_columns": len(header),
        "selected_columns": len(required),
        "survey_sha256": sha256_file(survey_path),
        "schema_sha256": sha256_file(schema_path),
    }


def ingest_selected(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read selected columns in chunks and persist an immutable selected raw layer."""
    ensure_output_directories(config)
    source_info = validate_source_schema(config)
    survey_path = _survey_path(config)
    selected = config["data"]["selected_columns"]
    chunk_size = int(config["data"]["chunk_size"])

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        survey_path,
        usecols=selected,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)

    if frame.empty or frame["ResponseId"].isna().all():
        raise ValueError("Ingestion produced an empty or invalid dataset.")

    interim_path = project_path(config, "interim_dir") / "selected_raw.parquet"
    frame.to_parquet(interim_path, index=False)
    manifest = {
        **source_info,
        "created_at_utc": utc_now_iso(),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "chunk_size": chunk_size,
        "selected_raw_path": str(interim_path),
        "selected_raw_sha256": sha256_file(interim_path),
    }
    write_json(manifest, project_path(config, "artifacts_dir") / "ingestion_manifest.json")
    return frame, manifest


def parse_experience(series: pd.Series) -> pd.Series:
    """Convert Stack Overflow experience strings to numeric years."""
    text = series.astype("string").str.strip()
    mapped = text.map(EXPERIENCE_SPECIAL)
    numeric = pd.to_numeric(text, errors="coerce")
    result = mapped.fillna(numeric).astype("float64")
    return result.where(result.between(0, 51))


def _decision(
    column: str,
    issue: str,
    action: str,
    rationale: str,
    affected_rows: int,
) -> dict[str, Any]:
    return {
        "column": column,
        "issue": issue,
        "action": action,
        "rationale": rationale,
        "affected_rows": int(affected_rows),
    }


def build_data_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact, report-ready data profile."""
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "non_null": frame.notna().sum(),
            "missing_count": frame.isna().sum(),
            "missing_pct": (100 * frame.isna().mean()).round(3),
            "unique_non_null": frame.nunique(dropna=True),
        }
    ).rename_axis("column").reset_index()


def clean_selected(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply documented clustering-oriented cleaning and cohort selection."""
    df = frame.copy()
    decisions: list[dict[str, Any]] = []
    initial_rows = len(df)

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        original = df[column]
        cleaned = original.astype("string").str.strip()
        cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
        changed = int(((original.astype("string") != cleaned) & original.notna()).sum())
        df[column] = cleaned
        decisions.append(
            _decision(
                column,
                "whitespace/type consistency",
                "strip text and map empty strings to missing",
                "Prevents duplicate categories caused by formatting drift.",
                changed,
            )
        )

    duplicate_mask = df.duplicated(subset=["ResponseId"], keep="first")
    decisions.append(
        _decision(
            "ResponseId",
            "duplicate respondents",
            "keep first occurrence",
            "Duplicate records artificially inflate local density.",
            int(duplicate_mask.sum()),
        )
    )
    df = df.loc[~duplicate_mask].copy()

    failed_attention = df["Check"].notna() & ~df["Check"].str.casefold().eq("apples")
    decisions.append(
        _decision(
            "Check",
            "failed attention check",
            "exclude explicit failures; retain missing checks",
            "Explicit failures indicate unreliable responses; missing is not assumed failure.",
            int(failed_attention.sum()),
        )
    )
    df = df.loc[~failed_attention].copy()

    df["CountryOriginal"] = df["Country"]
    df["Country"] = df["Country"].fillna("Missing")
    country_counts = df["Country"].value_counts(dropna=False)
    min_count = int(config["data"]["country_min_count"])
    rare_countries = set(country_counts[country_counts < min_count].index) - {"Missing"}
    rare_mask = df["Country"].isin(rare_countries)
    df.loc[rare_mask, "Country"] = "Other"
    decisions.append(
        _decision(
            "Country",
            "high cardinality / rare categories",
            f"group countries with fewer than {min_count} respondents as Other",
            "Limits sparse one-hot dimensions while retaining common geographic context.",
            int(rare_mask.sum()),
        )
    )

    for source in ("YearsCode", "YearsCodePro"):
        target = f"{source}Numeric"
        df[target] = parse_experience(df[source])
        invalid = int((df[source].notna() & df[target].isna()).sum())
        decisions.append(
            _decision(
                source,
                "numeric values stored as strings and boundary labels",
                f"parse to {target}; Less than 1 year=0.5, More than 50 years=51",
                "A continuous experience scale is required for distance-based clustering.",
                invalid,
            )
        )

    df["WorkExpNumeric"] = pd.to_numeric(df["WorkExp"], errors="coerce")
    df["WorkExpNumeric"] = df["WorkExpNumeric"].where(df["WorkExpNumeric"].between(0, 50))

    experience_numeric = ["YearsCodeNumeric", "YearsCodeProNumeric", "WorkExpNumeric"]
    for column in experience_numeric:
        missing_flag = f"{column}Missing"
        imputed = f"{column}Imputed"
        df[missing_flag] = df[column].isna().astype("int8")
        group_median = df.groupby("Country", observed=True)[column].transform("median")
        global_median = float(df[column].median())
        df[imputed] = df[column].fillna(group_median).fillna(global_median)
        decisions.append(
            _decision(
                column,
                "missing experience",
                f"country-median imputation with global median fallback; retain {missing_flag}",
                "Missingness is preserved explicitly and imputation avoids artificial zero-experience clusters.",
                int(df[column].isna().sum()),
            )
        )

    df["ExperienceConsensus"] = df[experience_numeric].median(axis=1, skipna=True)
    df["ExperienceConsensus"] = df["ExperienceConsensus"].fillna(
        df["ExperienceConsensus"].median()
    )
    df["ExperienceSpread"] = df[experience_numeric].max(axis=1, skipna=True) - df[
        experience_numeric
    ].min(axis=1, skipna=True)
    df["ExperienceSpread"] = df["ExperienceSpread"].fillna(0.0)
    df["ProfessionalExperienceRatio"] = (
        df["YearsCodeProNumericImputed"] / df["YearsCodeNumericImputed"].clip(lower=0.5)
    ).clip(0, 2)

    df["OrgSizeMissing"] = df["OrgSize"].isna().astype("int8")
    df["OrgSizeNumeric"] = df["OrgSize"].map(ORG_SIZE_ORDINAL)
    org_median = float(df["OrgSizeNumeric"].median())
    df["OrgSizeNumeric"] = df["OrgSizeNumeric"].fillna(org_median)
    df["OrgSizeLog"] = np.log1p(df["OrgSizeNumeric"])
    decisions.append(
        _decision(
            "OrgSize",
            "ordinal ranges and missing/unknown values",
            "map ranges to midpoints, median-impute unknowns, then log1p",
            "Preserves order while reducing the extreme range of organisation sizes.",
            int(df["OrgSizeMissing"].sum()),
        )
    )

    tech_columns = config["data"]["technology_columns"]
    tech_breadth_columns: list[str] = []
    for column in tech_columns:
        df[f"{column}Missing"] = df[column].isna().astype("int8")
        df[column] = df[column].fillna("")
        prefix = column.replace("HaveWorkedWith", "")
        breadth_column = f"{prefix}Breadth"
        df[breadth_column] = df[column].map(
            lambda value: len({part.strip() for part in str(value).split(";") if part.strip()})
        ).astype("int16")
        tech_breadth_columns.append(breadth_column)
    df["TechnologyBreadth"] = df[tech_breadth_columns].sum(axis=1).astype("int16")
    any_tech_response = df[[f"{c}Missing" for c in tech_columns]].sum(axis=1) < len(
        tech_columns
    )
    excluded_no_tech = int((~any_tech_response).sum())
    if config["data"].get("require_any_technology_response", True):
        df = df.loc[any_tech_response].copy()
    decisions.append(
        _decision(
            "technology matrix",
            "no response in all seven technology domains",
            "exclude rows with all seven domains missing",
            "The research question cannot be answered without any technology-stack evidence.",
            excluded_no_tech,
        )
    )

    for column in ("Employment", "RemoteWork", "DevType", "EdLevel", "AISelect"):
        df[column] = df[column].fillna("Missing")

    invalid_comp = df["ConvertedCompYearly"].notna() & (df["ConvertedCompYearly"] <= 0)
    df.loc[invalid_comp, "ConvertedCompYearly"] = np.nan
    decisions.append(
        _decision(
            "ConvertedCompYearly",
            "non-positive compensation",
            "set to missing; exclude from clustering feature space",
            "Compensation is supplementary and invalid values must not distort downstream regression.",
            int(invalid_comp.sum()),
        )
    )

    output_path = project_path(config, "interim_dir") / "cleaned_cohort.parquet"
    df.to_parquet(output_path, index=False)

    decisions_frame = pd.DataFrame(decisions)
    tables_dir = project_path(config, "tables_dir")
    decisions_frame.to_csv(tables_dir / "cleaning_decision_log.csv", index=False, encoding="utf-8-sig")
    build_data_profile(frame).to_csv(
        tables_dir / "data_profile_raw.csv", index=False, encoding="utf-8-sig"
    )
    build_data_profile(df).to_csv(
        tables_dir / "data_profile_cleaned.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "created_at_utc": utc_now_iso(),
        "rows_initial": int(initial_rows),
        "rows_cleaned": int(len(df)),
        "rows_removed": int(initial_rows - len(df)),
        "explicit_attention_failures_removed": int(failed_attention.sum()),
        "duplicate_response_ids_removed": int(duplicate_mask.sum()),
        "all_technology_domains_missing_removed": excluded_no_tech,
        "rare_country_threshold": min_count,
        "rare_country_categories_grouped": int(len(rare_countries)),
        "cleaned_path": str(output_path),
        "cleaned_sha256": sha256_file(output_path),
    }
    write_json(summary, project_path(config, "artifacts_dir") / "cleaning_summary.json")
    return df, decisions_frame, summary
