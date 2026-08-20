"""Prepare Cell Painting morphology matrices for one genotype/line workflow.

The input parquet files are expected to contain morphology features plus
``Metadata_line_ID`` and ``Metadata_genotype``. Machine-specific paths are kept
in YAML and are placeholders in the tracked configuration files.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from config_utils import load_config


def filter_line_genotype(df, line_id: str, genotype: str):
    line_mask = df["Metadata_line_ID"].astype(str).str.strip() == str(line_id)
    genotype_mask = (
        df["Metadata_genotype"].astype(str).str.strip().str.lower()
        == str(genotype).strip().lower()
    )
    return df.loc[line_mask & genotype_mask].copy()


def derive_feature_columns(dataframes: dict[str, pd.DataFrame], drop_patterns):
    common = set.intersection(*(set(df.columns) for df in dataframes.values()))
    candidates = sorted(
        column
        for column in common
        if not any(pattern in column.lower() for pattern in drop_patterns)
    )
    valid_sets = []
    for df in dataframes.values():
        numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
        valid_sets.append(set(numeric.columns[numeric.notna().any(axis=0)]))
    return sorted(set.intersection(*valid_sets))


def prepare_data(cfg: dict):
    states = cfg["states"]
    line_id = str(cfg["selection"]["line_id"])
    genotype = str(cfg["selection"]["genotype"])

    filtered = {}
    for state_name, state_cfg in states.items():
        df = pd.read_parquet(state_cfg["input_parquet"])
        subset = filter_line_genotype(df, line_id=line_id, genotype=genotype)
        if subset.empty:
            raise ValueError(
                f"No cells found for state={state_name}, line={line_id}, genotype={genotype}."
            )
        filtered[state_name] = subset
        print(f"{state_name}: {len(subset):,} cells")

    reference_path = cfg["features"].get("reference_feature_file")
    if reference_path:
        feature_names = pd.read_csv(reference_path)["feature"].astype(str).tolist()
        for name, df in filtered.items():
            missing = [f for f in feature_names if f not in df.columns]
            if missing:
                raise ValueError(
                    f"{name} is missing {len(missing)} reference morphology features; "
                    f"examples: {missing[:5]}"
                )
    else:
        feature_names = derive_feature_columns(
            filtered,
            tuple(str(x).lower() for x in cfg["features"]["drop_patterns"]),
        )

    expected = cfg["features"].get("expected_n_features")
    if expected is not None and len(feature_names) != int(expected):
        raise ValueError(
            f"Expected {expected} morphology features, found {len(feature_names)}."
        )

    output_dir = Path(cfg["paths"]["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_names, name="feature").to_csv(
        output_dir / "common_feature_columns.csv", index=False
    )

    for state_name, df in filtered.items():
        X = (
            df[feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        np.save(output_dir / f"{state_name}_X.npy", X)
        # Metadata required to trace the exact line/genotype selection.
        meta_cols = [c for c in ["Metadata_line_ID", "Metadata_genotype"] if c in df.columns]
        df[meta_cols].reset_index(drop=True).to_csv(
            output_dir / f"{state_name}_metadata.csv", index=False
        )

    return {name: str(output_dir / f"{name}_X.npy") for name in filtered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_data(load_config(args.config))


if __name__ == "__main__":
    main()
