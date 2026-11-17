#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pipeline import (
    load_csv_pair,
    sha256_file,
    normalize_labels,
    stratified_train_test_split,
)

RAW = ROOT / "data/raw/unsw-nb15"


def summarize(path: Path | None, frame: pd.DataFrame, label: str) -> dict:
    target = "Attack" if "Attack" in frame.columns else "attack_cat"
    target_counts = normalize_labels(frame[target]).value_counts().sort_index().to_dict()
    numeric = frame.select_dtypes(include=["number"])
    return {
        "path": str(path) if path is not None else None,
        "label": label,
        "sha256": sha256_file(path) if path is not None else None,
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "infinite_cells_numeric": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "class_counts": {str(k): int(v) for k, v in target_counts.items()},
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
    }


def main():
    config_path = ROOT / "configs/experiment.yaml"
    import yaml

    config = yaml.safe_load(config_path.read_text())
    ds = config["dataset"]

    data_file = ds.get("data_file")
    train_file = ds.get("train_file")
    test_file = ds.get("test_file")
    target_column = ds.get("target_column", "attack_cat")
    test_fraction = ds.get("test_fraction", 0.2)
    split_seed = ds.get("split_seed", 42)

    result = {
        "dataset": ds.get("name", "UNSW-NB15"),
        "source": ds.get("source"),
        "source_url": ds.get("source_url"),
        "raw_dir": str(RAW),
    }

    if data_file:
        data_path = RAW / data_file
        missing = [str(data_path)] if not data_path.is_file() else []
        if missing:
            raise FileNotFoundError(
                "Missing dataset file:\n  " + "\n  ".join(missing)
                + "\nSee docs/DATASET.md for acquisition instructions."
            )
        df = pd.read_csv(data_path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if target_column not in df.columns:
            raise ValueError(
                f"Target column {target_column!r} not found in {data_file}. "
                f"Columns: {df.columns.tolist()}"
            )

        result["data_file"] = data_file
        result["data_size_bytes"] = int(data_path.stat().st_size)
        result["data_sha256"] = sha256_file(data_path)

        full = summarize(data_path, df, "full_dataset")
        result["full_dataset"] = full

        # Reproducible stratified train/test split.
        train, test, split_meta = stratified_train_test_split(
            df, target_column, test_fraction=test_fraction, seed=split_seed
        )
        result["split"] = split_meta

        result["train"] = summarize(None, train, "train_split")
        result["test"] = summarize(None, test, "test_split")

        # Cross-split duplicate-record leakage check.
        train_key = train.assign(__source="train")
        test_key = test.assign(__source="test")
        combined = pd.concat([train_key, test_key], ignore_index=True)
        dup_cols = [c for c in combined.columns if c != "__source"]
        cross_dups = combined[combined.duplicated(subset=dup_cols, keep=False)]
        result["cross_split_duplicate_rows"] = int(cross_dups.shape[0])

        # Label consistency between splits.
        train_labels = set(normalize_labels(train[target_column]))
        test_labels = set(normalize_labels(test[target_column]))
        result["train_labels"] = sorted(train_labels)
        result["test_labels"] = sorted(test_labels)
        result["test_only_labels"] = sorted(test_labels - train_labels)
        result["train_only_labels"] = sorted(train_labels - test_labels)

        # Duplicate feature rows with conflicting labels (known label noise).
        feature_cols = [c for c in df.columns if c not in (target_column, ds.get("binary_label_column", "label"))]
        dup_mask = df.duplicated(subset=feature_cols, keep=False)
        dups = df[dup_mask]
        if len(dups) > 0:
            grp = dups.groupby(feature_cols)[target_column].nunique()
            result["duplicate_feature_rows"] = int(len(dups))
            result["duplicate_feature_groups_with_conflicting_labels"] = int((grp > 1).sum())
            result["duplicate_feature_groups_with_conflicting_binary_label"] = int(
                (dups.groupby(feature_cols)[ds.get("binary_label_column", "label")].nunique() > 1).sum()
            )
        else:
            result["duplicate_feature_rows"] = 0
            result["duplicate_feature_groups_with_conflicting_labels"] = 0
            result["duplicate_feature_groups_with_conflicting_binary_label"] = 0

    elif train_file and test_file:
        train, test, train_path, test_path = load_csv_pair(RAW, train_file, test_file)
        result["train_file"] = train_file
        result["test_file"] = test_file
        result["train"] = summarize(train_path, train, "train")
        result["test"] = summarize(test_path, test, "test")
        train_key = train.assign(__source="train")
        test_key = test.assign(__source="test")
        combined = pd.concat([train_key, test_key], ignore_index=True)
        dup_cols = [c for c in combined.columns if c != "__source"]
        cross_dups = combined[combined.duplicated(subset=dup_cols, keep=False)]
        result["cross_split_duplicate_rows"] = int(cross_dups.shape[0])
        train_labels = set(normalize_labels(train[target_column]))
        test_labels = set(normalize_labels(test[target_column]))
        result["test_only_labels"] = sorted(test_labels - train_labels)
    else:
        raise ValueError("Config must provide either data_file or train_file+test_file.")

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_inspection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if result.get("test_only_labels"):
        raise SystemExit("ERROR: test contains labels absent from train.")
    if result.get("cross_split_duplicate_rows", 0) > 0:
        print(
            f"WARNING: {result['cross_split_duplicate_rows']} duplicate records found across train/test splits. Investigate provenance."
        )
    print("\nInspection complete. This script does not train a model.")


if __name__ == "__main__":
    main()
