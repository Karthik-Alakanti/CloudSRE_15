#!/usr/bin/env python3
from pathlib import Path
import sys, yaml, json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pipeline import preprocess_dataset

def main():
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    ds = config["dataset"]
    metadata = preprocess_dataset(
        ROOT / ds["raw_dir"],
        ROOT / config["outputs"]["processed_dir"],
        data_file=ds.get("data_file"),
        train_file=ds.get("train_file"),
        test_file=ds.get("test_file"),
        target_column=ds.get("target_column", "attack_cat"),
        binary_label_column=ds.get("binary_label_column", "label"),
        excluded_columns=ds.get("excluded_feature_columns", []),
        test_fraction=ds.get("test_fraction", 0.2),
        split_seed=ds.get("split_seed", 42),
    )
    print(json.dumps(metadata, indent=2))
    print("\nPreprocessing complete. No model training was performed.")

if __name__ == "__main__":
    main()
