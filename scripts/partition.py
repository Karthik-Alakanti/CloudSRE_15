#!/usr/bin/env python3
from pathlib import Path
import sys, json, yaml
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pipeline import dirichlet_partition, split_client_indices

def main():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    proc = ROOT / cfg["outputs"]["processed_dir"]
    part_dir = ROOT / cfg["outputs"]["partitions_dir"]
    part_dir.mkdir(parents=True, exist_ok=True)

    y = np.load(proc / "y_train.npy")
    X = sparse.load_npz(proc / "X_train.npz").tocsr()
    fed = cfg["federation"]
    split_cfg = cfg["client_split"]
    partitions = dirichlet_partition(
        y, fed["num_clients"], fed["dirichlet_alpha"], fed["seed"], min_size=1
    )
    metadata = {
        "num_clients": fed["num_clients"],
        "dirichlet_alpha": fed["dirichlet_alpha"],
        "seed": fed["seed"],
        "source_train_rows": int(len(y)),
        "clients": {},
    }

    for client_id, indices in enumerate(partitions):
        tr, va, ca = split_client_indices(
            indices, y,
            split_cfg["train_fraction"],
            split_cfg["validation_fraction"],
            split_cfg["calibration_fraction"],
            seed=split_cfg["seed"] + client_id,
        )
        for name, idx in (("train", tr), ("validation", va), ("calibration", ca)):
            sparse.save_npz(part_dir / f"client_{client_id:02d}_{name}.npz", X[idx])
            np.save(part_dir / f"client_{client_id:02d}_{name}_y.npy", y[idx])
            np.save(part_dir / f"client_{client_id:02d}_{name}_indices.npy", idx)
        counts = {}
        for name, idx in (("train", tr), ("validation", va), ("calibration", ca)):
            values, n = np.unique(y[idx], return_counts=True)
            counts[name] = {str(int(v)): int(c) for v, c in zip(values, n)}
        metadata["clients"][f"{client_id:02d}"] = {
            "total_rows": int(len(indices)),
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "calibration_rows": int(len(ca)),
            "split_seed": int(split_cfg["seed"] + client_id),
            "class_counts": counts,
        }

    (part_dir / "partition_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print("\nPartitioning complete. No model training was performed.")

if __name__ == "__main__":
    main()
