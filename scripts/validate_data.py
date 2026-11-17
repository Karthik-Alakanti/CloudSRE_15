#!/usr/bin/env python3
from pathlib import Path
import sys, json, yaml
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]

def fail(errors, message):
    errors.append(message)

def main():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    proc = ROOT / cfg["outputs"]["processed_dir"]
    part_dir = ROOT / cfg["outputs"]["partitions_dir"]
    results = ROOT / cfg["outputs"]["results_dir"]
    errors, warnings = [], []

    required = [
        proc/"X_train.npz", proc/"X_test.npz", proc/"y_train.npy", proc/"y_test.npy",
        proc/"preprocessing_metadata.json", proc/"label_encoder.json",
        part_dir/"partition_metadata.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        fail(errors, "Missing generated artifacts: " + ", ".join(missing))
    else:
        Xtr = sparse.load_npz(proc/"X_train.npz").tocsr()
        Xte = sparse.load_npz(proc/"X_test.npz").tocsr()
        ytr = np.load(proc/"y_train.npy")
        yte = np.load(proc/"y_test.npy")
        pmeta = json.loads((proc/"preprocessing_metadata.json").read_text())
        lmap = json.loads((proc/"label_encoder.json").read_text())
        partmeta = json.loads((part_dir/"partition_metadata.json").read_text())

        if Xtr.shape[0] != len(ytr): fail(errors, "Training X/y row count mismatch.")
        if Xte.shape[0] != len(yte): fail(errors, "Test X/y row count mismatch.")
        if Xtr.shape[1] != Xte.shape[1]: fail(errors, "Train/test feature dimension mismatch.")
        if Xtr.shape[1] != pmeta.get("transformed_features"): fail(errors, "Feature dimension disagrees with preprocessing metadata.")
        if not np.isfinite(Xtr.data).all(): fail(errors, "Training matrix contains non-finite values.")
        if not np.isfinite(Xte.data).all(): fail(errors, "Test matrix contains non-finite values.")
        if pmeta.get("test_used_for_fit") is not False: fail(errors, "Metadata does not confirm test was excluded from fitting.")
        forbidden = {"attack_cat", "label", "id", "srcip", "dstip", "stime", "ltime"}
        actual_inputs = {str(x).lower() for x in pmeta.get("input_columns", [])}
        bad = sorted(actual_inputs & forbidden)
        if bad: fail(errors, f"Forbidden raw columns present in model inputs: {bad}")
        if len(lmap) != len(set(lmap.values())): fail(errors, "Label mapping contains duplicate encoded IDs.")
        if set(np.unique(ytr)) != set(range(len(lmap))): fail(errors, "Training labels do not match encoded label map.")
        if not set(np.unique(yte)).issubset(set(np.unique(ytr))): fail(errors, "Test has unseen encoded labels.")

        # Verify exact one-time assignment and disjoint local splits using saved row IDs.
        # Current partition artifacts store matrices only; reconstruct assignment metadata is checked via totals.
        expected_clients = int(cfg["federation"]["num_clients"])
        if int(partmeta.get("num_clients", -1)) != expected_clients:
            fail(errors, "Partition metadata client count mismatch.")
        total = 0
        all_assigned_indices = []
        for client_id in range(expected_clients):
            key = f"{client_id:02d}"
            if key not in partmeta.get("clients", {}):
                fail(errors, f"Missing client metadata: {key}")
                continue
            cm = partmeta["clients"][key]
            total += int(cm["total_rows"])
            for split in ("train", "validation", "calibration"):
                x_path = part_dir / f"client_{key}_{split}.npz"
                y_path = part_dir / f"client_{key}_{split}_y.npy"
                i_path = part_dir / f"client_{key}_{split}_indices.npy"
                if not x_path.is_file() or not y_path.is_file() or not i_path.is_file():
                    fail(errors, f"Missing client {key} {split} files (including source indices).")
                    continue
                Xc = sparse.load_npz(x_path)
                yc = np.load(y_path)
                idx = np.load(i_path)
                all_assigned_indices.extend(idx.tolist())
                if len(idx) != len(yc): fail(errors, f"Client {key} {split}: index/label mismatch.")
                if len(np.unique(idx)) != len(idx): fail(errors, f"Client {key} {split}: duplicate source indices.")
                if np.any(idx < 0) or np.any(idx >= len(ytr)): fail(errors, f"Client {key} {split}: source index out of range.")
                elif len(idx) == len(yc) and not np.array_equal(yc, ytr[idx]): fail(errors, f"Client {key} {split}: labels do not match source indices.")
                if Xc.shape[0] != len(yc): fail(errors, f"Client {key} {split}: X/y mismatch.")
                if Xc.shape[1] != Xtr.shape[1]: fail(errors, f"Client {key} {split}: feature dimension mismatch.")
                if not np.isfinite(Xc.data).all(): fail(errors, f"Client {key} {split}: non-finite values.")
                if len(yc) != int(cm[f"{split}_rows"]): fail(errors, f"Client {key} {split}: metadata row mismatch.")
            for cls, count in cm["class_counts"]["calibration"].items():
                if count < 5:
                    warnings.append(f"Client {key} has only {count} calibration examples for encoded class {cls}.")
        if total != len(ytr):
            fail(errors, f"Client partition total {total} != training rows {len(ytr)}.")
        if len(all_assigned_indices) != len(ytr):
            fail(errors, f"Assigned source-index count {len(all_assigned_indices)} != training rows {len(ytr)}.")
        elif len(set(all_assigned_indices)) != len(ytr) or set(all_assigned_indices) != set(range(len(ytr))):
            fail(errors, "Client splits do not form an exact, non-overlapping partition of all training rows.")
        if partmeta.get("source_train_rows") != len(ytr):
            fail(errors, "Partition source row count mismatch.")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "training_started": False,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results/"data_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    print("\nDATA VALIDATION PASSED. Training remains disabled pending M1 novelty approval and user sign-off.")

if __name__ == "__main__":
    main()
