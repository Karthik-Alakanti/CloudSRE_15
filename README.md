# Federated IDS Calibration — Research Starter

This repository contains the **M1/M2 foundation only**: research decisions, dataset acquisition guidance, preprocessing, Non-IID partitioning, validation, visualizations, and tests.

**It does not train a model.** Do not begin model training until `validate_data.py` passes and the research gates in `docs/` are reviewed.

## Current research question (provisional)

Can client-side calibration summaries be aggregated to choose class-specific decision thresholds for a federated multiclass network IDS, improving minority-class detection while respecting a benign false-positive constraint under Non-IID data?

The proposed mechanism is exploratory. It is **not yet a verified novelty claim** and must pass the literature/feasibility gate in `docs/RESEARCH_PROBLEM.md`.

## Dataset: NF-UNSW-NB15-v2 (University of Queensland)

This project uses **NF-UNSW-NB15-v2**, a NetFlow v2 rendering of UNSW-NB15 created at the University of Queensland (UQRDM) by M. Sarhan. It is **not** the original UNSW-NB15 training/testing partition CSVs from UNSW Canberra.

The dataset has been downloaded and placed at:

```text
data/raw/unsw-nb15/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv
```

File identity:
- Size: 441,867,785 bytes
- SHA-256: `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`
- Source record: https://espace.library.uq.edu.au/view/UQ:ffbb0c1

NF-UNSW-NB15-v2 is a **single combined file** with no official train/test partition. The pipeline creates a reproducible stratified 80/20 train/test split by the `Attack` column (seed=42).

See `docs/DATASET.md` for the full dataset description, differences from original UNSW-NB15, class distribution, and data quality notes.

## Quick start (Arch Linux, Fish shell)

### 1. Create and activate the virtual environment

System Python 3.14.10 (`/usr/bin/python`) is externally managed (PEP 668) and, on this Arch system, its `pyexpat` extension is mismatched against the installed `libexpat`, so `python -m venv` fails during `ensurepip` and `pip install` crashes in the wheel installer. **Do not** use `--break-system-packages`, `sudo pip`, or modify system Python.

Python 3.13 (`/home/shreezn/.local/bin/python3.13`) is installed and works. Create the project venv from it:

```fish
/usr/bin/python3.13 -m venv --without-pip .venv
# bootstrap pip (read-only copy from Python 3.13's site-packages; system Python untouched)
cp -r /home/shreezn/.local/lib/python3.13/site-packages/pip .venv/lib/python3.13/site-packages/pip
cp -r /home/shreezn/.local/lib/python3.13/site-packages/pip-26.3.dist-info .venv/lib/python3.13/site-packages/
printf '#!/home/shreezn/repos/clsre2/fed-ids-calibration/.venv/bin/python\n# -*- coding: utf-8 -*-\nimport re\nimport sys\nfrom pip._internal.cli.main import main\nif __name__ == "__main__":\n    sys.argv[0] = re.sub(r"(-script\\.pyw|\\.exe)?$", "", sys.argv[0])\n    sys.exit(main())\n' > .venv/bin/pip
chmod +x .venv/bin/pip
ln -sf pip .venv/bin/pip3
```

Activate with Fish and install dependencies:

```fish
. .venv/bin/activate.fish
python -m pip --version
python -m pip install -r requirements.txt
```

Verify `python` now resolves to the venv interpreter (`which python` → `.venv/bin/python`, Python 3.13).

### 2. Run the M1/M2 pipeline (dataset must be present first)

```fish
python -m pytest -q
python scripts/inspect_dataset.py
python scripts/preprocess.py
python scripts/partition.py
python scripts/validate_data.py
python scripts/visualize_data.py
```

The pipeline stops before any training. `validate_data.py` writes `results/data_validation.json` and exits non-zero on a failed check. If the dataset file is absent, `inspect_dataset.py` reports the missing file and the pipeline stops there — see `docs/DATASET.md`.

### 3. Environment notes

- The venv is isolated (`include-system-site-packages = false`) and all generated artifacts live under `data/processed`, `data/partitions`, and `results`.
- System Python is never modified and no packages are installed into it.
- Raw dataset files under `data/raw/**` are never committed and are git-ignored.

## Dataset handling rules

- Target: `Attack` (multiclass, 10 classes). `Label` is a redundant binary target and is excluded from model inputs.
- `id` column does not exist in NF-UNSW-NB15-v2.
- IP addresses `IPV4_SRC_ADDR` and `IPV4_DST_ADDR` are excluded by default to reduce identity/time leakage.
- There are no `Stime`/`Ltime` timestamp columns in NF-UNSW-NB15-v2.
- The test split is never used to fit imputers, encoders, scalers, partitioning, or calibration thresholds.
- All ten labels are retained initially, including rare Worms (164 records). Do not remove `Worms` just because it is rare; inspect its counts and decide from evidence.
- The feature transformer is fitted on the training split only. This is a reproducible benchmark simulation; it is **not** a fully decentralized preprocessing protocol. The limitation is documented.
- Client partitions are made from the processed training split after preprocessing. Each client's train/validation/calibration split is local.
- The test split remains a final held-out test set and is not used by this pipeline for tuning.

## Output files

```text
data/processed/
  X_train.npz, y_train.npy, X_test.npz, y_test.npy
  feature_names.json, label_encoder.json, preprocessing_metadata.json, preprocessor.joblib
data/partitions/
  client_00_train.npz, client_00_train_y.npy, client_00_train_indices.npy, ...
  partition_metadata.json
results/
  dataset_inspection.json
  data_validation.json
  figures/
    global_class_distribution.png
    client_class_heatmap.png
```

## Important

NF-UNSW-NB15-v2 differs from the original UNSW-NB15: it is a single NetFlow v2 CSV from the University of Queensland, not the UNSW Canberra train/testing partition CSVs. Feature names, target column names, and row counts differ. Cite the UQ source record (https://espace.library.uq.edu.au/view/UQ:ffbb0c1) in any publication.

Never commit raw/processed data, virtual environments, model checkpoints, or large result files. `.gitignore` is provided.
