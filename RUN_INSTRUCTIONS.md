# Submission — Federated Multiclass Network Intrusion Detection with CAA-FedCal

This package contains the complete research code, results, plots, and the IEEE LaTeX paper for the federated multiclass network intrusion detection project (Class-Absence-Aware Federated Calibration — CAA-FedCal).

## Package contents

| Path | Description |
|------|-------------|
| `scripts/` | Pipeline scripts: inspect, preprocess, partition, validate, visualize, experiments, sensitivity analysis |
| `src/` | Source packages: data pipeline, MLP models, FedAvg, CAA-FedCal calibration |
| `configs/` | YAML configuration files |
| `results/` | All experiment artifacts: metrics JSON, checkpoints (.pt), history JSON, figures (.png) |
| `paper/` | IEEEtran LaTeX paper source (`main.tex`), bibliography (`references.bib`), compiled `main.pdf`, figures, Makefile, README |
| `docs/` | Methodology, dataset, experiment protocol, novelty review, M4 audit findings |
| `M3_REPORT.md` | Main M3 milestone report |
| `README.md` | Project overview and quick start |
| `AGENTS.md` | Kilo coding-agent instructions / constraints |
| `requirements.txt` | Python package dependencies |
| `RUN_INSTRUCTIONS.md` | This file — how to download the dataset and run the pipeline |

## 1. Dataset download (required before running)

The pipeline uses **NF-UNSW-NB15-v2**, a single combined NetFlow v2 CSV from the University of Queensland (UQRDM). It is **not** the original UNSW-NB15 train/testing CSVs from UNSW Canberra.

**Official source record:**
- https://espace.library.uq.edu.au/view/UQ:ffbb0c1
- BagIt identifier: `fe6cb615d161452c_MOHANAD_A4706`
- RAiD: https://hdl.handle.net/102.100.100/388313

**Steps:**
1. Download the BagIt package `fe6cb615d161452c_MOHANAD_A4706.zip` from the official UQ source record above.
2. Extract it into `data/raw/unsw-nb15/` so the CSV lands at:
   `data/raw/unsw-nb15/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv`
3. (Optional companion file) `NetFlow_v2_Features.csv` in the same `data/` folder.

**File identity (verify after download):**
- `NF-UNSW-NB15-v2.csv` — 441,867,785 bytes
- SHA-256: `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`

**Do not** download from unofficial mirrors. **Do not** modify or overwrite raw dataset files. Raw data under `data/raw/` is never committed and is git-ignored.

The dataset has **2,390,275 rows × 45 columns**, with a multiclass `Attack` target (10 classes: Benign + 9 attack types). Benign ≈ 96.02%. The pipeline creates a reproducible stratified 80/20 train/test split by the `Attack` column (seed=42): 1,912,220 train / 478,055 test records.

Cite the UQ source record (https://espace.library.uq.edu.au/view/UQ:ffbb0c1) in any publication.

## 2. Environment setup

The project requires Python 3.13+ (system Python 3.14 is PEP 668 externally managed on Arch — do not use `--break-system-packages` or `sudo pip`).

```bash
# Create a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the pipeline (M1/M2, no model training)

Run in this exact order. Stop immediately if any step fails:

```bash
# 1. Run tests first
pytest -q

# 2. Inspect the raw dataset (requires the CSV from step 1 above)
python scripts/inspect_dataset.py

# 3. Preprocess (stratified 80/20 split, fit transformers on train only)
python scripts/preprocess.py

# 4. Non-IID partitioning (10 clients, Dirichlet alpha=0.3)
python scripts/partition.py

# 5. Validate data splits
python scripts/validate_data.py

# 6. Generate visualizations
python scripts/visualize_data.py
```

`validate_data.py` writes `results/data_validation.json` and exits non-zero on a failed check. Do not bypass validation failures.

## 4. Model training & optimization (M3)

After M1/M2 validation passes, training and evaluation scripts are available:

```bash
# Run the full M3 experiment suite (centralized + FedAvg + calibration)
python scripts/run_experiments.py

# Run the seed-42 optimization pilot (exploratory, single seed)
python scripts/optimization_pilot.py

# Sensitivity analysis on train/test feature overlap
python scripts/sensitivity_analysis.py
```

## 5. Outputs

Generated artifacts live under `data/processed`, `data/partitions`, and `results/`:

- `results/dataset_inspection.json` — dataset profile
- `results/data_validation.json` — validation status
- `results/partitions/partition_metadata.json` — client split metadata
- `results/m3/metrics/` — per-experiment metric JSON files (three seeds)
- `results/m3/checkpoints/` — model checkpoints (.pt) and training history (.json)
- `results/optimization/` — seed-42 optimization pilot metrics
- `results/figures/` — PNG figures: metric comparison, per-class F1, client class heatmap, convergence, global class distribution

No figure values are invented; all are rendered from saved experiment artifacts.

## 6. Paper

The IEEE conference-mode LaTeX paper is in `paper/`:

- `paper/main.tex` — manuscript source
- `paper/references.bib` — 16 verified references (all cited in-text)
- `paper/main.pdf` — compiled PDF (6 pages)
- `paper/figures/` — actual result figures copied from `results/m3/figures/`
- `paper/Makefile` — `pdflatex` + `bibtex` driver
- `paper/README.md` — compilation instructions

Compile with: `cd paper && make` (or `pdflatex main.tex && bibtex main && pdflatex main.tex x2`).

## 7. Key constraints (from AGENTS.md)

- Never use the official test set to fit preprocessing, select features, tune thresholds, or partition clients.
- Do not claim formal privacy or differential privacy.
- Do not claim novelty, state-of-the-art, or first-of-its-kind until the novelty gate is resolved.
- Keep deterministic seeds (42, 123, 7); write metadata for all transformations.
- Raw dataset files under `data/raw/` must never be committed or modified.

## Contact / provenance

Supervisor: Dr. Malay Kumar, Indian Institute of Information Technology Dharwad (IIIT Dharwad).
