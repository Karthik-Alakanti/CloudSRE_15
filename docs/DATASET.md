# Dataset acquisition and handling — NF-UNSW-NB15-v2

## Dataset identity

This project uses **NF-UNSW-NB15-v2**, a NetFlow v2 rendering of the UNSW-NB15 dataset created by Mohanad Sarhan at the University of Queensland (UQRDM). It is **not** the original UNSW-NB15 train/test partition CSVs from UNSW Canberra.

| Property | Value |
|---|---|
| Dataset | NF-UNSW-NB15-v2 |
| Source | University of Queensland, School of IT and Electrical Engineering |
| Creator | M. Sarhan (s4608505@uq.edu.au, ORCID 0000-0001-8597-6042) |
| Source record | https://espace.library.uq.edu.au/view/UQ:ffbb0c1 |
| BagIt identifier | fe6cb615d161452c_MOHANAD_A4706 |
| RAiD | https://hdl.handle.net/102.100.100/388313 |

## Key differences from the original UNSW-NB15

| | Original UNSW-NB15 | NF-UNSW-NB15-v2 |
|---|---|---|
| Provider | UNSW Canberra | University of Queensland (UQRDM) |
| File layout | Separate official training + testing CSVs | Single combined NetFlow v2 CSV |
| Target column | `attack_cat` (multiclass) | `Attack` (multiclass) |
| Binary label column | `label` | `Label` |
| Identifier column | `id` | none |
| IP columns | `srcip`, `dstip` | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` |
| Timestamp columns | `Stime`, `Ltime` | none |
| Total rows | ~257,673 (175,341 train + 82,332 test) | 2,390,275 (single file) |
| Classes | Normal + 9 attack types | Benign + 9 attack types (10 total) |

## Files used

Downloaded into `data/raw/unsw-nb15/fe6cb615d161452c_MOHANAD_A4706/data/`:

- `NF-UNSW-NB15-v2.csv` — 441,867,785 bytes, SHA-256 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`
- `NetFlow_v2_Features.csv` — 2,115 bytes, SHA-256 `f20e86a939f9eb53b06ace15de4cc977ed61c02e538c6465efe72b635448e939`

Do not download from unofficial mirrors. Do not overwrite or modify raw dataset files.

## Train/test split

NF-UNSW-NB15-v2 is a single combined file with no official train/test partition. The pipeline creates a **reproducible stratified 80/20 train/test split** by the `Attack` column (seed=42, `test_fraction=0.20`), so every class — including Worms (164 records) — appears in both splits. Preprocessing is fit on the training split only; the test split is transformed with the training-fitted transformer and never used for fitting, partitioning, or calibration.

## Verification

Run:
```bash
python scripts/inspect_dataset.py
```

The script reports rows, columns, labels, missingness, duplicates, infinities, file SHA-256, and cross-split duplicate-record leakage.

## Class distribution (full dataset)

| Class | Count |
|---|---|
| Benign | 2,295,222 |
| Exploits | 31,551 |
| Fuzzers | 22,310 |
| Generic | 16,560 |
| Reconnaissance | 12,779 |
| DoS | 5,794 |
| Analysis | 2,299 |
| Backdoor | 2,169 |
| Shellcode | 1,427 |
| Worms | 164 |

All ten classes are retained. Worms is the rarest class (0.0069% of records).

## Data quality notes

- Missing values: 0
- Infinite values: 0
- Exact duplicate rows: 0
- Duplicate feature rows (excluding labels): 14,488 rows across 2,450 feature-row groups with conflicting `Attack` labels (known label noise in UNSW-NB15-derived data). Zero groups have conflicting binary `Label` values.
- 40 unique source IPs and 40 unique destination IPs.

## Dataset citations

Cite the dataset source record at https://espace.library.uq.edu.au/view/UQ:ffbb0c1 and any related UQ/NB15 publications. Do not cite the original UNSW-NB15 papers as the source of this specific file; they describe the original UNSW-NB15, not the NF-UNSW-NB15-v2 NetFlow rendering.

## Caveats

- NF-UNSW-NB15-v2 is a NetFlow v2 aggregation of UNSW-NB15 flows, not the original UNSW-NB15 row-level records. Feature names and semantics differ from the original 49-feature UNSW-NB15 schema.
- The label `Attack` is the multiclass target. `Label` is a binary target and must not be an input feature.
- IP address columns (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`) are excluded from model inputs by default.
- Keep all attack categories initially, including rare Worms. Any exclusion must be justified and approved after inspecting counts.
