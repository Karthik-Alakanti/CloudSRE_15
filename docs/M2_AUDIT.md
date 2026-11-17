# M2 Code Audit — Data Pipeline Readiness

Audit of `src/data/pipeline.py` and `scripts/` against the documented protocol in `docs/METHODOLOGY.md`, `docs/EXPERIMENT_PROTOCOL.md`, `AGENTS.md`, and `configs/experiment.yaml`.

**Constraint**: The official UNSW-NB15 CSVs are absent from `data/raw/unsw-nb15/`. No pipeline script has been executed against real data. All findings are static code review only.

## Protocol compliance checklist

| Protocol requirement | Implementation | Status |
|---|---|---|
| Target: `attack_cat` | `TARGET = "attack_cat"` (pipeline.py:17) | PASS |
| Exclude `attack_cat`, `label`, `id` from features | `NON_FEATURE_COLUMNS = {"attack_cat", "label", "id"}` (pipeline.py:18) | PASS |
| Exclude `srcip`, `dstip`, `Stime`, `Ltime` | `DEFAULT_EXCLUDED = {"srcip", "dstip", "stime", "ltime"}` (pipeline.py:19), case-insensitive matching | PASS |
| Preprocessing fitted on training only | `transformer.fit_transform(X_train_frame)` then `transformer.transform(X_test_frame)` (pipeline.py:149-150); metadata `test_used_for_fit: False` | PASS |
| Test transformed with training-fitted transformer | Same as above | PASS |
| All ten classes retained | `retain_all_classes: true` in config; no class dropping in code | PASS |
| Label encoding from training labels only | `classes = sorted(set(y_train_text))`; `label_to_id` from train; test maps through same map; fails on unseen test labels (pipeline.py:125-127) | PASS |
| Dirichlet partitioning α=0.3, 10 clients | `dirichlet_partition(y, num_clients=10, alpha=0.3, seed=42)` (partition.py:21-23) | PASS |
| Every training row assigned exactly once | Class-wise multinomial allocation conserves class rows (pipeline.py:216); validated by `validate_data.py` (lines 89-92) | PASS |
| Client train/validation/calibration disjoint | `split_client_indices` creates disjoint buckets (pipeline.py:246-273); validated by `validate_data.py` (lines 76-77, 91-92) | PASS |
| Calibration records separate from train/validation | Separate files per split; calibration never used for gradient updates (no training code exists) | PASS |
| Official test set untouched | Test file loaded but never partitioned; `preprocess_dataset` fits on train only | PASS |
| Deterministic seeds | `seed=42` throughout config; `rng = np.random.default_rng(seed)` | PASS |
| Record file hashes | `sha256_file` for train/test (pipeline.py:182-183) | PASS |
| Record feature names | `feature_names.json` (pipeline.py:172) | PASS |
| Record label mapping | `label_encoder.json` (pipeline.py:173) | PASS |
| Record client class distributions | `partition_metadata.json` with per-client class counts (partition.py:44-54) | PASS |

## Implementation bugs and methodological inconsistencies

### Bug 1: `dirichlet_partition` `min_size=1` can cause excessive retries with rare classes

**Location**: `partition.py:22` calls `dirichlet_partition(y, fed["num_clients"], fed["dirichlet_alpha"], fed["seed"], min_size=1)`.

**Issue**: With Dirichlet α=0.3 and UNSW-NB15, rare classes (e.g., Worms) may have very few samples. The multinomial allocation may assign 0 samples of a rare class to most clients. The `min_size=1` constraint requires every client to have ≥1 total sample (not per class), which is easy to satisfy. However, some clients will have 0 samples of rare classes. This is expected and correctly reported in metadata. The `min_size=1` constraint itself is not a bug, but the retry loop (up to 10,000 iterations) could be slow if the constraint is hard to satisfy. With α=0.3 and 175K rows, this is unlikely to be a problem in practice.

**Risk**: Low. The constraint is on total rows per client, not per-class. Rare-class absence is expected and reported.

### Bug 2: `split_client_indices` gives validation/calibration 0 samples for n=1 and n=2 classes

**Location**: `pipeline.py:250-253`.

**Issue**: When a class has only 1 sample in a client partition, `counts = [1, 0, 0]` — all to train, none to validation or calibration. When n=2, `counts = [1, 0, 1]` — validation gets 0. This means some clients have no validation or calibration samples for some classes.

**Impact**: The `validate_data.py` script warns when calibration count < 5 per class (line 84-86), but does not fail. This is acceptable for reporting but means some clients cannot estimate class-specific thresholds for absent-in-calibration classes.

**Risk**: Medium. This directly affects the calibration reliability for rare classes. The Candidate 1 direction (class-absence-aware calibration) explicitly handles this by only aggregating from clients that have the class in calibration.

### Bug 3: No train/test duplicate-record (leakage) check

**Location**: `scripts/inspect_dataset.py` checks `duplicate_rows` within each file but not across train/test. `scripts/validate_data.py` checks label overlap but not record overlap.

**Issue**: The stop conditions in `docs/EXPERIMENT_PROTOCOL.md` (line 42) list "duplicate records across train/test that could indicate leakage" as a reason to stop. No code implements this check.

**Risk**: Medium. UNSW-NB15 train/test are official partitions, so cross-file duplication is unlikely, but the check is missing as a safeguard.

### Bug 4: Missing reproducibility metadata (package versions)

**Location**: `pipeline.py:174-189` and `partition.py:24-30`.

**Issue**: `docs/METHODOLOGY.md` section "Reproducibility" (line 62-63) requires recording "package versions". The preprocessing metadata does not include sklearn/pandas/numpy/scipy versions. The partition metadata does not record the per-client split seeds (only the base `seed` from config; per-client seeds are derived as `split_cfg["seed"] + client_id` but not recorded).

**Risk**: Low-Medium. Reproducibility is partially addressed (seeds are fixed and deterministic), but version metadata is missing.

### Bug 5: `inspect_dataset.py` `test_only_labels` uses `set()` on a pandas Series

**Location**: `inspect_dataset.py:42`.

**Issue**: `set(normalize_labels(train.attack_cat))` — calling `set()` on a pandas Series iterates over values, which works correctly. However, `train.attack_cat` accesses the column via attribute access, which fails if the column name conflicts with a pandas method (e.g., "count", "sum"). Not a current issue since column is "attack_cat", but fragile.

**Risk**: Low.

### Bug 6: `visualize_data.py` class list may miss classes absent from all clients

**Location**: `visualize_data.py:31-32`.

**Issue**: `classes` is computed from the union of class counts across all clients. If a class is absent from every client's partition (extremely unlikely with α=0.3 but theoretically possible for very rare classes), it won't appear in the heatmap's class list, though `len(labels)` from `label_encoder.json` still provides the column count. The matrix would have all-zero columns for missing classes.

**Risk**: Low. Cosmetic only; heatmap would still render correctly with all-zero columns.

### Bug 7: Strict train/test schema match may be too brittle

**Location**: `pipeline.py:135-139`.

**Issue**: `preprocess_dataset` requires exact column match between train and test feature sets (missing or extra columns raise ValueError). If the official CSVs have any schema difference (e.g., different column ordering, an extra column in test), preprocessing fails hard.

**Risk**: Low-Medium. This is intentional fail-fast behavior, but could be a problem if the official files differ slightly from expectations.

### Bug 8: No check that `label` column is not used as a feature

**Location**: `validate_data.py:43-46`.

**Issue**: The forbidden set includes `label`, and `input_columns` from preprocessing metadata should exclude it. `build_preprocessor` drops `label` via `NON_FEATURE_COLUMNS`. This is correctly implemented, but `validate_data.py` only checks `input_columns` (raw feature names), not the transformed feature names. Transformed feature names from OneHotEncoder include the original column prefix (e.g., "proto_tcp"), so `label` would not appear in transformed names unless a raw column named "label" was included. This is correctly handled upstream.

**Risk**: Low. Correctly implemented.

### Bug 9: `split_client_indices` seed derivation not recorded in metadata

**Location**: `partition.py:38` and `partition.py:48-54`.

**Issue**: Per-client split uses `seed=split_cfg["seed"] + client_id`, but the partition metadata only records the federation seed, not the per-client split seeds. For full reproducibility, the per-client split seeds should be recorded.

**Risk**: Low. The derivation rule is deterministic and documented in code, but not in metadata.

### Bug 10: `validate_data.py` does not verify the official test set is excluded from fitting

**Location**: `validate_data.py:42`.

**Issue**: It checks `pmeta.get("test_used_for_fit") is not False`, which relies on the preprocessing metadata self-reporting. There is no independent verification that the test data was not used in fitting. This is a metadata-trust issue, not a code bug.

**Risk**: Low. The code correctly implements train-only fitting; the validation is a consistency check.

## Summary of required corrections

The following corrections are genuinely necessary before M2 validation can be trusted:

1. **Add train/test duplicate-record leakage check** to `scripts/inspect_dataset.py` or `scripts/validate_data.py` (required by EXPERIMENT_PROTOCOL stop condition).
2. **Add package version metadata** to preprocessing metadata (required by METHODOLOGY.md reproducibility section).
3. **Record per-client split seeds** in partition metadata (for full reproducibility).
4. **Document the n=1/n=2 calibration absence behavior** in METHODOLOGY.md or EXPERIMENT_PROTOCOL.md (known limitation, not a bug).

The following are not bugs but should be noted:
- `min_size=1` in `dirichlet_partition` means some clients will have 0 samples of rare classes — this is expected and reported.
- `split_client_indices` gives 0 validation/calibration samples for classes with n=1 or n=2 in a client — affects calibration reliability for rare classes, addressed by Candidate 1's class-absence-aware design.

## Readiness for M2 execution

**Not ready.** The official UNSW-NB15 CSVs are absent. Once placed at:
- `data/raw/unsw-nb15/UNSW_NB15_training-set.csv`
- `data/raw/unsw-nb15/UNSW_NB15_testing-set.csv`

The pipeline can be executed:
```
. .venv/bin/activate.fish
python -m pytest -q
python scripts/inspect_dataset.py
python scripts/preprocess.py
python scripts/partition.py
python scripts/validate_data.py
python scripts/visualize_data.py
```

Corrections 1-3 above should be implemented before running the pipeline to ensure validation is complete and metadata is reproducible.
