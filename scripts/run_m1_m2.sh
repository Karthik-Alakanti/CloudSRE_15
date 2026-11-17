#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q
python scripts/inspect_dataset.py
python scripts/preprocess.py
python scripts/partition.py
python scripts/validate_data.py
python scripts/visualize_data.py

echo
echo "M1/M2 scripts completed. No model training was run."
echo "Review results/data_validation.json and docs/NOVELTY_REVIEW.md."
