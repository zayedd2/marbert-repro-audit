#!/usr/bin/env bash
# CPU-only smoke test of the WHOLE pipeline with fake data and a fake model (about 1-2 minutes).
# Everything goes to data_synthetic/ and results_synthetic/ and is watermarked SYNTHETIC.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_synthetic_data.py --data-root data_synthetic
python scripts/data_audit.py --data-root data_synthetic --results-root results_synthetic
python scripts/run_experiments.py --backend dummy --data-root data_synthetic --results-root results_synthetic
python scripts/verify_results.py --results-root results_synthetic
python scripts/make_report_assets.py --results-root results_synthetic --data-root data_synthetic
echo "Dry run OK. Synthetic outputs: results_synthetic/ (NOT real results)."
