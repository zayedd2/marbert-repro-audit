#!/usr/bin/env bash
# Full pipeline on a GPU machine (e.g. Google Colab). Safe to re-run: finished runs are skipped.
#   bash scripts/run_all.sh                       # everything
#   RESUME_ROOT=/content/drive/MyDrive/ckpt bash scripts/run_all.sh   # keep per-epoch checkpoints on Drive
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6 prepare data"
python scripts/prepare_data.py all
echo "== 2/6 data audit"
python scripts/data_audit.py
echo "== 3/6 historical protocol (18 runs)"
python scripts/run_experiments.py --protocols historical ${RESUME_ROOT:+--resume-root "$RESUME_ROOT"}
echo "== 4/6 independent protocol (18 runs)"
python scripts/run_experiments.py --protocols independent ${RESUME_ROOT:+--resume-root "$RESUME_ROOT"}
echo "== 5/6 integrity check"
python scripts/verify_results.py
echo "== 6/6 tables, figures and report numbers"
python scripts/make_report_assets.py
echo "Done. See results/SUMMARY.md and report/generated/."
