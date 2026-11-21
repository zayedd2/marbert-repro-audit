#!/usr/bin/env bash
# CPU-only smoke test of the whole analysis pipeline with FAKE runs in the team's Colab format (about 1 minute).
# Everything goes to results_synthetic/ and is watermarked SYNTHETIC; nothing touches results/.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_synthetic_runs.py --out results_synthetic/team_outputs --epochs 4
python scripts/import_team_runs.py --team-outputs results_synthetic/team_outputs --results-root results_synthetic
python scripts/verify_results.py --results-root results_synthetic
python scripts/make_report_assets.py --results-root results_synthetic
echo "Dry run OK. Synthetic outputs: results_synthetic/ (NOT real results)."
