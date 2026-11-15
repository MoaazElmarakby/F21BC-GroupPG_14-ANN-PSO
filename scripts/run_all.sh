#!/usr/bin/env bash
# Whole pipeline from a clean checkout: install, test, run every study (30 runs each), statistics, figures.
#   scripts/run_all.sh            about 6 minutes
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e ".[dev]"
python -m pytest
psoann sweep all --out results          # training + evaluation of every study (2200 runs)
python scripts/make_outputs.py          # stats tables and figures
echo "done: results/*.csv, results/stats/*.txt, results/figures/*.png"
