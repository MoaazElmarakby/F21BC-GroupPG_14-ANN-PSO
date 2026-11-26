#!/usr/bin/env bash
# Whole pipeline from a clean checkout: install, test, run every study (30 runs per setting),
# then the statistics tables and the figures. Takes about 6 to 9 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -e ".[dev]"
python -m pytest

psoann sweep all --out results                                       # training and evaluation of every study
python scripts/make_outputs.py                                       # statistics tables and boxplots
python scripts/plot_main_results.py                                  # architecture, budget, coefficients, informants
python scripts/plot_diversity_and_boundary_hits.py                   # swarm diversity and boundary hits
python scripts/plot_boundary_rule_significance.py                    # significance of the boundary rules
python scripts/plot_boundary_rule_significance.py --study schedule   # significance of the schedules

echo "done: results/*.csv, results/stats/*.txt, results/figures/*.png"
