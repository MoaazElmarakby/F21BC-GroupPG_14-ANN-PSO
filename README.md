# psoann

A feed-forward **artificial neural network** trained by **particle swarm optimisation** (Algorithm 39 of
Luke, *Essentials of Metaheuristics*), applied to the UCI **Banknote Authentication** task. Both the network
and the optimiser are implemented from scratch with NumPy: there is no backpropagation and no ANN or PSO
library. pandas, Matplotlib and SciPy are used only for analysis and plotting.

Design and conventions are described in [docs/design.md](docs/design.md), the studies in
[docs/STUDIES.md](docs/STUDIES.md), the sources in [LITERATURE.md](LITERATURE.md), and the use of AI tools
in [docs/AI_USAGE.md](docs/AI_USAGE.md).

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # numpy, pandas, matplotlib, scipy, pytest, ruff
python -m pytest                   # unit, integration and architecture tests
```

## Quick start

```bash
psoann run --out demo                          # 10 runs with the default setup -> demo/default.csv
psoann sweep informants --runs 5 --out demo    # a built-in study               -> demo/sweep_informants.csv
psoann summarise demo/sweep_informants.csv --by point --metric test_accuracy --metric test_f1
psoann stats demo/sweep_informants.csv --by point
```

Defaults: a 4-8-1 network (49 parameters) with tanh hidden units and a logistic output, 30 particles × 100
iterations (3000 fitness evaluations), α = 0.7298, β = γ = 1.4962, δ = 0, three random informants, weights
bounded to ±2, and the training accuracy as fitness.

## Reproducing the reported results

```bash
psoann sweep all --out results                          # every study, 30 runs per setting
python scripts/make_outputs.py                          # statistics tables and boxplots
python scripts/plot_main_results.py                     # figures for architecture, budget, coefficients, informants
python scripts/plot_diversity_and_boundary_hits.py      # swarm diversity and boundary hits
python scripts/plot_boundary_rule_significance.py       # significance of the boundary rules
python scripts/plot_boundary_rule_significance.py --study schedule
```

`scripts/run_all.sh` runs the whole sequence. Runs are seeded, so the same commands give the same numbers.
The results of this sequence are included in [results/](results/); see [docs/STUDIES.md](docs/STUDIES.md)
for the studies and the output files.

## Command reference

| Command | Purpose |
|---|---|
| `psoann run [--runs N] [--seed S] [--set key=value ...]` | one experiment from the defaults, for example `--set pso.informants=ring --set ann.hidden_layers=[8,4]` |
| `psoann sweep NAME` / `psoann sweep all` | a built-in study (`psoann list` shows the names) |
| `psoann summarise CSV --by COLUMN [--metric M ...]` | mean, standard deviation, minimum and maximum per group |
| `psoann stats CSV --by point --metric M [--control NAME]` | Kruskal-Wallis, Holm-corrected Mann-Whitney U and the A12 effect size |
| `psoann plot-sweep CSV --x COLUMN --out FILE.png` | a metric against a swept column |
| `psoann plot-box CSV --by point --y M --out FILE.png` | the spread over the repeated runs, one box per setting |
| `psoann plot-heatmap CSV --x COL --y COL --value M --out FILE.png` | the mean of a metric over a two-factor grid |
| `psoann plot-convergence HISTORIES.jsonl --out FILE.png` | best, mean, diversity or boundary-hit curves (`--key`) |
| `psoann list` | the registered activations, informant strategies, boundary rules, schedules, metrics and studies |

## Features

| Area | Features | Module |
|---|---|---|
| ANN | configurable layers, widths and per-layer activations (logistic, tanh, ReLU, leaky ReLU, Gaussian, identity); activation functions evolved inside the particle | `ann/` |
| PSO | Algorithm 39 with informants; informant strategies (global, random fixed or regenerated, nearest neighbour, ring); six boundary rules (clamp, reject, reflect, wrap, random, none); coefficient schedules (constant, linearly decreasing inertia, TVAC); equal-budget random-search control | `pso/` |
| Coupling | a particle is the whole network as one flat vector; the fitness is the training accuracy (or MSE, cross-entropy) | `training/` |
| Data | stratified 70/30 split, standardisation with training statistics only | `data/` |
| Evaluation | accuracy, precision, recall, F1, MSE, evaluations to a target, swarm diversity, boundary hits | `evaluation/`, `experiments/runner.py` |
| Experiments | seeded repeated runs, sweeps and the built-in studies | `experiments/` |
| Analysis | summaries, non-parametric statistics, plots | `analysis/`, `scripts/` |

## Repository layout

```
src/psoann/
  config.py        frozen, validated configuration dataclasses with dotted overrides
  registry.py      name -> implementation registry used by every extension point
  data/            CSV loader, stratified split, training-only standardiser
  ann/             activations, flat-vector codec, feed-forward network
  pso/             Algorithm 39 optimiser, informants, boundary rules, schedules, random-search control
  evaluation/      fitness and reporting metrics (higher is better)
  training/        fitness wrapper and PSOTrainer: the only place where ANN and PSO meet
  experiments/     runner (seeding), sweeps, built-in studies, CSV and JSONL result files
  analysis/        summaries, statistics, plots
  cli.py           the psoann command
data/raw/          banknote.csv (1372 rows, 4 features; 762 of class 0 and 610 of class 1)
tests/             unit, integration and architecture tests
scripts/           reproduction of the results, statistics and figures
results/           results of the studies (CSV, JSONL, statistics, figures)
docs/              design, studies, AI usage
```

`tests/test_architecture.py` enforces the dependency rule: `ann/` and `pso/` never import each other.

## Reproducibility

Every result row has the full configuration (flattened `cfg.*` columns plus `config_json`). Run `i` is seeded
from `SeedSequence(base_seed, spawn_key=(i,))`, so it does not depend on `n_runs`, and any row can be
regenerated with `run_single(ExperimentConfig.from_json(row["config_json"]), row["run"])`; only the
`runtime_sec` column differs.

## Extending

An activation, informant strategy, boundary rule, coefficient schedule or fitness metric is added by writing
one small function or class and registering it (see section 4 of [docs/design.md](docs/design.md)); it is
then selected by name in the configuration.

## Data

`data/raw/banknote.csv` is the UCI Banknote Authentication data set (Lohweg, 2012): a headerless CSV with the
variance, skewness, kurtosis and entropy of a wavelet transform of each image, followed by the class label
(0 or 1). It has 1372 rows and no missing values; 24 rows are duplicates and are kept.
