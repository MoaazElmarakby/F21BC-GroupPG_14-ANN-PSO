# psoann

A from-scratch **feed-forward ANN** trained by **Particle Swarm Optimisation** (Algorithm 39 of Luke,
*Essentials of Metaheuristics*) on the UCI **Banknote Authentication** task. No backpropagation, no
ANN/PSO libraries: NumPy only for the core (pandas/matplotlib for analysis).

The design contract, conventions and decisions log are in [docs/design.md](docs/design.md); the
literature behind the defaults is in [LITERATURE.md](LITERATURE.md).

## Quick start

```bash
pip install -e ".[dev]"     # numpy + pandas, matplotlib, pytest
psoann run                  # 10 runs with the default setup -> results/default.csv
```

That's it: it trains the network 10 times and prints the mean test accuracy. Then try a built-in
parameter study and look at it:

```bash
psoann sweep informants --runs 5                        # -> results/sweep_informants.csv
psoann summarise results/sweep_informants.csv --by point --metric test_accuracy --metric test_f1
psoann stats results/sweep_informants.csv --by point    # Kruskal-Wallis, Holm-corrected Mann-Whitney, A12
```

The full study plan (Q0 to Q8, 75 settings x 30 runs, one command each) is in [docs/STUDIES.md](docs/STUDIES.md);
`psoann sweep all` runs everything in a few minutes.

## Command reference

| Command | What it does |
|---|---|
| `psoann run [--runs N] [--seed S] [--set key=value ...]` | one experiment from the defaults; e.g. `--set pso.informants=ring --set ann.hidden_layers=[8,4]` |
| `psoann sweep NAME` / `psoann sweep all` | built-in parameter study (`psoann list` shows the names) |
| `psoann summarise CSV --by COLUMN [--metric M ...]` | mean/std/min/max per group |
| `psoann stats CSV --by point --metric M [--control NAME]` | Kruskal-Wallis, Holm-corrected Mann-Whitney U and A12 effect size |
| `psoann plot-sweep CSV --x COLUMN --out FILE.png` | metric against a swept column |
| `psoann plot-box CSV --by point --y M --out FILE.png` | spread over the repeated runs, one box per setting |
| `psoann plot-heatmap CSV --x COL --y COL --value M --out FILE.png` | mean of a metric over a two-factor grid |
| `psoann plot-convergence HISTORIES.jsonl --out FILE.png` | best-so-far, mean, diversity or boundary-hit curves (`--key`) |
| `psoann list` | registered activations, topologies, boundary handlers, coefficient schedules, metrics, sweeps |
| `python -m pytest` | the test suite (150+ tests, a few seconds) |

## What is implemented

| Brief | Where |
|---|---|
| ANN from scratch, configurable layers, widths, per-layer activations (logistic, ReLU, tanh, plus leaky ReLU, Gaussian, identity) | `ann/` |
| PSO = Algorithm 39 with informants, comments cite the book's line numbers | `pso/optimizer.py` |
| Coupling: a particle is the whole ANN as one flat vector; fitness = training accuracy | `training/` |
| Banknote data, stratified 70/30 split, train-only standardisation | `data/` |
| Metrics: accuracy, precision, recall, F1, MSE, evaluations-to-target | `evaluation/`, `experiments/runner.py` |
| Hyperparameter studies, 30 runs each, mean and standard deviation | `experiments/presets.py`, `docs/STUDIES.md` |
| **Going further:** activation functions evolved inside the particle (one gene per hidden layer) | `ann/network.py` (`EvolvedActivationNetwork`) |
| **Going further:** six boundary rules (clamp, reject, reflect, wrap, random, none) with boundary-hit logging | `pso/bounds.py` |
| **Going further:** time-varying coefficients (linearly decreasing inertia, TVAC) against fixed ones | `pso/schedules.py`, `psoann sweep schedule` |
| **Going further:** informant strategies (global, random fixed, random regenerated, nearest-neighbour, ring) | `pso/informants.py` |
| **Going further:** fitness-function comparison (error rate, MSE, cross-entropy) | `evaluation/metrics.py`, `psoann sweep fitness` |
| **Going further:** statistics (Kruskal-Wallis, Holm, A12), equal-budget random-search control, convergence and diversity logging | `analysis/stats.py`, `pso/random_search.py`, `pso/optimizer.py` |

Defaults: 4-8-1 network (49 parameters), tanh hidden / logistic output, 30 particles x 100 iterations
(3000 fitness evaluations), alpha=0.7298, beta=gamma=1.4962, delta=0, 3 random informants, weights
bounded to +-2, fitness = training accuracy. Typical result: test accuracy about 0.99.

## Layout

```
src/psoann/
  config.py        frozen, validated dataclasses (dotted overrides)
  registry.py      tiny name -> implementation registry used by every extension point
  data/            CSV loader, stratified split / k-fold, train-only standardiser
  ann/             activations, flat-vector codec, feed-forward network
  pso/             Algorithm 39 optimiser, informant topologies, boundary handling, coefficient schedules,
                   random-search control
  evaluation/      fitness and reporting metrics (higher is better)
  training/        fitness wrapper + PSOTrainer: the only place ANN and PSO meet
  experiments/     runner (seeding), sweeps + built-in presets, CSV/JSONL result files
  analysis/        summaries, statistics, plots
  cli.py           psoann run | sweep | summarise | stats | plot-* | list
data/raw/          banknote.csv (1372 rows, 4 features, 762 of class 0, 610 of class 1)
tests/             unit, integration and architecture (import rule) tests
docs/              design.md (contract), STUDIES.md (study plan), SUBMISSION.md, AI_USAGE.md
scripts/           make_submission.sh (builds the code zip)
```

`tests/test_architecture.py` enforces the dependency rule: `ann/` and `pso/` never import each other.

## Reproducibility

Every result row has the full configuration (flattened `cfg.*` columns plus `config_json`). Run `i` is
seeded from `SeedSequence(base_seed, spawn_key=(i,))`, so it is independent of `n_runs`, and any row can be
regenerated with `run_single(ExperimentConfig.from_json(row["config_json"]), row["run"])` (only the
`runtime_sec` column differs).

## Extending

Add an activation, informant topology, boundary handler or fitness metric by writing one small
function/class and registering it (table in docs/design.md section 4); select it by name in the config.

## Data

`data/raw/banknote.csv` is the UCI "Banknote Authentication" data (headerless CSV: variance, skewness,
kurtosis and entropy of a wavelet transform, then the class label 0/1). It was downloaded from a public
GitHub mirror of the UCI file; checked: 1372 rows, no missing values, class counts 762 (label 0) and 610
(label 1).
