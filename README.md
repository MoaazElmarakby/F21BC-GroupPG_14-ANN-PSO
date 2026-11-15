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
psoann sweep topology                                   # -> results/sweep_topology.csv
psoann summarise results/sweep_topology.csv --by cfg.pso.informants
```

## Command reference

| Command | What it does |
|---|---|
| `psoann run [--runs N] [--seed S] [--set key=value ...]` | one experiment from the defaults; e.g. `--set pso.informants=ring --set ann.hidden_layers=[8,4]` |
| `psoann sweep NAME` / `psoann sweep all` | built-in parameter study (`psoann list` shows the names) |
| `psoann summarise CSV --by COLUMN` | mean/std per group |
| `psoann plot-sweep CSV --x COLUMN --out FILE.png` | metric against a swept column |
| `psoann plot-convergence HISTORIES.jsonl --out FILE.png` | best-so-far fitness curves |
| `psoann list` | registered activations, topologies, boundary handlers, metrics, sweeps |
| `python -m pytest` | the test suite (100+ tests, < 2 s) |

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
  pso/             Algorithm 39 optimiser, informant topologies, boundary handling
  evaluation/      fitness metrics (higher is better)
  training/        fitness wrapper + PSOTrainer: the only place ANN and PSO meet
  experiments/     runner (seeding), sweeps + built-in presets, CSV/JSONL result files
  analysis/        summaries and plots
  cli.py           psoann run | sweep | summarise | plot-convergence | plot-sweep | list
data/raw/          banknote.csv (1372 rows, 4 features, 762 of class 0, 610 of class 1)
tests/             unit, integration and architecture (import rule) tests
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
