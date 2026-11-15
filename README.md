# psoann

A from-scratch **feed-forward ANN** trained by **Particle Swarm Optimisation** (Algorithm 39 of Luke,
*Essentials of Metaheuristics*) on the UCI **Banknote Authentication** task. No backpropagation, no
ANN/PSO libraries: NumPy only for the core (pandas/matplotlib for analysis).

The design contract, conventions and decisions log are in [docs/design.md](docs/design.md); the
literature behind the defaults is in [LITERATURE.md](LITERATURE.md).

## Quick start

```bash
pip install -e ".[dev]"            # numpy + pandas, matplotlib, pytest
make test                          # 100+ tests, < 2 s

psoann run configs/default.toml                        # 10 runs -> results/default.csv (+ .histories.jsonl)
psoann run configs/default.toml --runs 3 --seed 7 \
    --set pso.informants=ring --set ann.hidden_layers=[8,4]
psoann sweep configs/sweeps/informants.toml            # parameter study -> results/sweep_informants.csv
psoann summarise results/sweep_informants.csv --by cfg.pso.n_informants
psoann plot-sweep results/sweep_informants.csv --x cfg.pso.n_informants --y test_accuracy --out informants.png
psoann plot-convergence results/default.histories.jsonl --out convergence.png
psoann list                                            # registered activations, topologies, ...
```

Defaults: 4-8-1 network (49 parameters), tanh hidden / logistic output, 30 particles x 100 iterations
(3000 fitness evaluations), alpha=0.7298, beta=gamma=1.4962, delta=0, 3 random informants, weights
bounded to +-2, fitness = training accuracy. Typical result: test accuracy about 0.99.

## Layout

```
src/psoann/
  config.py        frozen, validated dataclasses (+ TOML loading, dotted overrides)
  registry.py      tiny name -> implementation registry used by every extension point
  data/            CSV loader, stratified split / k-fold, train-only standardiser
  ann/             activations, flat-vector codec, feed-forward network
  pso/             Algorithm 39 optimiser, informant topologies, boundary handling
  evaluation/      fitness metrics (higher is better)
  training/        fitness wrapper + PSOTrainer: the only place ANN and PSO meet
  experiments/     runner (seeding), sweeps, CSV/JSONL result files
  analysis/        summaries and plots
  cli.py           psoann run | sweep | summarise | plot-convergence | plot-sweep | list
configs/           default.toml and sweep definitions (configs/sweeps/*.toml)
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
