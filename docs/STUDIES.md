# Study plan: what to run, in which order, and which column answers which question

All studies use the **same fixed stratified 70/30 split** (seed 0), **30 runs per setting with the same
seeds across settings**, full-batch fitness on the **training** set only, and a budget of 20 particles x
100 iterations = **2000 fitness evaluations** (unless the study varies the budget). The test set is only
ever used to report; nothing is selected on it. Everything is one command each; nothing is hidden in code:

```bash
pip install -e ".[dev]"
psoann sweep all --out results            # about 5-6 minutes on a laptop; or one study at a time:
psoann sweep architecture --out results
```

| Study | `psoann sweep ...` | Settings | Question | Hypothesis to test (and where it comes from) |
|---|---|---|---|---|
| Q0 control | `random_search` | 2 | Does PSO beat random sampling at the same budget? | Sanity check: the swarm should clearly beat sampling |
| Q1 (required) | `architecture` | 13 | What effect do layers and neurons have? | Larger nets need more evaluations; accuracy against parameter count (5 to 641) shows where PSO struggles (lecture slide 8: about 100 dimensions) |
| Q2 (required) | `swarm_vs_iterations` | 12 | How best to spend a fixed budget (500 and 2000 evals)? | A mid-size swarm beats both extremes (the brief's 10x50 against 50x10 example) |
| Q3 (required) | `coefficients` | 15 | Social against cognitive balance? | Slide 14: too much beta is egocentric, too much gamma stagnates, too much delta converges early. Pilot first: beta+gamma+delta = 4 with alpha 0.7298 may be unstable |
| Q4 | `informants` | 16 | Do the number and choice of informants matter? | Slide 9: using the whole population converges prematurely. Strategies: global, random fixed, random regenerated, nearest-neighbour; k = 1, 2, 3, 5, 10 |
| Q5 | `bounds` | 5 | Does the boundary rule matter? | Slide 26 invites exploring it; `boundary_hits` explains differences |
| Q6 | `activations` | 5 | Fixed or evolved activation function? | The brief promises extra marks for activations inside the particle. Compare against the best fixed choice (2 hidden layers of 8) |
| Q7 | `fitness` | 3 | Does the fitness function matter? | The error-rate landscape has plateaus; smooth losses may help. Compare on test accuracy and F1, **not** on each other's `best_fitness` values |
| pilots | `pilot_bound`, `pilot_velocity` | 5, 2 | Hidden hyperparameters | Choose weight bound and initial velocity on **training** fitness only, then justify each in one line |

71 settings x 30 runs = 2130 runs for Q0 to Q7.

## Reading the results

Every row of `results/sweep_<name>.csv` is one run. Group by the `point` column (the label of the setting).

| Need | Command |
|---|---|
| mean / std / min / max table | `psoann summarise results/sweep_architecture.csv --by point --metric test_accuracy --metric test_f1 --metric reached_target --metric evals_to_target` |
| is the difference real? | `psoann stats results/sweep_informants.csv --by point --metric test_accuracy` (Kruskal-Wallis, Holm-corrected Mann-Whitney, A12 effect size); add `--control global` to compare one baseline with the rest |
| spread over the 30 runs | `psoann plot-box results/sweep_bounds.csv --by point --y test_f1 --out fig/bounds_box.png` |
| two-factor grid | `psoann plot-heatmap results/sweep_swarm_vs_iterations.csv --x cfg.pso.swarm_size --y cfg.pso.iterations --value test_accuracy --out fig/budget.png` |
| convergence | `psoann plot-convergence results/sweep_x.histories.jsonl --out fig/conv.png` |
| diversity (explains premature convergence) | `... --key diversity` |
| boundary hits (explains the Q5 differences) | `... --key boundary_hits` |

Metrics: accuracy and F1 (forged = positive class) in the main text; precision, recall, MSE and
convergence speed are in the CSV. `evals_to_target` is the evaluations until the best training accuracy
first reaches `target_fitness` (0.98 in the presets); `reached_target` is the share of runs that got
there. Because final accuracy sits near the ceiling on this dataset, convergence speed is usually what
separates settings, so use it.

## Things to decide yourselves (the code deliberately does not hide them)

* The `target_fitness` of 0.98 is a placeholder: run a pilot and move it if every setting (or none) reaches it.
* `pilot_bound` and `pilot_velocity` set the weight bound (default 2) and the initial velocity (default
  half the vector between two random points). Judge them on training fitness and keep the choice in the report.
* The dataset has 24 duplicate rows (1.7 percent) that are kept as specified; mention possible train/test
  leakage as a limitation.
* The five boundary rules are clamp, reflect, wrap, random re-initialisation and none. Check them against
  the names on the boundary-handling lecture slide and rename or swap one if the lecture lists a different set.
