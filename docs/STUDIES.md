# Studies and how to reproduce the results

Every study is defined in `src/psoann/experiments/presets.py` and run with one command. All studies share
the following protocol:

* one **fixed stratified 70/30 split** (seed 0): 960 training and 412 test samples;
* **30 runs per setting**, with the same 30 seeds for every setting (the two pilot studies use 10 runs);
* **full-batch fitness on the training set only**; the test set is used for reporting and never for
  selecting a setting;
* a budget of 20 particles × 100 iterations = **2000 fitness evaluations**, unless the study varies it.

| Study | `psoann sweep …` | Settings | Question |
|---|---|---|---|
| Q0, control | `random_search` | 2 | Does PSO beat random sampling with the same evaluation budget? |
| Q1 | `architecture` | 13 | How do the number of layers and neurons affect accuracy? Widths 2, 4, 8, 16 × depths 1, 2, 3, plus a network with no hidden layer (5 to 641 parameters). |
| Q2 | `swarm_vs_iterations` | 12 | How should a fixed budget (500 or 2000 evaluations) be split between swarm size (5 to 100) and iterations? |
| Q3 | `coefficients` | 15 | What balance of personal-best, informant-best and global-best weights works best? All whole-number triples that sum to 4. |
| Q4 | `informants` | 16 | Do the number and the choice of informants matter? Global, random (fixed and regenerated) and nearest-neighbour informants with k = 1, 2, 3, 5, 10. |
| Q5 | `bounds` | 6 | Does the boundary rule matter? clamp, reject, reflect, wrap, random re-initialisation, none. |
| Q6 | `activations` | 5 | Fixed or evolved activation functions? Two hidden layers of 8 with one fixed activation (logistic, tanh, ReLU or leaky ReLU) in every hidden layer, or one evolved gene per hidden layer that chooses among logistic, tanh and ReLU (`ann.activation_choices`; leaky ReLU is evaluated only as a fixed activation). |
| Q7 | `fitness` | 3 | Does the fitness function matter? Error rate (accuracy), MSE, cross-entropy. |
| Q8 | `schedule` | 3 | Do time-varying coefficients (linearly decreasing inertia, TVAC) beat fixed ones? |
| pilots | `pilot_bound`, `pilot_velocity` | 5, 2 | Hidden hyperparameters (weight bound, initial velocity), chosen on training accuracy only. |

The nine studies Q0 to Q8 have 75 settings (2250 runs); with the pilots a full run is 2320 runs.

## Reproducing the results

Python 3.11 or newer is required.

```bash
pip install -e ".[dev]"
python -m pytest                              # unit, integration and architecture tests
psoann sweep all --out results                # trains and evaluates every study (about 6 to 9 minutes)
python scripts/make_outputs.py                # statistics tables and one boxplot per study
python scripts/plot_main_results.py           # figures for Q1 to Q4 (architecture, budget, coefficients, informants)
python scripts/plot_diversity_and_boundary_hits.py     # swarm diversity (Q4) and boundary hits (Q5)
python scripts/plot_boundary_rule_significance.py      # pairwise significance of the boundary rules (Q5)
python scripts/plot_boundary_rule_significance.py --study schedule    # the same for the schedules (Q8)
```

`scripts/run_all.sh` runs the whole sequence. Runs are seeded, so the same command reproduces the same
numbers; only the `runtime_sec` column differs.

## Output files

| Path | Content |
|---|---|
| `results/sweep_<study>.csv` | one row per run: accuracy, precision, recall, F1, MSE (training and test), evaluations to the final best and to the target, the full configuration |
| `results/sweep_<study>.histories.jsonl` | per-iteration best, mean, iteration best, diversity and boundary hits of every run |
| `results/stats/*.txt` | Kruskal-Wallis, Holm-corrected Mann-Whitney U and A12 for every study and for each study against its control setting |
| `results/figures/*.png` | the figures listed above and one boxplot per study |

## Reading the results

Every row of a results file is one run; group by the `point` column (the label of the setting).

| Need | Command |
|---|---|
| mean, standard deviation, minimum, maximum | `psoann summarise results/sweep_architecture.csv --by point --metric test_accuracy --metric test_f1 --metric reached_target --metric evals_to_target` |
| are two settings different? | `psoann stats results/sweep_informants.csv --by point --metric test_accuracy` (add `--control random-k3` to compare one setting with the rest) |
| spread over the runs | `psoann plot-box results/sweep_bounds.csv --by point --y test_f1 --out bounds_box.png` |
| two-factor grid | `psoann plot-heatmap results/sweep_swarm_vs_iterations.csv --x cfg.pso.swarm_size --y evaluations --value test_accuracy --out budget.png` |
| convergence, diversity, boundary hits | `psoann plot-convergence results/sweep_bounds.histories.jsonl --key boundary_hits --out hits.png` |

## Metrics

Accuracy and F1 (forged banknote as the positive class) are reported on the training and the test set;
precision, recall and MSE are in the CSV files. Because final accuracy is close to its ceiling on this
data set, convergence speed is reported as well: `evals_to_target` is the number of evaluations until the
best training accuracy first reaches `target_fitness` (0.98), and `reached_target` is the share of runs
that got there. `evals_to_target` averages only the runs that reached the target, so it should be read
together with `reached_target`. It is not available for the MSE and cross-entropy fitness functions,
because the target is defined on accuracy.

## Notes

* The data set has 24 duplicate rows (1.7%), which are kept as specified.
* The `random` boundary rule redraws only the offending coordinate, whereas the lecture strategy
  re-initialises the whole particle.
* The schedule values (inertia 0.9 to 0.4; TVAC 2.5 to 0.5 and 0.5 to 2.5) are constants at the top of
  `src/psoann/pso/schedules.py`.
