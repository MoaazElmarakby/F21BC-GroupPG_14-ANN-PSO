# AI usage log

This file records how AI tools were used in this coursework, as the brief requires. AI was used to write and test code and to explain concepts. The report text was written by the group.

## Tools

| Tool | Model | Used by | When | Branch |
|---|---|---|---|---|
| Claude Code (Anthropic), cloud session | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | ANN owner | November 2026 | `claude/wonderful-johnson-f6h13v` |
| Claude (Anthropic), Claude app | `claude-opus-5-5`, `claude-fable-5-2` for in-depth technical questions | PSO owner | November 2026 | `pso-schedules-reject` |


## Code written with

Claude Code (ANN owner):

| Area | Files | What the AI did | How we checked it |
|---|---|---|---|
| Repository base | `src/psoann/` (config, data, ann, pso, training, experiments, analysis, cli), `tests/`, `docs/design.md`, `README.md` | First implementation of the ANN, the Algorithm 39 PSO, their coupling, the experiment harness and the tests | TO COMPLETE |
| Metrics | `evaluation/metrics.py`, `training/fitness.py` | Precision, recall and F1 (forged = positive) | TO COMPLETE |
| Boundary handling | `pso/bounds.py`, `pso/optimizer.py` | `wrap` and `random` handlers; boundary-hit counter | TO COMPLETE |
| Informants | `pso/informants.py`, `pso/optimizer.py` | `nearest` (k nearest by position, recomputed each iteration) | TO COMPLETE |
| Evolved activations | `ann/network.py`, `config.py`, `training/trainer.py` | One gene per hidden layer selects the activation; bounds per dimension | TO COMPLETE |
| Random-search control | `pso/random_search.py`, `config.py`, `training/trainer.py` | Equal-budget control optimiser | TO COMPLETE |
| Studies | `experiments/presets.py`, `experiments/sweep.py` | Q0 to Q7 study definitions and explicit sweep points | TO COMPLETE |
| Statistics and plots | `analysis/stats.py`, `analysis/plots.py`, `cli.py`, `scripts/make_outputs.py`, `scripts/plot_main_results.py` | Kruskal-Wallis, Holm-corrected Mann-Whitney, A12, boxplot and heatmap; the statistics files and the architecture, budget, coefficient and informant figures (report Figures 1 and 2) | TO COMPLETE |
| Tests and docs | `tests/test_extensions.py`, `docs/STUDIES.md` | Tests for the above and the description of the studies | TO COMPLETE |
| Final cleanup (26 November 2026) | all of `src/`, `tests/`, `scripts/`, `README.md`, `docs/`, `LITERATURE.md`, `pyproject.toml` | Formatting and lint rules (`ruff`); removed unused code (`stratified_kfold`, `Standardizer.fit_transform`) and internal decision tags from comments; split the two plotting scripts from the PSO owner into functions without changing their output; rewrote the README, design, studies and literature documents; removed the planning files and the design PDF | TO COMPLETE |


Claude (PSO owner):

| Area | Files | What the AI did | How we checked it |
|---|---|---|---|
| Coefficient schedules | `pso/schedules.py`, `pso/optimizer.py`, `config.py` (`pso.schedule`) | `constant`, `linear_inertia`, `tvac`; the optimiser asks the schedule for alpha, beta, gamma, delta before each velocity update | Ran the test suite (167 passed). Ran `psoann run` with `pso.schedule=tvac` and the Q8 study ourselves |
| Boundary handling | `pso/bounds.py`, `pso/optimizer.py` | `reject` (the lecture's strategy 2); handlers also receive the positions before the move | Ran the test suite. Ran `psoann run` with `pso.bound_handling=reject` and the Q5 study ourselves; checked the boundary-hit curves against the rule's behaviour |
| Studies | `experiments/presets.py`, `scripts/make_outputs.py` | `reject` added to the Q5 `bounds` study; new Q8 `schedule` study | Ran both studies; the five existing boundary rules gave the same means as before the change |
| Report figures | `scripts/plot_diversity_and_boundary_hits.py`, `scripts/plot_boundary_rule_significance.py` | Figure 3 (diversity and boundary hits) and the pairwise-significance charts, drawn from our result files | Ran both scripts on our results; compared the printed means and p-values with `psoann summarise` and `psoann stats` |
| Tests and docs | `tests/test_schedules.py`, `tests/test_extensions.py`, `docs/design.md` (D19, D20), `docs/STUDIES.md`, `README.md`, `LITERATURE.md` | Tests for the above; documentation | Ran the tests; read the documentation changes |

## Experiment runs

1. On 15 November 2026 the AI tool (Claude Code) executed `psoann sweep all --out results` (30 runs per
   setting; the two pilot studies use 10), producing `results/*.csv` and `results/*.histories.jsonl`
   (2200 runs, about 5 minutes). It ran the code unchanged and did not select or edit any numbers.
2. On 22 November 2026 we reran all studies ourselves with the same command in the VS Code terminal
   (2320 runs with the new PSO additions, about 9 minutes) and produced the tables and figures with the
   scripts listed below. Runs are seeded, so the same command gives the same numbers. The results in the
   report are from this run.
3. On 26 November 2026 the AI tool (Claude Code) reran `psoann sweep all --out results` and the five scripts below after the
   final cleanup, so that the results used in the report are included in the repository. The 2320 result rows and the
   per-iteration histories are identical to those of the code before the cleanup (only `runtime_sec` differs), and the values
   printed in the report (boundary-rule means and Holm-corrected p-values, schedule p-values, boundary-hit counts,
   Kruskal-Wallis p for the fitness functions, evolved-activation counts) are reproduced by these files.

Environment of run 2: conda Python environment, Python 3.13.11, Windows. The time taken was measured with

```powershell
(Get-ChildItem results\sweep_*.csv | ForEach-Object { Import-Csv $_ } | Measure-Object -Property runtime_sec -Sum).Sum / 60
# 9.22 (minutes)
```

### Commands used to reproduce the results

```bash
psoann sweep all --out results                                       # all studies
python scripts/make_outputs.py                                       # statistics files and boxplots
python scripts/plot_main_results.py                                  # report Figures 1 and 2
python scripts/plot_diversity_and_boundary_hits.py                   # report Figure 3
python scripts/plot_boundary_rule_significance.py                    # report Figure 4
python scripts/plot_boundary_rule_significance.py --study schedule   # Q3 appendix figure
```

Close an image before regenerating it if the operating system locks the file.

## What the AI did not do

- It did not write the report text.
- It did not verify citations.