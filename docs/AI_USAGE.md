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
| Statistics and plots | `analysis/stats.py`, `analysis/plots.py`, `cli.py` | Kruskal-Wallis, Holm-corrected Mann-Whitney, A12, boxplot and heatmap | TO COMPLETE |
| Tests and docs | `tests/test_extensions.py`, `docs/STUDIES.md`, `docs/SUBMISSION.md`, `scripts/make_submission.sh` | Tests for the above and the planning documents | TO COMPLETE |


Claude (PSO owner):

| Area | Files | What the AI did | How we checked it |
|---|---|---|---|
| Coefficient schedules | `pso/schedules.py`, `pso/optimizer.py`, `config.py` (`pso.schedule`) | `constant`, `linear_inertia`, `tvac`; the optimiser asks the schedule for alpha, beta, gamma, delta before each velocity update | Ran the test suite (167 passed). Ran `psoann run` with `pso.schedule=tvac` and the Q8 study ourselves |
| Boundary handling | `pso/bounds.py`, `pso/optimizer.py` | `reject` (the lecture's strategy 2); handlers also receive the positions before the move | Ran the test suite. Ran `psoann run` with `pso.bound_handling=reject` and the Q5 study ourselves; checked the boundary-hit curves against the rule's behaviour |
| Studies | `experiments/presets.py`, `scripts/make_outputs.py` | `reject` added to the Q5 `bounds` study; new Q8 `schedule` study | Ran both studies; the five existing boundary rules gave the same means as before the change |
| Report figures | `scripts/plot_diversity_and_boundary_hits.py`, `scripts/plot_boundary_rule_significance.py` | Figure 3 (diversity and boundary hits) and the pairwise-significance charts, drawn from our result files | Ran both scripts on our results; compared the printed means and p-values with `psoann summarise` and `psoann stats` |
| Tests and docs | `tests/test_schedules.py`, `tests/test_extensions.py`, `docs/design.md` (D19, D20), `docs/STUDIES.md`, `README.md`, `LITERATURE.md` | Tests for the above; documentation | Ran the tests; read the documentation changes |

## Experiment runs

1. On 15 November 2026, the AI tool (Claude Code) executed `psoann sweep all --out results` (30 runs per
setting; the two pilot studies use 10), producing `results/*.csv` and `results/*.histories.jsonl`
(2200 runs, about 5 minutes). It ran the code unchanged and did not select or edit any numbers.

2. On 22 November 2026, we reran all studies ourselves with the same command, in the VS Code terminal
(2320 runs with the new PSO additions, about 9 minutes), and produced the tables and figures with
`scripts/make_outputs.py` and the two plotting scripts above. Runs are seeded, so the same command gives
the same numbers. The results in the report are from this run.

## Rerun Steps in Visual Studio Code Terminal

*terminal: conda python environment, version: 3.13.11*

*Run all the sweeps, and then print the base results (figures, stats, json)*
- \F21BC-GroupPG_14-ANN-PSO> psoann sweep all --out results 

*Check for time taken in Windows*
(Get-ChildItem results\sweep_*.csv | ForEach-Object { Import-Csv $_ } | Measure-Object -Property runtime_sec -Sum).Sum / 60*
--> 9.22176156166242 *(minutes)*

*Generate output and charts (close diversity_and_boundary_hits.png or resulting images to allow overwriting them)*
python scripts/make_outputs.py
python scripts/plot_diversity_and_boundary_hits.py
python scripts/plot_boundary_rule_significance.py

## What the AI did not do

- It did not write the report text.
- It did not verify citations.