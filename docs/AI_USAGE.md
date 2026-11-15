# AI usage log

The coursework requires AI use to be documented. Keep this file true: add a row whenever AI helps with code,
and fill in the "What we checked or changed" column yourselves. **The report text, the experiments and their
analysis must be your own work; AI is not used for them.**

## Tool

Claude Code (Anthropic), model Claude Sonnet 5.5 (`claude-sonnet-5-5`), used in a cloud session on the
`claude/wonderful-johnson-f6h13v` branch, November 2026.

## What the AI wrote

| Area | Files | What it did | What we checked or changed (fill in) |
|---|---|---|---|
| Repository base | `src/psoann/` (config, data, ann, pso, training, experiments, analysis, cli), `tests/`, `docs/design.md`, `README.md` | Initial from-scratch implementation of the ANN, Algorithm 39 PSO, coupling, experiment harness and tests (earlier sessions) | |
| Metrics | `evaluation/metrics.py`, `training/fitness.py` | Precision, recall and F1 (forged = positive) | |
| Boundary handling | `pso/bounds.py` | Added `wrap` and `random` handlers; boundary-hit counter in `pso/optimizer.py` | |
| Informants | `pso/informants.py`, `pso/optimizer.py` | Added `nearest` (k nearest by position, recomputed each iteration) | |
| Evolved activations | `ann/network.py`, `config.py`, `training/trainer.py` | One gene per hidden layer selects the activation; bounds per dimension | |
| Random-search control | `pso/random_search.py`, `config.py`, `training/trainer.py` | Equal-budget control optimiser | |
| Studies | `experiments/presets.py`, `experiments/sweep.py` | Q0 to Q7 study definitions and explicit sweep points | |
| Statistics and plots | `analysis/stats.py`, `analysis/plots.py`, `cli.py` | Kruskal-Wallis, Holm-corrected Mann-Whitney, A12, boxplot and heatmap | |
| Tests and docs | `tests/test_extensions.py`, `docs/STUDIES.md`, `docs/SUBMISSION.md`, `scripts/make_submission.sh` | Tests for the above and the planning documents | |

## What the AI did not do

- It did not run the reported experiments, interpret results, or write any report text.
- It did not verify citations. `LITERATURE.md` lists what still has to be read and checked by you.

## Not covered by this log

Anything you ask an AI tool to do from now on. Add it here.
