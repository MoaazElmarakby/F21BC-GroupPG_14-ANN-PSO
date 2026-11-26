# AI usage log

AI tools were used to write and test code and to explain concepts (November 2026). The report text was written by the group.

## Tools

| Tool | Model | Used by | Branch |
|---|---|---|---|
| Claude Code (Anthropic), cloud session | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | ANN owner | `claude/wonderful-johnson-f6h13v` |
| Claude (Anthropic), Claude app | `claude-opus-5-5`, `claude-fable-5-2` | PSO owner | `pso-schedules-reject` |

## What the AI wrote and how we checked it

**Claude Code (ANN owner)**

| Area | Files | How we checked it |
|---|---|---|
| Repository base: ANN, Algorithm 39 PSO, coupling, experiment harness, CLI, tests | `src/psoann/`, `tests/`, `docs/design.md` | TO COMPLETE |
| Extensions and studies: F1 metrics, `nearest` informants, `wrap` and `random` boundary rules, evolved activations, random-search control, studies Q0 to Q7, statistics | `evaluation/`, `pso/informants.py`, `pso/bounds.py`, `pso/random_search.py`, `ann/network.py`, `experiments/presets.py`, `analysis/stats.py`, `tests/test_extensions.py` | TO COMPLETE |
| Scripts for statistics and Figures 1 and 2; final cleanup (formatting, unused code removed, documents rewritten) | `scripts/make_outputs.py`, `scripts/plot_main_results.py`, `README.md`, `docs/`, `LITERATURE.md` | TO COMPLETE |

**Claude (PSO owner)**

| Area | Files | How we checked it |
|---|---|---|
| Coefficient schedules (`constant`, `linear_inertia`, `tvac`) and the `reject` boundary rule; new studies Q5 (with `reject`) and Q8 | `pso/schedules.py`, `pso/bounds.py`, `pso/optimizer.py`, `experiments/presets.py`, `tests/test_schedules.py` | Ran the test suite (167 passed) and both studies ourselves; the five existing boundary rules gave the same means as before |
| Figure scripts: diversity and boundary hits (Figure 3), pairwise significance (Figure 4) | `scripts/plot_diversity_and_boundary_hits.py`, `scripts/plot_boundary_rule_significance.py` | Compared the printed means and p-values with `psoann summarise` and `psoann stats` |

## Experiment runs

1. 15 November 2026: Claude Code ran `psoann sweep all --out results` (2200 runs). It ran the code unchanged and did not select or edit any numbers.
2. 22 November 2026: we reran all studies ourselves with the same command (2320 runs, including the new PSO additions) and produced the tables and figures with the scripts in `scripts/`. Runs are seeded, so the same command gives the same numbers. **The results in the report are from this run.**
3. 26 November 2026: Claude Code reran everything after the final cleanup. The result files are identical to those produced by the code before the cleanup (only `runtime_sec` differs), and the values printed in the report are reproduced.

The commands to reproduce the results are in `README.md` and `docs/STUDIES.md`.

## What the AI did not do

- It did not write the report text.
- It checked the bibliographic details of the references online (authors, venue, pages) but did not read the papers.
