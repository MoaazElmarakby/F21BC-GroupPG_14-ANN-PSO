# AI usage log

AI tools were used to write and test code and to explain concepts (November 2026). The report text was written by the group.

## Tools

- Claude Code (Anthropic), cloud session
- Claude (Anthropic), Claude app

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

## Prompts used to design and build the repository

1. **Design first (no code).** Help me design the solution before any code: the components we need to build and the decisions we must make. Support each decision with papers from a literature review. Do not implement anything yet.
2. **PSO versus backprop.** Give me a quick overview of the PSO optimiser and where it sits in the design. I know deep learning with Adam and backpropagation, so explain how training an ANN with PSO differs.
3. **Literature review per module.** Do a thorough literature search before answering. For each module in the design (fitness function, ANN architecture and activations, PSO variant, informants, boundary handling, coupling, hyperparameter search, evaluation metrics), list the candidate approaches found in the literature and highlight the best one, with reasons and references.
4. **Build the repo.** Build a repository with the base implementation of this design, so we can add features and alternatives later. Design the repo structure first, then implement it. Make it professional: modular, with clean-code principles. (Works best with the earlier design and decisions pasted in.)
5. **Diagram and choices.** Draw a diagram of this design, and list the design choices I need to make.
6. **Recommendations.** Given the brief, the guide and our design, what do you recommend we change? Rank the recommendations.
7. **Check the lecture decks** (attach the four Week 7 decks). Here are the Week 7 lecture decks. Check them against our design and give me your conclusions before we implement anything. Focus on the PSO coefficient roles (which of gamma and delta is which), Algorithm 39 details, informant selection, whether fitness is maximised or minimised, and boundary handling.
8. **Plan for the highest marks.** Aim for the highest marks under the marking rubric and all the instructions in the coursework PDF. Think step by step before implementing. Tell me which design decisions I need to take and what I'm missing. My list so far:
    1. the research questions my experiments answer
    2. the fitness function (can we compare several?)
    3. evaluation metrics beyond accuracy
    4. the ANN architecture and why
    5. activation functions, included in the experiments
    6. hyperparameters and grid search
    7. how PSO and the ANN are coupled
    8. how to compare and present results
    9. different PSO designs

    Decide what we do for each, and design the implementation and experiments to cover all of it.
9. **Decision sheet.** Give me a table or doc listing what is important and affects my marks, what is over-engineering or not required, and the choices I can consider.

## Experiment runs

1. 15 November 2026: Claude Code ran `psoann sweep all --out results` (2200 runs). It ran the code unchanged and did not select or edit any numbers.
2. 22 November 2026: we reran all studies ourselves with the same command (2320 runs, including the new PSO additions) and produced the tables and figures with the scripts in `scripts/`. Runs are seeded, so the same command gives the same numbers. **The results in the report are from this run.**
3. 26 November 2026: Claude Code reran everything after the final cleanup. The result files are identical to those produced by the code before the cleanup (only `runtime_sec` differs), and the values printed in the report are reproduced.

The commands to reproduce the results are in `README.md` and `docs/STUDIES.md`.

## What the AI did not do

- It did not write the report text.
- It checked the bibliographic details of the references online (authors, venue, pages) but did not read the papers.
