# Report plan (F21BC): what to write, where, and with which evidence

This is a plan, not report text. The brief and your decision sheet say the report and the analysis must be
your own words. Everything below tells you *what* each part must contain and *which file* supplies the evidence.

Limits: **6 pages and 3000 words** excluding references and appendices, **Arial or Calibri, 12 pt or larger**,
submitted as **PDF**. Plan for about **5.5 pages and 2,200 words** so captions and slack fit. Sections required
by the brief: Implementation, Results and Discussion, Conclusions, References (appendices optional, and the
marker barely reads them).

## Where the marks are (F21BC weights)

| Criterion | Weight | What the marker looks for | Where you earn it |
|---|---|---|---|
| Implementation | 35 | Correct, configurable, commented code; clear description of what you built and exactly how PSO couples to the ANN; understanding shown; development process documented | Section 1, the code, `docs/design.md`, `docs/AI_USAGE.md` |
| Experiments, results, discussion | 30 | All required questions answered; sound method; sensible values; repeated runs; clear tables and plots; conclusions supported by evidence; **explanations**, not descriptions; limitations acknowledged | Sections 2 and 3 |
| Going further | 25 | Extensions that are *well motivated* and whose *impact is evaluated*; literature used to motivate, interpret or compare; breadth and maturity | Sections 1, 3 and 4 (depth beats count) |
| Report | 10 | Clear, concise, well structured, within limits, figures and tables integrated, references used properly | Whole document |

Every claim in Section 3 follows one pattern: **what we expected (and why, with a citation) -> what we measured
(mean, standard deviation, test result) -> why we think it happened (with a diagnostic) -> caveat.**

## Section by section

### 0. Title block and abstract (about 0.1 page, 60 words)
Course, names, matriculation numbers, one sentence on what was done and the headline result. Skip if space is tight.

### 1. Implementation (about 1.0 page, 450 words, 1 small figure or table)

Include, in this order:
1. **Code structure** in 4-5 lines: `ann/` (feed-forward network, activations, flat-vector codec), `pso/` (Algorithm 39,
   informants, boundary handling), `training/` (the only place they meet), `experiments/` (sweeps, seeding),
   `analysis/`. Mention the rule that `ann/` and `pso/` never import each other (enforced by a test).
2. **Exactly how PSO is coupled to the ANN** (the brief asks for this explicitly):
   vector layout (per layer: weights row-major, then biases; length = sum of n_in x n_out + n_out; 49 for 4-8-1),
   one particle = whole network, fitness = classification accuracy on the **training** set (minimising error rate
   is equivalent), full-batch and deterministic, search bounds [-2, +2], architecture fixed for the whole run.
3. **Algorithm 39 mapping**: a compact table or two sentences using the table in `docs/design.md` section 5
   (line numbers 7-9, 12-15, 16-19, 20-24, 25-26, 28). State the assumptions: informants include the particle itself;
   strict improvement for bests; evaluation budget = swarm x iterations exactly.
4. **Extensions you built** (one line each, detail goes in Section 3): evolved activation genes, nearest-neighbour and
   regenerated informants, four boundary rules plus none, MSE and cross-entropy fitness, random-search control,
   non-parametric statistics.
5. **Data and protocol**: banknote, 1372 rows, one fixed stratified 70/30 split (seed 0), standardisation fitted on
   training data only, test set used for reporting only.
6. **AI use**: two sentences (tool, which code, what you fixed) pointing to `docs/AI_USAGE.md`. The brief requires
   it; fill that file in honestly first.

Do not re-explain what an ANN or PSO is. The brief says the reader already knows.

### 2. Methods and protocol (about 0.75 page, 350 words, **Table 1**)

**Table 1 is the most valuable table in the report**: one row per hyperparameter you studied.

| Factor | Values | Why these values (cite) |
|---|---|---|
| Architecture | no hidden layer; widths 2, 4, 8, 16 x depth 1, 2, 3 (5 to 641 parameters) | Linear baseline shows what hidden units buy; slide 8 (about 100 dimensions) |
| Budget split | 500 and 2000 evaluations; swarm 5, 10, 20, 25, 50, 100 | The brief's 10x50 against 50x10 example, extended |
| Coefficients | (beta, gamma, delta) whole numbers summing to 4; alpha 0.7298 | Clerc and Kennedy constriction values; slide 14 |
| Informants | global, random fixed, random regenerated, nearest; k = 1, 2, 3, 5, 10 | Luke; Kennedy and Mendes; slide 9 |
| Boundary | clamp, reflect, wrap, random re-initialisation, none | Helwig et al. (check); slide 26 |
| Activation | logistic, tanh, ReLU, leaky ReLU, evolved (2 hidden layers of 8) | The brief's activation list |
| Fitness | error rate (accuracy), MSE, cross-entropy | Brief specifies accuracy; others as a design comparison |

Then state, briefly: 30 runs per setting with the **same seeds across settings** (paired), mean and standard
deviation, fixed split, budget 2000 unless varied, how you picked hidden hyperparameters (the weight bound and
initial velocity pilots, **judged on training accuracy only**), metrics (accuracy and F1 with forged as the positive
class; evaluations-to-target for speed, and why: accuracy sits near the ceiling), and the statistics (Kruskal-Wallis,
Holm-corrected Mann-Whitney, A12 effect size, why non-parametric). One sentence on the 24 duplicate rows as a limitation.

### 3. Results and discussion (about 3.25 pages, 1,150 words, 5 figures, 3 tables)

Order by the brief's three required questions first, then extensions. Spend words where the effects are biggest.

| Part | Evidence file | Show | Say (explain, do not just describe) |
|---|---|---|---|
| **Control Q0** (2-3 sentences, in Table 2) | `results/stats/random_search__test_accuracy.txt` | PSO 0.996 against random search 0.931, A12 = 1 | PSO is doing real work: it beats equal-budget sampling every time |
| **Q1 Architecture** (required) | `architecture_vs_parameters.png` (**Fig 1**), `architecture_heatmap_accuracy.png`, `stats/architecture__test_accuracy__vs_1x8.txt` | Accuracy against parameter count, by depth; linear baseline | The data is nearly linearly separable (linear 0.988); small single-layer nets are as good as anything; deeper and wider nets get worse, and 3x16 (641 dimensions) is worst and slowest. Explain: search-space dimension grows while the budget is fixed (slide 8); depth adds saturating compositions that make the landscape harder |
| **Q2 Allocation** (required) | `budget_heatmap_accuracy.png` (**Fig 2**), `stats/swarm_vs_iterations__...vs_b2000-s20.txt` | Swarm size by budget | At 2000, swarms of 10 and 20 are tied and best; 5 is too small, 50-100 leave too few iterations. At 500, small swarms win outright. Explain with exploration against exploitation and the number of iterations available to converge |
| **Q3 Coefficients** (required) | `coefficients_heatmap_accuracy.png` (**Fig 3**), `stats/coefficients__...vs_b1-g2-d1.txt` | beta against gamma, delta implied | The best mixes balance personal and informant pull with a little global (b1, g2, d1). Extremes fail: all-delta converges prematurely, all-beta (4,0,0) has no social information. Link to slide 14. Caveat: sum of 4 with alpha 0.7298 is outside the usual stable region, so interpret cautiously |
| **Q4 Informants** | `informants_lines_accuracy.png` (**Fig 4**), `informants_convergence_diversity.png`, `stats/informants__...vs_random-k3.txt` | Strategy by k | k = 1 is significantly worse for every strategy; from k = 3 up no significant gains, including global. Explain with diversity curves: too few informants slows information flow. Be careful: the brief's slide 9 predicts that global converges too early; your data does **not** show a significant penalty, so say that and give a possible reason (small, easy problem) |
| **Q5 Boundary** | `bounds_convergence_boundary_hits.png` (**Fig 5**), `stats/bounds__...vs_bound_handling=clamp.txt` | Hits per iteration for each rule | Wrap and none leave about 350-500 coordinates outside the box all run, and wrap is worst; clamp, random and reflect hold hits near 25-65. This is a diagnostic-backed explanation: exactly what the marker wants |
| **Q6 Activations** | `stats/activations__...vs_tanh.txt`, `activations_evolved_choices.txt` | Table 3 | Only logistic is significantly worse; evolved is tied with the best fixed. Report the negative result honestly: evolution chose a mix (ReLU 13, tanh 10, logistic 7 per layer) and gave no gain. Possible reason: the choice matters little on this problem, and the extra genes add noise |
| **Q7 Fitness** | `stats/fitness__test_accuracy.txt` | Table 3 | No significant difference (p = 0.053). Compare on test accuracy and F1 only, never on `best_fitness` values (different units) |

Table plan (max 4): **T1** protocol (Section 2), **T2** control + architecture summary (mean, std, accuracy, F1,
reached-target), **T3** Q6 and Q7 and boundary summary with Holm p and A12, **T4** (optional) published-results comparison.

Plot plan (max 5): Fig 1 to Fig 5 above. Everything else goes to the appendix with **one sentence** naming it
(for example "boxplots for every study, convergence curves for the control and for the budget study").

Rules for this section:
- Always give **mean +/- standard deviation over 30 runs**, never single runs.
- When a difference is not significant after Holm correction, **say so**. Several visible gaps are not significant;
  claiming them costs credibility.
- Report `evals_to_target` **together with** `reached_target` (the share of runs that reached it); the mean alone
  ignores runs that failed.
- Compare with **published banknote results** (one short paragraph, 3-5 lines): you must find the sources yourselves.
- Acknowledge limitations here, not only at the end: one fixed split, 24 duplicate rows, near-ceiling accuracy.

### 4. Conclusions (about 0.5 page, 250 words)

1. Two or three sentences summarising what you did and the three required answers, each with a number.
2. The extensions and what they showed (including the evolved-activation negative result).
3. **What you would do with more time** (the brief asks for this and rewards breadth): cross-validated random-search
   tuning, adaptive or decaying inertia, constriction and asynchronous PSO, other datasets, more than one split,
   letting PSO choose layer widths, comparison with backpropagation. One sentence each is enough.

### References (outside the limit)
Harvard style. Every reference must **exist**, be **cited in the text**, and be **read by you**. Start from
`LITERATURE.md` (rows marked `TODO(verify)` were written from memory and are not yet checked). You still need:
sources on activation functions, published banknote accuracies, and at least one paper per extension.

### Appendices (optional, barely read)
Extra figures (boxplots, convergence curves), the full pairwise statistics tables, the extra metrics (precision,
recall, MSE), and the contribution summary if you embed the signing sheet.

## How to work

| Step | When | What |
|---|---|---|
| 1 | now | Run `scripts/run_all.sh` yourselves (see the runbook), so the numbers are yours and reproducible |
| 2 | now | Read the literature rows, fill `LITERATURE.md`, fill `docs/AI_USAGE.md` |
| 3 | now | Write Section 1 while the code is fresh: it needs no results |
| 4 | next | Pick the 5 figures and 4 tables; write one-sentence captions that state the finding |
| 5 | next | Write Section 3 in the order Q1 to Q7; for each, write the claim first, then check it against the stats file |
| 6 | then | Cut to 3000 words (aim 2200), check 6 pages at 12 pt |
| 7 | last | Each of you explains the other's half aloud (viva); sign and add the contribution sheet; build the zip |

Suggested split: ANN owner writes the implementation of the ANN and Q1, Q6, Q7; PSO owner writes the PSO
implementation and Q2 to Q5 and Q0. Both write the introduction of Section 2 and edit each other's parts.

## Do not do this

- Do not select a setting using test accuracy. For the pilots use **training** accuracy
  (`--metric train_accuracy`).
- Do not describe a result without an explanation, a diagnostic, or an honest "we do not know".
- Do not put important results in the appendix.
- Do not cite a paper you have not read, or list one you do not cite.
- Do not use AI to write the report or analyse the results; log any AI help in `docs/AI_USAGE.md`.
- Do not exceed the page and word limits: figures and tables count towards pages.
