# Design

This document is the contract for the code base. Read it before adding features, and update it when a
decision changes. The coursework brief (F20BC/F21BC) is the source of requirements; the literature that
motivates the defaults is mapped in [LITERATURE.md](../LITERATURE.md).

**Status:** the base described here is implemented (`src/psoann`, 100+ tests). Sections 5-7 record what
changed from the first draft of this document.

## 1. Goals and non-goals

**Goals**

1. A correct, readable, from-scratch **ANN** (configurable depth, width, activations) and **PSO**
   (Algorithm 39 of Luke's *Essentials of Metaheuristics*), coupled so PSO trains the ANN on the
   Banknote Authentication task.
2. A **reproducible experiment harness**: every result row carries its full configuration and can be
   regenerated from `(config, base_seed, run index)`.
3. **Extensibility**: new activations, informant topologies, boundary handlers, fitness metrics and PSO
   variants are added by writing one small class/function and registering it, without editing the core.

**Non-goals**: backpropagation or any gradient code (the brief says PSO trains the network); GPU
support; ANN/PSO libraries (the brief penalises them). NumPy is used for arrays only (pandas and
matplotlib only in `analysis/`).

## 2. Architecture

```
        config.py, registry.py  (leaves: import nothing from psoann)
                     │
   ┌─────────────────┼──────────────────────────┐
   ▼                 ▼                          ▼
 data/             ann/                        pso/
 loader, split,    activations, network,       optimizer (Algorithm 39),
 scaler            codec                       informants, bounds, schedules, types
   │                 │                          │        evaluation/
   │                 │                          │        metrics
   │                 └──────────┬───────────────┴────────────┐
   │                            ▼                            │
   │                       training/                         │
   │                       fitness (vector → score), trainer │
   └────────────────────────────┬────────────────────────────┘
                                ▼
                          experiments/
                          runner, sweep, io
                                ▼
                          analysis/            cli.py
                          summary, plots
```

**Dependency rule:** arrows point downward only. `ann/` and `pso/` never import each other. The *only*
place the two meet is `training/`, through a single callable:

```python
Fitness = Callable[[NDArray[np.float64]], float]  # parameter vector -> score, HIGHER IS BETTER
```

This keeps the optimiser reusable and independently testable (the PSO tests optimise a sphere function
with no neural network in sight), and lets the two people on the team work in parallel. The rule is
enforced by `tests/test_architecture.py`, which reads the import statements. (`training/` also imports
`evaluation/` for the metrics; `registry.py` is a second leaf next to `config.py`.)

## 3. Conventions (read these)

| Topic | Convention |
|---|---|
| Direction | **Fitness is maximised everywhere** (matches Algorithm 39). Losses are negated (`neg_mse`). |
| Randomness | No global RNG. Everything takes a `numpy.random.Generator`. |
| Seeding | Run *i* uses `SeedSequence(base_seed, spawn_key=(i,))`, which is exactly child *i* of `SeedSequence(base_seed).spawn(n_runs)`, so run *i* is independent of `n_runs`. Each run spawns separate streams for the split and for PSO. The `fixed` split uses `SeedSequence([base_seed, 1])`. |
| RNG draws | PSO draws the same random numbers regardless of coefficient values and coefficient schedule (and regardless of the boundary handler and of K for random informants), so a sweep over α/β/γ/δ compares like with like (same initial swarm). |
| Evaluation budget | `swarm_size × iterations` fitness calls, exactly. The final position update is not assessed. |
| Split policy | `fixed` (default): one split for all runs, so the run-to-run std measures PSO stochasticity only. `per_run`: new split each run. Splits are stratified; standardisation uses training statistics only. |
| Parameter vector | Per layer: weights (`n_in × n_out`, row-major) then biases (`n_out`); layers concatenated in order. Length = Σ (n_in·n_out + n_out). |
| Single output | One logistic unit, predict `1` if output ≥ 0.5. Multiple outputs: argmax. |
| Config | Frozen dataclasses, validated in `__post_init__`, unknown keys rejected (typos fail loudly). Numeric ranges are validated in `config.py`; *names* (activation, topology, ...) are validated when the component is built. |
| Results | One CSV row per run (metrics + the full flattened config as `cfg.*` columns + `config_json`) and a JSONL file with per-iteration histories. `runtime_sec` is the only non-deterministic column. |

## 4. Interfaces and extension points

| To add… | Implement | Register with | Select via config |
|---|---|---|---|
| an activation | `fn(x) -> x` | `register_activation(name, fn)` | `ann.hidden_activation = "name"` |
| an informant topology | subclass `InformantStrategy` (`build`, optionally `regenerate`) | `register_informant_strategy(name, lambda cfg: Strategy(...))` | `pso.informants = "name"` |
| a boundary handler | subclass `BoundaryHandler` (`apply`, in place, no random draws; it receives the pre-drawn `noise` and the `previous` positions) | `register_boundary_handler(name, cls)` | `pso.bound_handling = "name"` |
| a coefficient schedule | subclass `CoefficientSchedule` (`at(progress) -> Coefficients`, no random draws) | `register_coefficient_schedule(name, lambda cfg: Schedule(...))` | `pso.schedule = "name"` |
| a fitness metric | `metric(outputs, labels) -> float` | `register_fitness_metric(name, fn)` | `training.fitness = "name"` |
| a PSO variant | a class with `optimise(fitness, space, rng) -> PSOResult` | pass to `PSOTrainer(optimizer=...)` or `run_single(optimizer=...)` | — |
| a sweep | an entry in `experiments/presets.py` (`grid` and/or `fixed_budget`) | — | `psoann sweep name` |

Informant matrices have shape `(swarm_size, m)`; every row contains the particle itself in column 0, so
ties in fitness are resolved in favour of the particle itself.

Implemented since the base (all selected by config, none changes the default behaviour):

* **Activation functions inside the particle** (`ann.evolve_activations`): `EvolvedActivationNetwork`
  appends one gene per hidden layer to the weight vector; the integer part of gene g in `[0, K)` selects
  that layer's activation from `ann.activation_choices`. Weights keep `+-pso.bound`; genes use
  `[0, K)`. The vector length is fixed and the architecture is fixed, as the brief requires. A network
  provides its own per-dimension bounds (`network.bounds(b)`), so `ann/` never imports `pso/`.
* **Position-based informants** (`pso.informants = "nearest"`): the `k` nearest particles in search
  space, recomputed every iteration (`InformantStrategy.uses_positions`).
* **Boundary rules** `reject`, `wrap` and `random` besides `clamp`, `reflect`, `none`. `random` needs random
  numbers, so the optimiser always draws one noise matrix per move and passes it to every handler: the
  random stream is identical whichever handler is used. `reject` cancels the move in the offending
  coordinate only, so every handler also receives the positions before the move (`previous`).
  `clamp`, `reject`, `reflect` and `random` are the four strategies of the Week 7 lecture.
* **Coefficient schedules** (`pso.schedule`, `pso/schedules.py`): `constant` (Algorithm 39 as written,
  the default), `linear_inertia` (α 0.9 → 0.4) and `tvac` (β 2.5 → 0.5 while γ 0.5 → 2.5). The optimiser
  asks the schedule for α, β, γ, δ before every velocity update; `progress` runs from 0 at the first move
  to 1 at the last. Coefficients a schedule does not name keep their config values.
* **Random-search control** (`training.optimizer = "random_search"`): same budget, same history keys.
* **Studies as data**: `experiments/presets.py` defines Q0 to Q8; sweeps accept explicit `points`.

Still planned (deliberately not implemented):

* **Cross-validation**: `data.split.stratified_kfold` exists (and is tested); add a runner mode that
  loops folds.

## 5. Algorithm 39 → code

Line numbers follow Luke (2013), *Essentials of Metaheuristics*, 2nd ed., Algorithm 39, **checked against
the book text supplied with the project** (the first draft of this table was recalled from memory; every
number turned out to be right). The code comments use the form `[Alg. 39, line N]`.

| Alg. 39 | Meaning | Code (`pso/optimizer.py`) |
|---|---|---|
| 1–6 | parameters: swarm size, α, β, γ, δ, ε | `PSOConfig` fields (α, β, γ, δ reach the update through `self.schedule.at(...)`, fixed by default) |
| 7–9 | create random particles with random velocities | `_initialise_swarm` |
| 10 | `Best ← ∅` | `gbest_fitness = -inf` in `optimise` |
| 11, 27 | repeat … until out of time | the `for iteration` loop (budget = swarm × iterations) |
| 12–13 | assess the fitness of every particle | `_assess` |
| 14–15 | `Best ← x` if fitter (strict) | the `scores[leader] > gbest_fitness` block (personal bests use the same strict test) |
| 17 | `x*`: particle's previous fittest location | `pbest_position` |
| 18 | `x+`: fittest location among informants (incl. itself) | `_informant_best` |
| 19 | `x!`: fittest location of any particle | `gbest_position` |
| 16 | loop over particles to compute their targets | `_informant_best` (vectorised over all particles) |
| 20–24 | per-dimension velocity update (b, c, d are lines 21–23) | `_update_velocity` |
| 25–26 | position update `x ← x + ε·v` | `_move` |
| 28 | return `Best` | `PSOResult` |

Assumptions the pseudo-code leaves open (all documented in the module docstring): informants include
the particle itself; a particle's best changes only on strict improvement; the last iteration assesses
but does not move.

Deliberate departures from the pseudo-code (all optional and off by default unless stated):
boundary handling (`bound_handling`, default `clamp`), optional velocity clamping
(`max_velocity_fraction`), informant resampling (`regenerate_informants`), position-based informants
(`nearest`) and time-varying coefficients (`schedule`, default `constant`). Initial velocity follows
the book's default (half the vector between two random points); `initial_velocity = "zero"` is the
alternative the book mentions.

## 6. Decisions log

| # | Decision | Rationale | Revisit if… |
|---|---|---|---|
| D1 | NumPy only, no ANN/PSO libraries | Brief penalises libraries | — |
| D2 | Fitness = training accuracy by default | Brief specifies accuracy; MSE/cross-entropy provided for comparison | Accuracy plateaus stall the swarm |
| D3 | Defaults α=0.7298, β=γ=1.4962, δ=0 | Constriction-equivalent values (Clerc & Kennedy 2002); δ=0 means informants-only | The coefficient study finds better |
| D4 | `n_informants = 3`, random, static | SPSO default K=3; sweep 2–8 and "all" | Informant study contradicts |
| D5 | Weights bounded to ±2, clamp | Bounded search helps NN training (Rakitianskaia & Engelbrecht 2014); ±2 follows Garro & Vázquez 2015 | Bound study |
| D6 | Standardise features using train statistics | Keeps weight bounds meaningful; avoids leakage | — |
| D7 | Default net: 4-8-1, tanh hidden, logistic output | Small net (49 parameters) keeps the search space tractable | Architecture study |
| D8 | CSV per run + JSONL histories | Portable, pandas-readable, self-describing | — |
| D9 | `n_informants` = number of *other* particles drawn per particle; the particle itself is always added (so K=3 means 4 informants in total) | Matches Luke's "x is always one of its own informants"; makes K=1 meaningful | The brief defines K differently |
| D10 | `clamp` zeroes the velocity component of any clamped dimension (absorbing wall); `reflect` reverses it | Standard behaviour; keeps clamped particles from pressing on the wall | Bound study shows otherwise |
| D11 | Defaults not fixed by earlier decisions: swarm 30, 100 iterations (3000 evaluations), 70/30 stratified split | Small, fast, reproduces the saturation noted in section 7; **assumptions, not requirements** | The brief prescribes sizes or a split |
| D12 | Sweeps are plain Python dicts in `experiments/presets.py`: `grid` (cartesian product of dotted keys) and `fixed_budget` (swarm sizes at equal evaluations, iterations = budget // swarm) | Studies are declarative data (no config files), every point is an ordinary experiment | — |
| D14 | Evolved activations: one gene per hidden layer in `[0, K)`, decoded by floor and clipped; output layer stays logistic | Keeps the particle fixed-length (the brief) while letting PSO choose the non-linearity | The activation study shows no effect |
| D15 | Studies use 20 particles x 100 iterations = 2000 evaluations, 30 runs, fixed split, same seeds across settings; Q2 also uses 500 | Evaluation budget is the unit in the brief; paired seeds reduce noise; budgets of 500 and 2000 avoid a ceiling effect | The pilot contradicts |
| D16 | Random-search control uses `swarm_size` samples per "iteration" | Identical evaluation count and history shape as PSO | — |
| D17 | Statistics are non-parametric (Kruskal-Wallis, Mann-Whitney with Holm, A12) | Accuracy saturates near 1.0 and has ties; no normality assumption | — |
| D18 | Models are selected on training fitness only (pilots); the test set is for reporting | Selecting on the test set is a validity flaw | — |
| D19 | Time-varying coefficients are named schedules with the literature's values built in: `linear_inertia` α 0.9 → 0.4 (Shi and Eberhart 1999), `tvac` β 2.5 → 0.5 and γ 0.5 → 2.5 (Ratnaweera et al. 2004) | One config key makes the comparison a three-setting study (Q8) and keeps each start/end pair together; `constant` leaves every earlier result bit-identical | A study needs other start/end values (register another schedule) |
| D20 | `reject` cancels the move only in the offending coordinates and leaves the velocity alone | The lecture's strategy 2; completes the lecture's four strategies in the boundary study (Q5) | Bound study shows otherwise |
| D13 | `ring` topology uses `max(1, n_informants // 2)` neighbours each side; `global` (alias `all`) uses everyone | Gives the informant study its two extremes without new config keys | — |

## 7. Open questions for the team

* Output encoding for the report: one logistic unit (current) vs two outputs with argmax (both work:
  `ann.n_outputs = 2`).
* Fixed vs per-run split for the headline numbers (the std means different things).
* Which extensions to feature in the report (implemented: activation genes, boundary rules, informant
  strategies, fitness comparison, statistics); the brief rewards depth over count, so pick three.
* Confirm D11 against the brief (swarm size, iterations, train/test proportions, number of runs, what
  must be reported).
* Read the sources in LITERATURE.md and fill in or correct every bibliographic detail marked `TODO(verify)`.
* **Saturation.** In smoke tests (3–10 runs, default settings) the banknote task is solved almost
  perfectly (test accuracy ≈ 0.99–1.00), so final accuracy may barely separate settings. Consider
  also reporting convergence speed (`evals_to_final_best`, or `evals_to_target` with `target_fitness`
  set), using small budgets (the `swarm_vs_iterations` sweep), and `per_run` splits; decide
  before running the experiments you will report.
