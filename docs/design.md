# Design

This document describes the structure of the code base, the conventions it follows, and the decisions
behind its defaults. The literature that motivates the defaults is listed in
[LITERATURE.md](../LITERATURE.md).

## 1. Goals and non-goals

**Goals**

1. A correct, readable, from-scratch **ANN** (configurable depth, width and activations) and **PSO**
   (Algorithm 39 of Luke's *Essentials of Metaheuristics*), coupled so that PSO trains the ANN on the
   Banknote Authentication task.
2. A **reproducible experiment harness**: every result row carries its full configuration and can be
   regenerated from `(config, base_seed, run index)`.
3. **Extensibility**: new activations, informant strategies, boundary handlers, coefficient schedules and
   fitness metrics are added by writing one small class or function and registering it, without editing
   the core.

**Non-goals**: backpropagation or any gradient code (PSO trains the network); GPU support; ANN or PSO
libraries. NumPy is used for all ANN and PSO computation; pandas, Matplotlib and SciPy are used only in
`analysis/` and the plotting scripts.

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
                          summary, stats, plots
```

**Dependency rule:** arrows point downward only. `ann/` and `pso/` never import each other. The only place
the two meet is `training/`, through a single callable:

```python
Fitness = Callable[[NDArray[np.float64]], float]  # parameter vector -> score, HIGHER IS BETTER
```

This keeps the optimiser reusable and independently testable (the PSO tests optimise a sphere function
with no neural network involved). The rule is enforced by `tests/test_architecture.py`, which reads the
import statements. (`training/` also imports `evaluation/` for the metrics; `registry.py` is a second leaf
next to `config.py`.)

## 3. Conventions

| Topic | Convention |
|---|---|
| Direction | **Fitness is maximised everywhere** (as in Algorithm 39). Losses are negated (`neg_mse`). |
| Randomness | No global RNG. Everything takes a `numpy.random.Generator`. |
| Seeding | Run *i* uses `SeedSequence(base_seed, spawn_key=(i,))`, which is exactly child *i* of `SeedSequence(base_seed).spawn(n_runs)`, so run *i* does not depend on `n_runs`. Each run spawns separate streams for the split and for PSO. The `fixed` split uses `SeedSequence([base_seed, 1])`. |
| RNG draws | PSO draws the same random numbers regardless of the coefficient values and schedule, the boundary handler, and K for random informants, so a sweep over any of them compares settings on the same initial swarm. |
| Evaluation budget | `swarm_size × iterations` fitness calls, exactly. The final position update is not assessed. |
| Split policy | `fixed` (default): one split for all runs, so the run-to-run standard deviation measures the stochasticity of PSO only. `per_run`: a new split for each run. Splits are stratified; standardisation uses the training statistics only. |
| Parameter vector | Per layer: weights (`n_in × n_out`, row-major) then biases (`n_out`); layers concatenated in order. Length = Σ (n_in·n_out + n_out). With evolved activations, one gene per hidden layer is appended. |
| Single output | One logistic unit; predict class 1 if the output is at least 0.5. Several outputs: argmax. |
| Config | Frozen dataclasses, validated in `__post_init__`; unknown keys are rejected, so typos fail loudly. Numeric ranges are validated in `config.py`; *names* (activation, strategy, …) are validated when the component is built. |
| Results | One CSV row per run (metrics plus the full flattened config as `cfg.*` columns and `config_json`) and a JSONL file with per-iteration histories. `runtime_sec` is the only non-deterministic column. |

## 4. Interfaces and extension points

| To add… | Implement | Register with | Select via config |
|---|---|---|---|
| an activation | `fn(x) -> x` | `register_activation(name, fn)` | `ann.hidden_activation = "name"` |
| an informant strategy | subclass `InformantStrategy` (`build`, optionally `regenerate`) | `register_informant_strategy(name, lambda cfg: Strategy(...))` | `pso.informants = "name"` |
| a boundary handler | subclass `BoundaryHandler` (`apply`, in place, no random draws; it receives the pre-drawn `noise` and the `previous` positions) | `register_boundary_handler(name, cls)` | `pso.bound_handling = "name"` |
| a coefficient schedule | subclass `CoefficientSchedule` (`at(progress) -> Coefficients`, no random draws) | `register_coefficient_schedule(name, lambda cfg: Schedule(...))` | `pso.schedule = "name"` |
| a fitness metric | `metric(outputs, labels) -> float` | `register_fitness_metric(name, fn)` | `training.fitness = "name"` |
| an optimiser | a class with `optimise(fitness, space, rng) -> PSOResult` | pass to `PSOTrainer(optimizer=...)` or `run_single(optimizer=...)` | `training.optimizer` for the built-in ones |
| a study | an entry in `experiments/presets.py` (`grid`, `fixed_budget` and/or `points`) | — | `psoann sweep name` |

Informant matrices have shape `(swarm_size, m)`; every row contains the particle itself in column 0, so
ties in fitness are resolved in favour of the particle itself.

### Components beyond the Algorithm 39 baseline

All are selected by configuration; none changes the default behaviour.

* **Evolved activation functions** (`ann.evolve_activations`): `EvolvedActivationNetwork` appends one gene
  per hidden layer to the weight vector. The integer part of a gene in `[0, K)` selects that layer's
  activation from `ann.activation_choices` (default: logistic, tanh, ReLU). Weights keep `±pso.bound`;
  genes use `[0, K)`. The vector length and the architecture stay fixed. A network provides its own
  per-dimension bounds (`network.bounds(b)`), so `ann/` never imports `pso/`.
* **Position-based informants** (`pso.informants = "nearest"`): the `k` nearest particles in search space,
  recomputed every iteration (`InformantStrategy.uses_positions`).
* **Boundary rules**: `clamp` (default), `reject`, `reflect`, `wrap`, `random` and `none`. `random` needs
  random numbers, so the optimiser always draws one noise matrix per move and passes it to every handler,
  which keeps the random stream identical whichever handler is used. `reject` cancels the move in the
  offending coordinate only, so every handler also receives the positions before the move (`previous`).
  The optimiser records the number of coordinates outside the bounds at every iteration (`boundary_hits`).
* **Coefficient schedules** (`pso.schedule`, `pso/schedules.py`): `constant` (Algorithm 39 as written, the
  default), `linear_inertia` (α 0.9 → 0.4) and `tvac` (β 2.5 → 0.5 while γ 0.5 → 2.5). The optimiser asks
  the schedule for α, β, γ and δ before every velocity update; `progress` runs from 0 at the first move
  to 1 at the last. Coefficients that a schedule does not name keep their configured values.
* **Random-search control** (`training.optimizer = "random_search"`): the same evaluation budget and the
  same history keys as PSO.
* **Studies as data**: `experiments/presets.py` defines every study (see [STUDIES.md](STUDIES.md)); sweeps
  accept explicit `points` for settings that do not form a full grid.

## 5. Algorithm 39 → code

Line numbers follow Luke (2013), *Essentials of Metaheuristics*, 2nd ed., Algorithm 39. The code comments
use the form `[Alg. 39, line N]`.

| Alg. 39 | Meaning | Code (`pso/optimizer.py`) |
|---|---|---|
| 1–6 | parameters: swarm size, α, β, γ, δ, ε | `PSOConfig` fields (α, β, γ, δ reach the update through `self.schedule.at(...)`, fixed by default) |
| 7–9 | create random particles with random velocities | `_initialise_swarm` |
| 10 | `Best ← ∅` | `gbest_fitness = -inf` in `optimise` |
| 11, 27 | repeat … until out of time | the `for iteration` loop (budget = swarm × iterations) |
| 12–13 | assess the fitness of every particle | `_assess` |
| 14–15 | `Best ← x` if fitter (strict) | the `scores[leader] > gbest_fitness` block (personal bests use the same strict test) |
| 16 | loop over particles to compute their targets | `_informant_best` (vectorised over all particles) |
| 17 | `x*`: the particle's previous fittest location | `pbest_position` |
| 18 | `x+`: fittest location among the informants (including itself) | `_informant_best` |
| 19 | `x!`: fittest location of any particle | `gbest_position` |
| 20–24 | per-dimension velocity update (b, c, d are lines 21–23) | `_update_velocity` |
| 25–26 | position update `x ← x + ε·v` | `_move` |
| 28 | return `Best` | `PSOResult` |

Assumptions that the pseudo-code leaves open (documented in the module docstring): informants include the
particle itself; a best changes only on strict improvement; the last iteration assesses but does not move
the swarm.

Deliberate departures from the pseudo-code (all optional and off by default unless stated): the boundary
rule (`bound_handling`, default `clamp`), velocity clamping (`max_velocity_fraction`), informant
resampling (`regenerate_informants`), position-based informants (`nearest`) and time-varying
coefficients (`schedule`, default `constant`). The initial velocity follows the book's default (half the
vector between two random points); `initial_velocity = "zero"` is the alternative the book mentions.

## 6. Design decisions

| ID | Decision | Rationale |
|---|---|---|
| D1 | NumPy only for the ANN and PSO; no ANN or PSO libraries | The aim is to implement both algorithms at the code level |
| D2 | Fitness is the training accuracy by default | Accuracy is the quantity to optimise; MSE and cross-entropy are provided for comparison |
| D3 | Defaults α = 0.7298, β = γ = 1.4962, δ = 0 | Constriction-equivalent values (Clerc and Kennedy, 2002); δ = 0 makes the update use the informants only |
| D4 | `n_informants = 3`, random, static | Standard PSO default (Bratton and Kennedy, 2007); the informant study varies k and the strategy |
| D5 | Weights bounded to ±2, clamped | Bounded search helps network training (Rakitianskaia and Engelbrecht, 2014); confirmed by the weight-bound pilot |
| D6 | Features standardised with training statistics | Keeps the weight bounds meaningful and avoids test-set leakage |
| D7 | Default network 4-8-1, tanh hidden, logistic output | A small network (49 parameters) keeps the search space tractable |
| D8 | CSV per run plus JSONL histories | Portable, readable with pandas, self-describing |
| D9 | `n_informants` counts the *other* particles drawn for a particle; the particle itself is always added (k = 3 means four informants) | Matches Luke's "x is always one of its own informants" and makes k = 1 meaningful |
| D10 | `clamp` zeroes the velocity component of a clamped coordinate; `reflect` reverses it | Standard behaviour; stops clamped particles from pressing on the wall |
| D11 | Defaults not fixed by the decisions above: swarm 30, 100 iterations, 70/30 stratified split | Small and fast; the studies set their own budget (D15) |
| D12 | Studies are plain Python dictionaries (`grid`, `fixed_budget`, `points`) | Declarative, and every point is an ordinary experiment |
| D13 | `ring` uses `max(1, n_informants // 2)` neighbours on each side; `global` (alias `all`) uses everyone | Gives the informant study its two extremes without new configuration keys |
| D14 | Evolved activations: one gene per hidden layer in `[0, K)`, decoded by floor and clipped; the output layer stays logistic | Keeps the particle fixed-length while letting PSO choose the non-linearity |
| D15 | Studies use 20 particles × 100 iterations = 2000 evaluations, 30 runs, one fixed split and the same seeds for every setting; the budget study also uses 500 | The evaluation count is the unit of cost; shared seeds reduce noise; two budgets avoid a ceiling effect |
| D16 | The random-search control draws `swarm_size` samples per "iteration" | Identical evaluation count and history shape as PSO |
| D17 | Statistics are non-parametric (Mann-Whitney U with Holm correction; Kruskal-Wallis; A12) | Accuracy saturates near 1.0 and has ties, so no normality is assumed |
| D18 | Hyperparameters are selected on training fitness only; the test set is used for reporting | Selecting on the test set would be a validity flaw |
| D19 | Time-varying coefficients are named schedules with the literature's values built in: `linear_inertia` α 0.9 → 0.4 (Shi and Eberhart, 1999), `tvac` β 2.5 → 0.5 and γ 0.5 → 2.5 (Ratnaweera et al., 2004) | One configuration key turns the comparison into a three-setting study, and `constant` leaves every earlier result unchanged |
| D20 | `reject` cancels the move only in the offending coordinates and leaves the velocity alone | Completes the four strategies from the lectures (truncate, reject, bounce, re-initialise) in the boundary study |

## 7. Known limitations

* The dataset has 24 duplicate rows, which are kept; some test samples may therefore also occur in the
  training set.
* The studies use one fixed split. The `per_run` split policy exists for a robustness check.
* Final accuracy is close to its ceiling on this dataset, so convergence speed (`evals_to_target`,
  `evals_to_final_best`) is reported alongside it.
* The `random` boundary rule redraws only the offending coordinate, whereas the lecture strategy
  re-initialises the whole particle.
