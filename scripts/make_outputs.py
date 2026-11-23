"""Generate every statistics table and figure from the sweep results.

    python scripts/make_outputs.py [--results results]

Reads  <results>/sweep_<study>.csv and .histories.jsonl  (written by `psoann sweep all`)
Writes <results>/stats/*.txt   Kruskal-Wallis + Holm-corrected Mann-Whitney + A12 per study
       <results>/figures/*.png boxplots, heatmaps, line plots, convergence / diversity / boundary hits
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from psoann.analysis.plots import plot_boxplot, plot_heatmap, plot_sweep, plot_convergence
from psoann.experiments import read_jsonl

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", type=Path, default=Path("results"))
R = parser.parse_args().results
F, T = R / "figures", R / "stats"
F.mkdir(parents=True, exist_ok=True)
T.mkdir(parents=True, exist_ok=True)
def csv(n): return pd.read_csv(R / f"sweep_{n}.csv")

# ---- statistics (CLI, captured to text files) --------------------------------------------------------
def stats(study, metric, control=None):
    cmd = ["psoann", "stats", str(R / f"sweep_{study}.csv"), "--by", "point", "--metric", metric]
    if control: cmd += ["--control", control]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    name = f"{study}__{metric}" + (f"__vs_{control}" if control else "")
    (T / f"{name}.txt").write_text(out)
    return out

studies = ["random_search","architecture","swarm_vs_iterations","coefficients","informants","bounds","activations","fitness","schedule","pilot_bound","pilot_velocity"]
for s in studies:
    for m in ("test_accuracy", "test_f1"):
        stats(s, m)
for s, c in (("architecture","1x8"),("informants","random-k3"),("bounds","bound_handling=clamp"),("activations","tanh"),("coefficients","b1-g2-d1"),("swarm_vs_iterations","b2000-s20"),("schedule","schedule=constant")):
    stats(s, "test_accuracy", c)

# ---- figures -------------------------------------------------------------------------------------------
for s in studies:
    plot_boxplot(csv(s), "point", "test_accuracy", F / f"{s}_box_accuracy.png")
plot_boxplot(csv("fitness"), "point", "test_f1", F / "fitness_box_f1.png")

# Q1: parameters, width x depth
d = csv("architecture")
d["depth"] = d["point"].map(lambda p: 0 if p == "linear" else int(p.split("x")[0]))  # one line per depth: 1x4 and 3x2 share 25 parameters
plot_sweep(d, "n_parameters", "test_accuracy", F / "architecture_vs_parameters.png", hue="depth")
plot_sweep(d, "n_parameters", "evals_to_final_best", F / "architecture_evals_vs_parameters.png", hue="depth")
nh = d[d["point"] != "linear"].copy()
nh["width"] = nh["point"].str.split("x").str[1].astype(int)
plot_heatmap(nh, "width", "depth", "test_accuracy", F / "architecture_heatmap_accuracy.png")
plot_heatmap(nh, "width", "depth", "reached_target", F / "architecture_heatmap_reached_target.png")

# Q2: swarm size x budget
d = csv("swarm_vs_iterations")
plot_heatmap(d, "cfg.pso.swarm_size", "evaluations", "test_accuracy", F / "budget_heatmap_accuracy.png")
plot_heatmap(d, "cfg.pso.swarm_size", "evaluations", "reached_target", F / "budget_heatmap_reached_target.png")
plot_sweep(d, "cfg.pso.swarm_size", "test_accuracy", F / "budget_lines_accuracy.png", hue="evaluations")

# Q3: beta x gamma (delta = 4 - beta - gamma)
d = csv("coefficients")
plot_heatmap(d, "cfg.pso.beta", "cfg.pso.gamma", "test_accuracy", F / "coefficients_heatmap_accuracy.png")
plot_heatmap(d, "cfg.pso.beta", "cfg.pso.gamma", "reached_target", F / "coefficients_heatmap_reached_target.png")

# Q4: strategy x k
d = csv("informants")
d["strategy"] = d["point"].str.split("-k").str[0]
plot_sweep(d[d.strategy != "global"], "cfg.pso.n_informants", "test_accuracy", F / "informants_lines_accuracy.png", hue="strategy")
plot_sweep(d[d.strategy != "global"], "cfg.pso.n_informants", "evals_to_final_best", F / "informants_lines_evals_to_best.png", hue="strategy")

# convergence / diversity / boundary hits on chosen subsets (all 16 curves would be unreadable)
def subset(study, points):
    return [h for h in read_jsonl(R / f"sweep_{study}.histories.jsonl") if h["point"] in points]
plot_convergence(subset("random_search", {"pso","random_search"}), F / "control_convergence_best.png")
inf = {"global","random-k3","regen-k3","nearest-k3","random-k10"}
for key in ("best", "diversity"):
    plot_convergence(subset("informants", inf), F / f"informants_convergence_{key}.png", key=key)
b = read_jsonl(R / "sweep_bounds.histories.jsonl")
for key in ("best", "diversity", "boundary_hits"):
    plot_convergence(b, F / f"bounds_convergence_{key}.png", key=key)
for key in ("best", "diversity"):
    plot_convergence(read_jsonl(R / "sweep_schedule.histories.jsonl"), F / f"schedule_convergence_{key}.png", key=key)
plot_convergence(subset("swarm_vs_iterations", {"b2000-s5","b2000-s20","b2000-s100"}), F / "budget_convergence_best.png")
plot_convergence(subset("architecture", {"linear","1x8","3x16"}), F / "architecture_convergence_best.png")

# Q6: which activations did evolution choose?
a = csv("activations"); e = a[a.point == "evolved"]
choice = e["hidden_activations"].value_counts()
layers = e["hidden_activations"].str.split("-", expand=True)
tab = pd.DataFrame({f"layer{i+1}": layers[i].value_counts() for i in layers.columns}).fillna(0).astype(int)
(T / "activations_evolved_choices.txt").write_text("combinations:\n" + choice.to_string() + "\n\nper layer:\n" + tab.to_string() + "\n")
print(choice.to_string()); print(tab.to_string())
print(f"wrote {len(list(T.glob('*.txt')))} stats files to {T} and {len(list(F.glob('*.png')))} figures to {F}")
