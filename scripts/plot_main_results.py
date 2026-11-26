"""Report Figures 1 and 2: the three required studies (architecture, budget split, coefficients) and informants.

    python scripts/plot_main_results.py [--results results]

Reads  <results>/sweep_architecture.csv, sweep_swarm_vs_iterations.csv, sweep_coefficients.csv and
       sweep_informants.csv (written by `psoann sweep all`)
Writes <results>/figures/fig1_architecture_and_budget.png
           (a) test accuracy against parameter count, one line per depth;
           (b) mean test accuracy for each swarm size at two evaluation budgets
       <results>/figures/fig2_coefficients_and_informants.png
           (a) mean test accuracy over the personal-best and informant-best weights (the global-best
               weight is what remains of 4);
           (b) test accuracy against the number of informants for each informant strategy
"""

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_SIZE: Final[tuple[float, float]] = (7.2, 2.7)  # inches; two panels across a page
DPI: Final[int] = 220
STRATEGY_LABELS: Final[dict[str, str]] = {
    "random": "random, fixed",
    "regen": "random, regenerated",
    "nearest": "nearest",
}


def read_study(results: Path, study: str) -> pd.DataFrame:
    return pd.read_csv(results / f"sweep_{study}.csv")


def draw_heatmap(ax: plt.Axes, table: pd.DataFrame, xlabel: str, ylabel: str) -> None:
    """Heatmap of a pivot table with the value written in every cell."""
    image = ax.imshow(table.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), [f"{c:g}" for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [f"{i:g}" for i in table.index])
    low, high = np.nanmin(table.to_numpy()), np.nanmax(table.to_numpy())
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            if not np.isnan(value):
                colour = "white" if value < (low + high) / 2 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color=colour)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(image, ax=ax, label="mean test accuracy", fraction=0.05, pad=0.02)


def figure_architecture_and_budget(results: Path, out: Path) -> None:
    architecture = read_study(results, "architecture")
    architecture["depth"] = architecture["point"].map(lambda p: 0 if p == "linear" else int(p.split("x")[0]))
    budget = read_study(results, "swarm_vs_iterations")

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [1, 1.1]})
    for depth, frame in architecture.groupby("depth"):
        stats = frame.groupby("n_parameters")["test_accuracy"].agg(["mean", "std"])
        label = "linear" if depth == 0 else f"{depth} hidden"
        left.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", ms=3, capsize=2, lw=1, label=label)
    left.set_xscale("log")
    left.set_xlabel("parameters (log scale)")
    left.set_ylabel("test accuracy")
    left.grid(alpha=0.3)
    left.legend(loc="lower left")
    left.set_title("(a) Q1: architecture")

    table = budget.pivot_table(index="evaluations", columns="cfg.pso.swarm_size", values="test_accuracy")
    draw_heatmap(right, table, "swarm size", "evaluation budget")
    right.set_title("(b) Q2: swarm size at fixed budget")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def figure_coefficients_and_informants(results: Path, out: Path) -> None:
    coefficients = read_study(results, "coefficients")
    informants = read_study(results, "informants")
    informants["strategy"] = informants["point"].str.split("-k").str[0]

    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [1.1, 1]})
    table = coefficients.pivot_table(index="cfg.pso.gamma", columns="cfg.pso.beta", values="test_accuracy")
    draw_heatmap(left, table, r"$\beta$ (personal best)", r"$\gamma$ (informants)")
    left.set_title(r"(a) Q3: $\beta+\gamma+\delta=4$")

    for strategy, frame in informants[informants["strategy"] != "global"].groupby("strategy"):
        stats = frame.groupby("cfg.pso.n_informants")["test_accuracy"].agg(["mean", "std"])
        right.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            ms=3,
            capsize=2,
            lw=1,
            label=STRATEGY_LABELS[strategy],
        )
    right.axhline(
        informants[informants["strategy"] == "global"]["test_accuracy"].mean(),
        color="k",
        ls="--",
        lw=0.8,
        label="global",
    )
    right.set_xlabel("number of informants k")
    right.set_ylabel("test accuracy")
    right.grid(alpha=0.3)
    right.legend(loc="lower right")
    right.set_title("(b) Q4: informants")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    for name, draw in (
        ("fig1_architecture_and_budget.png", figure_architecture_and_budget),
        ("fig2_coefficients_and_informants.png", figure_coefficients_and_informants),
    ):
        draw(args.results, figures / name)
        print(f"wrote {figures / name}")


if __name__ == "__main__":
    main()
