"""Report figure: (a) Q4 swarm diversity, (b) Q5 boundary hits, side by side.

    python scripts/plot_diversity_and_boundary_hits.py [--results results] [--out FILE.png]

Reads  <results>/sweep_informants.histories.jsonl and <results>/sweep_bounds.histories.jsonl
       (written by `psoann sweep informants` and `psoann sweep bounds`, or `psoann sweep all`)
Writes <results>/figures/diversity_and_boundary_hits.png: a line per setting (mean over the runs) with a +/- one standard deviation band.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psoann.experiments import read_jsonl

# point id in the results -> (legend label, colour). Colours are fixed per setting, so a setting keeps
# its colour when others are added or removed.
informant_lines: Final[dict[str, tuple[str, str]]] = {
    "global": ("global", "tab:blue"),
    "random-k1": ("random k=1", "tab:orange"),
    "random-k3": ("random k=3", "tab:green"),
    "random-k10": ("random k=10", "tab:red"),
    "nearest-k3": ("nearest k=3", "tab:purple"),
}
boundary_lines: Final[dict[str, tuple[str, str]]] = {
    "bound_handling=clamp": ("clamp", "tab:blue"),
    "bound_handling=reject": ("reject", "tab:brown"),
    "bound_handling=reflect": ("reflect", "tab:orange"),
    "bound_handling=wrap": ("wrap", "tab:green"),
    "bound_handling=random": ("random", "tab:red"),
    "bound_handling=none": ("none", "tab:purple"),
}
figure_size: Final[tuple[float, float]] = (9.6, 3.4)  # inches; two panels across a page
band_alpha: Final[float] = 0.18
legend_headroom: Final[float] = 0.24  # extra height above the data, as a fraction of the data's range


def draw_panel(ax, histories_path: Path, key: str, lines: dict[str, tuple[str, str]], title: str, y_label: str) -> None:
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    evaluations: dict[str, np.ndarray] = {}
    for record in read_jsonl(histories_path):
        point = record["point"]
        if point in lines:
            curves[point].append(np.asarray(record["history"][key], dtype=float))
            evaluations.setdefault(point, np.asarray(record["history"]["evaluations"], dtype=float))
    missing = [point for point in lines if point not in curves]
    if missing:
        raise SystemExit(f"{histories_path} has no runs for {missing}; rerun that study with the current code")

    for point, (label, colour) in lines.items():
        stack = np.vstack(curves[point])
        mean, std = stack.mean(axis=0), stack.std(axis=0)
        ax.plot(evaluations[point], mean, color=colour, linewidth=1.8, label=label)
        ax.fill_between(evaluations[point], mean - std, mean + std, color=colour, alpha=band_alpha, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + legend_headroom * (high - low))  # empty strip at the top, so the legend covers no line
    ax.legend(fontsize=8, ncol=3, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / "figures" / "diversity_and_boundary_hits.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=figure_size)
    draw_panel(left, args.results / "sweep_informants.histories.jsonl", "diversity",
               informant_lines, "(a) Q4: diversity", "swarm diversity")
    draw_panel(right, args.results / "sweep_bounds.histories.jsonl", "boundary_hits",
               boundary_lines, "(b) Q5: boundary hits", "coordinates outside the bounds")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()