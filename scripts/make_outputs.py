"""Statistics tables and one boxplot per study, from the results of `psoann sweep all`.

    python scripts/make_outputs.py [--results results]

Reads  <results>/sweep_<study>.csv
Writes <results>/stats/<study>__<metric>.txt                 Kruskal-Wallis, then Holm-corrected Mann-Whitney U
                                                             and A12 for every pair of settings (`psoann stats`)
       <results>/stats/<study>__<metric>__vs_<control>.txt   the same, each setting against one control
       <results>/stats/activations_evolved_choices.txt       which activations the evolved networks chose
       <results>/figures/<study>_box_accuracy.png            test accuracy over the runs, one box per setting
"""

import argparse
import contextlib
import io
from pathlib import Path
from typing import Final

import pandas as pd

from psoann.analysis.plots import plot_boxplot
from psoann.cli import main as psoann

STUDIES: Final[tuple[str, ...]] = (
    "random_search",
    "architecture",
    "swarm_vs_iterations",
    "coefficients",
    "informants",
    "bounds",
    "activations",
    "fitness",
    "schedule",
    "pilot_bound",
    "pilot_velocity",
)
METRICS: Final[tuple[str, ...]] = ("test_accuracy", "test_f1")
# Studies that have a natural reference setting are also compared with it alone (fewer tests, more power).
CONTROLS: Final[dict[str, str]] = {
    "architecture": "1x8",
    "informants": "random-k3",
    "bounds": "bound_handling=clamp",
    "activations": "tanh",
    "coefficients": "b1-g2-d1",
    "swarm_vs_iterations": "b2000-s20",
    "schedule": "schedule=constant",
}


def stats_report(csv: Path, metric: str, control: str | None) -> str:
    """Run `psoann stats` and return what it prints."""
    arguments = ["stats", str(csv), "--by", "point", "--metric", metric]
    if control:
        arguments += ["--control", control]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = psoann(arguments)
    if status != 0:
        raise SystemExit(f"psoann stats failed for {csv}")
    return output.getvalue()


def evolved_choices(results: pd.DataFrame) -> str:
    """Counts of the activation combinations (and per layer) chosen by the evolved networks."""
    chosen = results.loc[results["point"] == "evolved", "hidden_activations"]
    per_layer = chosen.str.split("-", expand=True)
    counts = pd.DataFrame({f"layer{i + 1}": per_layer[i].value_counts() for i in per_layer.columns}).fillna(0)
    return f"combinations:\n{chosen.value_counts().to_string()}\n\nper layer:\n{counts.astype(int).to_string()}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    stats_dir, figure_dir = args.results / "stats", args.results / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for study in STUDIES:
        csv = args.results / f"sweep_{study}.csv"
        for metric in METRICS:
            (stats_dir / f"{study}__{metric}.txt").write_text(stats_report(csv, metric, control=None))
        if study in CONTROLS:
            name = f"{study}__test_accuracy__vs_{CONTROLS[study]}.txt"
            (stats_dir / name).write_text(stats_report(csv, "test_accuracy", CONTROLS[study]))
        plot_boxplot(pd.read_csv(csv), "point", "test_accuracy", figure_dir / f"{study}_box_accuracy.png")

    activations = pd.read_csv(args.results / "sweep_activations.csv")
    (stats_dir / "activations_evolved_choices.txt").write_text(evolved_choices(activations))
    print(f"wrote {len(list(stats_dir.glob('*.txt')))} statistics files to {stats_dir} and boxplots to {figure_dir}")


if __name__ == "__main__":
    main()
