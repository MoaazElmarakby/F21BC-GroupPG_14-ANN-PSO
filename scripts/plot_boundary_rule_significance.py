"""Which settings of a study really differ? Accuracy ranking with groups, and the test for every pair.

    python scripts/plot_boundary_rule_significance.py [--results results] [--study bounds]
                                                      [--metric test_accuracy] [--out FILE.png]

Reads  <results>/sweep_<study>.csv  (written by `psoann sweep <study>`; the default study is `bounds`)
Writes <results>/figures/<study>_significance.png
       (a) mean +/- std of the metric per setting, best first, with a group letter: settings that
           share a letter are not significantly different;
       (b) every pair of settings: "different" or "no difference", with its Holm-corrected p-value.

The tests are the ones `psoann stats` prints: two-sided Mann-Whitney U for all pairs, Holm-corrected.
"""
import argparse
from itertools import combinations
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from psoann.analysis.stats import pairwise_tests

significance_level: Final[float] = 0.05
max_settings_for_letters: Final[int] = 14  # the grouping below checks every subset of settings
ink: Final[str] = "#1f2430"
muted: Final[str] = "#5b6472"
grid_colour: Final[str] = "#e3e2dc"
mark_colour: Final[str] = "#2a78d6"
different_fill: Final[str] = "#cde2fb"
same_fill: Final[str] = "#eeeeea"


def short_label(point: str) -> str:
    """'bound_handling=clamp' -> 'clamp'; labels without '=' are kept."""
    return point.split("=", 1)[-1]


def format_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    exponent = int(np.floor(np.log10(p)))
    return f"{p / 10**exponent:.1f}e{exponent}"


def group_letters(order: list[str], different: dict[frozenset[str], bool]) -> dict[str, str]:
    """Letters such that two settings share one exactly when they are NOT significantly different.

    Each letter is a largest possible set of settings in which no pair differs; ``order`` is best first,
    so the letter A goes to the set holding the best setting.
    """
    def all_alike(members: tuple[str, ...]) -> bool:
        return not any(different[frozenset(pair)] for pair in combinations(members, 2))

    alike_sets = [set(c) for size in range(1, len(order) + 1) for c in combinations(order, size) if all_alike(c)]
    largest = [s for s in alike_sets if not any(s < other for other in alike_sets)]
    largest.sort(key=lambda s: sorted(order.index(member) for member in s))
    letters: dict[str, list[str]] = {setting: [] for setting in order}
    for index, members in enumerate(largest):
        for setting in members:
            letters[setting].append(chr(ord("A") + index))
    return {setting: " ".join(found) for setting, found in letters.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--study", default="bounds")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.results / "figures" / f"{args.study}_significance.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results / f"sweep_{args.study}.csv")
    summary = results.groupby("point")[args.metric].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    order = list(summary.index)  # best first
    tests = pairwise_tests(results, by="point", metric=args.metric)  # all pairs, Holm-corrected
    p_holm = {frozenset((row.group_a, row.group_b)): float(row.p_holm) for row in tests.itertuples()}
    different = {pair: p < significance_level for pair, p in p_holm.items()}
    letters = group_letters(order, different) if len(order) <= max_settings_for_letters else {}
    runs = int(summary["count"].min())

    count = len(order)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 0.55 * count + 1.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) mean +/- std per setting, best at the top, group letter at the right
    rows = np.arange(count)[::-1]
    low = float((summary["mean"] - summary["std"]).min())
    high = float((summary["mean"] + summary["std"]).max())
    span = high - low
    letter_x = high + 0.42 * span
    for setting, row in zip(order, rows):
        mean, std = float(summary.loc[setting, "mean"]), float(summary.loc[setting, "std"])
        left.plot([mean - std, mean + std], [row, row], color=mark_colour, linewidth=2, solid_capstyle="round")
        left.plot(mean, row, "o", color=mark_colour, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        left.text(mean + std + 0.03 * span, row, f"{mean:.3f}", va="center", fontsize=9, color=ink)
        if letters:
            left.text(letter_x, row, letters[setting], va="center", ha="center", fontsize=11, fontweight="bold", color=ink)
    if letters:
        left.text(letter_x, count - 0.35, "group", ha="center", fontsize=9, color=muted)
    left.set_yticks(rows, [short_label(setting) for setting in order], fontsize=10)
    left.set_xlim(low - 0.08 * span, high + 0.55 * span)
    left.set_ylim(-0.6, count - 0.1)
    left.set_xticks([tick for tick in left.get_xticks() if low - 0.08 * span <= tick <= high + 0.05 * span])
    left.set_xlabel(f"{args.metric.replace('_', ' ')} (mean ± std over {runs} runs)", color=ink)
    left.set_title(f"(a) {args.metric.replace('_', ' ').capitalize()} by setting", fontsize=11, loc="left", color=ink)
    left.grid(axis="x", color=grid_colour, linewidth=0.8)
    left.set_axisbelow(True)
    for side in ("top", "right", "left"):
        left.spines[side].set_visible(False)
    left.tick_params(axis="y", length=0)

    # (b) lower triangle: one cell per pair
    for i, row_setting in enumerate(order):
        for j, column_setting in enumerate(order[:i]):
            p = p_holm[frozenset((row_setting, column_setting))]
            is_different = p < significance_level
            right.add_patch(Rectangle(
                (j + 0.03, count - 1 - i + 0.03), 0.94, 0.94,
                facecolor=different_fill if is_different else same_fill,
                edgecolor=mark_colour if is_different else "none", linewidth=1.2,
            ))
            text_colour = ink if is_different else muted
            right.text(j + 0.5, count - 1 - i + 0.62, "different" if is_different else "no difference",
                       ha="center", va="center", fontsize=8.5, fontweight="bold" if is_different else "normal", color=text_colour)
            right.text(j + 0.5, count - 1 - i + 0.32, f"p = {format_p(p)}", ha="center", va="center", fontsize=8, color=text_colour)
    right.set_xlim(0, count - 1)
    right.set_ylim(0, count - 1)
    right.set_xticks(np.arange(count - 1) + 0.5, [short_label(s) for s in order[:-1]], fontsize=10)
    right.set_yticks(np.arange(count - 1) + 0.5, [short_label(s) for s in order[:0:-1]], fontsize=10)
    right.tick_params(length=0)
    right.xaxis.tick_top()
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title("(b) Is each pair really different?  (Holm-corrected p)", fontsize=11, loc="left", color=ink, pad=24)

    fig.text(0.5, 0.015,
             f"Settings that share a group letter are not significantly different "
             f"(Mann-Whitney U, Holm-corrected, threshold {significance_level}).",
             ha="center", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=300)

    table = summary.assign(group=pd.Series(letters)).rename(index=short_label) if letters else summary.rename(index=short_label)
    print(table.round(4).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()