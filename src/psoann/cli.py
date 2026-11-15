"""Command line interface: ``psoann run | sweep | summarise | stats | plot-* | list``."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from psoann.config import ExperimentConfig

if TYPE_CHECKING:
    from psoann.experiments.sweep import Sweep


def _parse_assignments(items: Sequence[str]) -> dict[str, Any]:
    """``--set pso.alpha=0.5 --set ann.hidden_layers=[8,4]`` -> {"pso.alpha": 0.5, ...}."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw  # bare strings such as pso.informants=ring
    return out


def _progress(done: int, total: int) -> None:
    print(f"  run {done}/{total}", file=sys.stderr, flush=True)


def _cmd_run(args: argparse.Namespace) -> int:
    from psoann.experiments import run_experiment, write_results

    config = ExperimentConfig(name=args.name)
    overrides = _parse_assignments(args.set)
    if args.runs is not None:
        overrides["n_runs"] = args.runs
    if args.seed is not None:
        overrides["base_seed"] = args.seed
    if overrides:
        config = config.with_overrides(overrides)
    result = run_experiment(config, progress=_progress)
    csv_path, hist_path = write_results(args.out, config.name, result.rows, result.histories)
    test = [row["test_accuracy"] for row in result.rows]
    print(f"{config.name}: {len(test)} runs, mean test accuracy {sum(test) / len(test):.4f}")
    print(f"wrote {csv_path} and {hist_path}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from psoann.experiments import SWEEPS, get_sweep

    names = sorted(SWEEPS) if args.sweep == "all" else [args.sweep]
    for name in names:
        _run_one_sweep(get_sweep(name), args)
    return 0


def _run_one_sweep(sweep: Sweep, args: argparse.Namespace) -> None:
    from psoann.experiments import run_sweep, write_results

    overrides = _parse_assignments(args.set)
    if args.runs is not None:
        overrides["n_runs"] = args.runs
    if overrides:
        sweep = dataclasses.replace(sweep, base=sweep.base.with_overrides(overrides))
    n_points = len(sweep.expand())
    print(f"sweep {sweep.name!r}: {n_points} point(s) x {sweep.base.n_runs} run(s)", file=sys.stderr)
    rows, histories = run_sweep(sweep, progress=lambda i, n, p: print(f"  point {i}/{n}: {p.point_id}", file=sys.stderr, flush=True))
    csv_path, hist_path = write_results(args.out, f"sweep_{sweep.name}", rows, histories)
    print(f"wrote {csv_path} and {hist_path}")


def _cmd_summarise(args: argparse.Namespace) -> int:
    from psoann.analysis.summary import load_results, summarise

    table = summarise(load_results(args.results), by=args.by or None, metrics=args.metric or None)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


def _cmd_plot_convergence(args: argparse.Namespace) -> int:
    from psoann.analysis.plots import plot_convergence
    from psoann.experiments import read_jsonl

    print(f"wrote {plot_convergence(read_jsonl(args.histories), args.out, key=args.key, group_by=args.group_by)}")
    return 0


def _cmd_plot_sweep(args: argparse.Namespace) -> int:
    from psoann.analysis.plots import plot_sweep
    from psoann.analysis.summary import load_results

    print(f"wrote {plot_sweep(load_results(args.results), args.x, args.y, args.out, hue=args.hue)}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from psoann.analysis.stats import kruskal_wallis, pairwise_tests
    from psoann.analysis.summary import load_results

    results = load_results(args.results)
    omnibus = kruskal_wallis(results, args.by, args.metric)
    print(f"Kruskal-Wallis on {args.metric} by {args.by}: H={omnibus['H']:.3f}, p={omnibus['p']:.4g} "
          f"({int(omnibus['groups'])} groups, n={int(omnibus['n'])})")
    table = pairwise_tests(results, args.by, args.metric, control=args.control)
    kind = f"against control {args.control!r}" if args.control else "all pairs"
    print(f"Mann-Whitney U, Holm-corrected ({kind}); a12 = P(group_a beats group_b)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


def _cmd_plot_box(args: argparse.Namespace) -> int:
    from psoann.analysis.plots import plot_boxplot
    from psoann.analysis.summary import load_results

    print(f"wrote {plot_boxplot(load_results(args.results), args.by, args.y, args.out)}")
    return 0


def _cmd_plot_heatmap(args: argparse.Namespace) -> int:
    from psoann.analysis.plots import plot_heatmap
    from psoann.analysis.summary import load_results

    print(f"wrote {plot_heatmap(load_results(args.results), args.x, args.y, args.value, args.out)}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    from psoann.ann import available_activations
    from psoann.experiments import SWEEPS
    from psoann.evaluation import available_fitness_metrics
    from psoann.pso import available_boundary_handlers, available_informant_strategies

    for title, names in (
        ("activations", available_activations()),
        ("informant strategies", available_informant_strategies()),
        ("boundary handlers", available_boundary_handlers()),
        ("fitness metrics", available_fitness_metrics()),
        ("sweeps", sorted(SWEEPS)),
    ):
        print(f"{title}: {', '.join(names)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psoann", description="Train an ANN with PSO (Algorithm 39).")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one experiment (default config + overrides)")
    run.add_argument("--name", default="default", help="experiment name, used for the output file names")
    run.add_argument("--runs", type=int, help="override n_runs")
    run.add_argument("--seed", type=int, help="override base_seed")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key, e.g. pso.alpha=0.5")
    run.add_argument("--out", type=Path, default=Path("results"))
    run.set_defaults(func=_cmd_run)

    sweep = sub.add_parser("sweep", help="run a built-in parameter sweep by name, or 'all'")
    sweep.add_argument("sweep", metavar="NAME", help="a sweep name (see `psoann list`) or 'all'")
    sweep.add_argument("--runs", type=int, help="override n_runs for every point")
    sweep.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a base config key")
    sweep.add_argument("--out", type=Path, default=Path("results"))
    sweep.set_defaults(func=_cmd_sweep)

    summ = sub.add_parser("summarise", help="mean/std per group from a results CSV")
    summ.add_argument("results", type=Path)
    summ.add_argument("--by", action="append", default=[], help="grouping column, e.g. cfg.pso.n_informants or point (repeatable)")
    summ.add_argument("--metric", action="append", default=[], help="metric column to report, e.g. test_f1 (repeatable; default: the standard set)")
    summ.set_defaults(func=_cmd_summarise)

    conv = sub.add_parser("plot-convergence", help="plot best-so-far fitness from a histories JSONL")
    conv.add_argument("histories", type=Path)
    conv.add_argument("--out", type=Path, required=True)
    conv.add_argument("--key", default="best", choices=["best", "mean", "iteration_best", "diversity", "boundary_hits"])
    conv.add_argument("--group-by", help="record field to group curves by (default: point/experiment)")
    conv.set_defaults(func=_cmd_plot_convergence)

    plot = sub.add_parser("plot-sweep", help="plot a metric against a swept column")
    plot.add_argument("results", type=Path)
    plot.add_argument("--x", required=True)
    plot.add_argument("--y", default="test_accuracy")
    plot.add_argument("--hue")
    plot.add_argument("--out", type=Path, required=True)
    plot.set_defaults(func=_cmd_plot_sweep)

    st = sub.add_parser("stats", help="Kruskal-Wallis + Holm-corrected Mann-Whitney + A12 between groups")
    st.add_argument("results", type=Path)
    st.add_argument("--by", default="point", help="grouping column (default: point)")
    st.add_argument("--metric", default="test_accuracy")
    st.add_argument("--control", help="compare only this group against the others (fewer tests, more power)")
    st.set_defaults(func=_cmd_stats)

    box = sub.add_parser("plot-box", help="boxplot of a metric per group over the repeated runs")
    box.add_argument("results", type=Path)
    box.add_argument("--by", default="point")
    box.add_argument("--y", default="test_accuracy")
    box.add_argument("--out", type=Path, required=True)
    box.set_defaults(func=_cmd_plot_box)

    heat = sub.add_parser("plot-heatmap", help="heatmap of the mean of a metric over two swept columns")
    heat.add_argument("results", type=Path)
    heat.add_argument("--x", required=True)
    heat.add_argument("--y", required=True)
    heat.add_argument("--value", default="test_accuracy")
    heat.add_argument("--out", type=Path, required=True)
    heat.set_defaults(func=_cmd_plot_heatmap)

    lst = sub.add_parser("list", help="show registered activations, topologies, boundary handlers, metrics and sweeps")
    lst.set_defaults(func=_cmd_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
