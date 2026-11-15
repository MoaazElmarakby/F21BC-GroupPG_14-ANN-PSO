import pandas as pd
import pytest

from psoann.analysis.plots import plot_convergence, plot_sweep
from psoann.analysis.summary import load_results, summarise
from psoann.cli import main
from psoann.experiments import read_jsonl
from tests.conftest import REPO

DEFAULT = str(REPO / "configs" / "default.toml")
FAST = ["--set", "pso.swarm_size=6", "--set", "pso.iterations=4"]


@pytest.fixture
def results_dir(tmp_path):
    assert main(["run", DEFAULT, "--runs", "3", *FAST, "--out", str(tmp_path)]) == 0
    return tmp_path


def test_run_writes_csv_and_histories(results_dir):
    df = pd.read_csv(results_dir / "default.csv")
    assert len(df) == 3 and df["evaluations"].eq(24).all() and df["cfg.pso.swarm_size"].eq(6).all()
    assert len(read_jsonl(results_dir / "default.histories.jsonl")) == 3


def test_seed_and_overrides_from_the_command_line(tmp_path):
    main(["run", DEFAULT, "--runs", "1", "--seed", "7", *FAST, "--set", "ann.hidden_layers=[4,2]",
          "--set", "pso.informants=ring", "--out", str(tmp_path)])
    row = pd.read_csv(tmp_path / "default.csv").iloc[0]
    assert row["base_seed"] == 7 and row["cfg.ann.hidden_layers"] == "4-2" and row["cfg.pso.informants"] == "ring"


def test_bad_overrides_exit_with_an_error(tmp_path, capsys):
    assert main(["run", DEFAULT, "--set", "pso.alfa=1", "--out", str(tmp_path)]) == 2
    assert "unknown config key" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.toml"), "--out", str(tmp_path)]) == 2


def test_sweep_command(tmp_path):
    sweep = tmp_path / "s.toml"
    sweep.write_text(f'name = "s"\nbase = "{DEFAULT}"\n[grid]\n"pso.n_informants" = [2, 3]\n')
    assert main(["sweep", str(sweep), "--runs", "2", *FAST, "--out", str(tmp_path)]) == 0
    df = pd.read_csv(tmp_path / "sweep_s.csv")
    assert len(df) == 4 and set(df["cfg.pso.n_informants"]) == {2, 3}


def test_summarise_command_and_function(results_dir, capsys):
    assert main(["summarise", str(results_dir / "default.csv"), "--by", "cfg.pso.n_informants"]) == 0
    assert "test_accuracy_mean" in capsys.readouterr().out
    table = summarise(load_results(results_dir / "default.csv"))
    assert table.loc[0, "n_runs"] == 3 and {"test_accuracy_mean", "test_accuracy_std"} <= set(table.columns)
    with pytest.raises(KeyError, match="not in results"):
        summarise(load_results(results_dir / "default.csv"), by=["nope"])


def test_plots_are_written(results_dir, tmp_path):
    png = tmp_path / "conv.png"
    assert main(["plot-convergence", str(results_dir / "default.histories.jsonl"), "--out", str(png)]) == 0
    assert png.stat().st_size > 1000
    sweep_png = tmp_path / "sweep.png"
    assert main(["plot-sweep", str(results_dir / "default.csv"), "--x", "run", "--y", "test_accuracy", "--out", str(sweep_png)]) == 0
    assert sweep_png.stat().st_size > 1000
    assert plot_convergence(read_jsonl(results_dir / "default.histories.jsonl"), tmp_path / "m.png", key="mean").exists()
    with pytest.raises(KeyError):
        plot_sweep(load_results(results_dir / "default.csv"), "nope", "test_accuracy", tmp_path / "x.png")


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "tanh" in out and "ring" in out and "reflect" in out and "accuracy" in out
