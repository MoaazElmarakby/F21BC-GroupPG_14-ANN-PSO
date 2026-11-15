"""Enforce the dependency rule from docs/design.md section 2 by reading the import statements."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "psoann"

# package -> psoann sub-packages/modules it may import (everything else is forbidden)
ALLOWED = {
    "config": set(),
    "registry": set(),
    "data": {"config"},
    "ann": {"config", "registry"},
    "pso": {"config", "registry"},
    "evaluation": {"registry"},
    "training": {"config", "ann", "pso", "evaluation"},
    "experiments": {"config", "data", "pso", "training", "evaluation", "ann"},
    "analysis": {"experiments", "config"},
    "cli": {"config", "experiments", "analysis", "ann", "pso", "evaluation"},
}


def imported_psoann_parts(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "psoann" and len(parts) > 1:
                found.add(parts[1])
    return found


def sources(package: str):
    target = SRC / package
    return sorted(target.rglob("*.py")) if target.is_dir() else [SRC / f"{package}.py"]


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_package_imports_only_what_the_design_allows(package):
    for path in sources(package):
        illegal = imported_psoann_parts(path) - ALLOWED[package] - {package}
        assert not illegal, f"{path.relative_to(SRC)} imports {sorted(illegal)}"


def test_ann_and_pso_never_import_each_other():
    for path in sources("ann"):
        assert "pso" not in imported_psoann_parts(path)
    for path in sources("pso"):
        assert "ann" not in imported_psoann_parts(path)


def test_every_source_file_is_covered_by_the_rule():
    top = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")}
    top |= {p.stem for p in SRC.glob("*.py") if not p.stem.startswith("__")}
    assert top <= set(ALLOWED), f"add a rule for: {sorted(top - set(ALLOWED))}"
