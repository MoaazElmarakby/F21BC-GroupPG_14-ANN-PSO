"""Result files: one CSV row per run and a JSONL file with per-iteration histories (stdlib only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

Row = dict[str, Any]


def _columns(rows: Sequence[Row]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def write_csv(path: str | Path, rows: Sequence[Row]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_columns(rows), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def read_csv(path: str | Path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_jsonl(path: str | Path, records: Iterable[Row]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Row]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_results(out_dir: str | Path, name: str, rows: Sequence[Row], histories: Iterable[Row]) -> tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.histories.jsonl`` into ``out_dir``; return both paths."""
    out_dir = Path(out_dir)
    return (
        write_csv(out_dir / f"{name}.csv", rows),
        write_jsonl(out_dir / f"{name}.histories.jsonl", histories),
    )
