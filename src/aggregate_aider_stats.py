#!/usr/bin/env python3
"""Aggregate DeepSeek-style Aider benchmark runs into one CSV.

Scans a results root (default: ``results/performance``) for run directories whose
names match::

    YYYYMMDD-HHMMSS ds aider-<mode> <short>ctx <short>predict

Example folder name::

    20260405-172231 ds aider-whole 16ctx 8predict

For each matching run, reads ``run-meta.txt``, ``metrics.csv``, and
``suite_runs.csv``. ``mode`` comes from the folder name; ``ctx``, ``n_predict``,
and ``reasoning_budget`` come from ``run-meta.txt``. Metrics are taken from
``metrics.csv`` rows with ``suite == aider`` and pivoted into columns.
``runtime_s`` comes from the single ``suite == aider`` row in ``suite_runs.csv``.

Usage::

    uv run python src/aggregate_aider_stats.py
    uv run python src/aggregate_aider_stats.py --results-root results/performance
    uv run python src/aggregate_aider_stats.py --out-csv results/performance/my-agg.csv

Default output path::

    results/performance/aider-stats-<YYYYMMDD-HHMMSS>.csv

Output columns (fixed prefix, then all metric names sorted):

- ``timestamp``, ``mode``, ``ctx``, ``n_predict``, ``reasoning_budget``,
  ``runtime_s``, then every distinct ``metric_name`` from the included runs
  (values copied from ``metric_value``).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import BenchmarkError, timestamp_slug, write_csv

# Folder: "<ts> ds aider-<mode> <digits ctx label> <digits predict label>predict"
_RUN_DIR_RE = re.compile(
    r"^(\d{8}-\d{6})\s+ds\s+aider-(\S+)\s+(\S+)\s*ctx\s+(\S+)\s*predict$"
)


def _parse_run_meta(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _read_suite_run_aider(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("suite") or "").strip() == "aider"]
    if not rows:
        raise BenchmarkError(f"no suite=aider row in {path}")
    if len(rows) > 1:
        raise BenchmarkError(
            f"expected exactly one suite=aider row in {path}, found {len(rows)}"
        )
    return rows[0]


def _pivot_metrics_aider(path: Path, expected_ts: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("suite") or "").strip() != "aider":
                continue
            ts = (row.get("timestamp") or "").strip()
            if ts and ts != expected_ts:
                raise BenchmarkError(
                    f"metrics timestamp mismatch in {path}: row {ts!r} vs folder {expected_ts!r}"
                )
            name = (row.get("metric_name") or "").strip()
            if not name:
                continue
            val = (row.get("metric_value") or "").strip()
            if name in metrics and metrics[name] != val:
                raise BenchmarkError(
                    f"duplicate conflicting metric {name!r} in {path}"
                )
            metrics[name] = val
    return metrics


def _collect_run(
    run_dir: Path, ts: str, mode: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    meta_path = run_dir / "run-meta.txt"
    metrics_path = run_dir / "metrics.csv"
    suite_path = run_dir / "suite_runs.csv"
    for p in (meta_path, metrics_path, suite_path):
        if not p.is_file():
            raise BenchmarkError(f"missing required file: {p}")

    meta = _parse_run_meta(meta_path)
    for key in ("ctx", "n_predict", "reasoning_budget"):
        if key not in meta:
            raise BenchmarkError(f"{meta_path}: missing key {key!r}")

    suite_row = _read_suite_run_aider(suite_path)
    if (suite_row.get("timestamp") or "").strip() != ts:
        raise BenchmarkError(
            f"suite_runs timestamp mismatch in {suite_path}: "
            f"{suite_row.get('timestamp')!r} vs folder {ts!r}"
        )

    metric_values = _pivot_metrics_aider(metrics_path, ts)

    row: Dict[str, str] = {
        "timestamp": ts,
        "mode": mode,
        "ctx": meta["ctx"],
        "n_predict": meta["n_predict"],
        "reasoning_budget": meta["reasoning_budget"],
        "runtime_s": (suite_row.get("runtime_s") or "").strip(),
    }
    row.update(metric_values)
    return row, metric_values


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Aider benchmark runs into one wide CSV."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/performance"),
        help="Directory containing run folders (default: results/performance)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<results-root>/aider-stats-<YYYYMMDD-HHMMSS>.csv"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    root = args.results_root
    if not root.is_dir():
        raise BenchmarkError(f"results root is not a directory: {root}")

    run_entries: List[Tuple[Path, str, str]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = _RUN_DIR_RE.match(child.name)
        if not m:
            continue
        ts, mode, _ctx_label, _predict_label = m.groups()
        run_entries.append((child, ts, mode))

    run_entries.sort(key=lambda t: (t[1], t[0].name))
    if not run_entries:
        raise BenchmarkError(
            f"no run directories matching pattern under {root}"
        )

    all_metric_names: List[str] = []
    seen_metrics: set[str] = set()
    built_rows: List[Dict[str, str]] = []

    for run_dir, ts, mode in run_entries:
        row, metric_vals = _collect_run(run_dir, ts, mode)
        built_rows.append(row)
        for name in metric_vals.keys():
            if name not in seen_metrics:
                seen_metrics.add(name)
                all_metric_names.append(name)

    all_metric_names.sort()

    prefix = [
        "timestamp",
        "mode",
        "ctx",
        "n_predict",
        "reasoning_budget",
        "runtime_s",
    ]
    headers = prefix + all_metric_names

    out_path = args.out_csv
    if out_path is None:
        out_path = root / f"aider-stats-{timestamp_slug()}.csv"

    write_csv(out_path, headers)
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        for row in built_rows:
            out_row = {h: row.get(h, "") for h in headers}
            writer.writerow(out_row)

    print(f"Wrote {len(built_rows)} run(s) to {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
