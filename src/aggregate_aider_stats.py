#!/usr/bin/env python3
"""Aggregate DeepSeek-style Aider benchmark runs into one CSV.

**Legacy mode** (default): scans ``results/performance`` for immediate child
run directories whose names match::

    YYYYMMDD-HHMMSS ds aider-<mode> <short>ctx <short>predict

Example folder name::

    20260405-172231 ds aider-whole 16ctx 8predict

For each matching run, reads ``run-meta.txt``, ``metrics.csv``, and
``suite_runs.csv``. ``mode`` comes from the folder name; ``ctx``, ``n_predict``,
and ``reasoning_budget`` come from ``run-meta.txt``. Metrics are taken from
``metrics.csv`` rows with ``suite == aider`` and pivoted into columns.
``runtime_s`` comes from the single ``suite == aider`` row in ``suite_runs.csv``.

**Target-folder mode** (``--target-folder``): scans **immediate children only**
of the given path; includes each subdirectory that contains ``run-meta.txt``,
``metrics.csv``, and ``suite_runs.csv`` together (any folder names). ``mode`` is taken from
``aider_mode`` in ``run-meta.txt``. The run ``timestamp`` slug is the directory
name if it matches ``YYYYMMDD-HHMMSS``, otherwise derived from
``run_timestamp_utc`` in ``run-meta.txt``.

Usage::

    uv run python src/aggregate_aider_stats.py
    uv run python src/aggregate_aider_stats.py --target-folder "results/performance/20260406 night runs"
    uv run python src/aggregate_aider_stats.py --out-csv results/performance/my-agg.csv

Default output path:

- Legacy: ``results/performance/aider-stats-<YYYYMMDD-HHMMSS>.csv``
- ``--target-folder``: ``<target-folder>/aider-stats-<YYYYMMDD-HHMMSS>.csv``

Output columns (fixed prefix, then all metric names sorted):

- ``timestamp``, ``mode``, ``ctx``, ``n_predict``, ``reasoning_budget``,
  ``runtime_s``, then every distinct ``metric_name`` from the included runs
  (values copied from ``metric_value``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import BenchmarkError, timestamp_slug, write_csv

DEFAULT_RESULTS_ROOT = Path("results/performance")

# Folder: "<ts> ds aider-<mode> <digits ctx label> <digits predict label>predict"
_RUN_DIR_RE = re.compile(
    r"^(\d{8}-\d{6})\s+ds\s+aider-(\S+)\s+(\S+)\s*ctx\s+(\S+)\s*predict$"
)

# Run directory whose name is only the timestamp slug (e.g. under a parent folder).
_TS_SLUG_NAME_RE = re.compile(r"^\d{8}-\d{6}$")


def _parse_run_meta(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _ts_slug_from_run_timestamp_utc(raw: str) -> str:
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    return f"{parsed.strftime('%Y%m%d')}-{parsed.strftime('%H%M%S')}"


def _derive_ts_slug_for_target_run(run_dir: Path, meta: Dict[str, str]) -> str:
    if _TS_SLUG_NAME_RE.match(run_dir.name):
        return run_dir.name
    if "run_timestamp_utc" not in meta or not meta["run_timestamp_utc"].strip():
        raise BenchmarkError(
            f"{run_dir}: cannot derive timestamp slug: folder name is not "
            f"YYYYMMDD-HHMMSS and run-meta.txt has no run_timestamp_utc"
        )
    return _ts_slug_from_run_timestamp_utc(meta["run_timestamp_utc"])


def _discover_run_dirs_in_target_folder(target: Path) -> List[Path]:
    out: List[Path] = []
    for child in sorted(target.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not (child / "run-meta.txt").is_file():
            continue
        if not (child / "metrics.csv").is_file():
            continue
        if not (child / "suite_runs.csv").is_file():
            continue
        out.append(child)
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
                    f"metrics timestamp mismatch in {path}: row {ts!r} vs expected {expected_ts!r}"
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
    suite_ts = (suite_row.get("timestamp") or "").strip()
    if suite_ts and suite_ts != ts:
        raise BenchmarkError(
            f"suite_runs timestamp mismatch in {suite_path}: "
            f"{suite_ts!r} vs expected {ts!r}"
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
        "--target-folder",
        type=Path,
        default=None,
        help=(
            "Aggregate runs in immediate subdirectories of this path only (each "
            "must have run-meta.txt, metrics.csv, suite_runs.csv). Mode from "
            "run-meta.txt aider_mode. Default output CSV is written under this folder."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: aider-stats-<YYYYMMDD-HHMMSS>.csv under "
            "results/performance (default mode) or under --target-folder when set."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    run_entries: List[Tuple[Path, str, str]] = []
    out_base: Path

    if args.target_folder is not None:
        root = args.target_folder
        if not root.is_dir():
            raise BenchmarkError(f"target folder is not a directory: {root}")
        out_base = root
        for run_dir in _discover_run_dirs_in_target_folder(root):
            meta_path = run_dir / "run-meta.txt"
            meta = _parse_run_meta(meta_path)
            mode = (meta.get("aider_mode") or "").strip()
            if not mode:
                raise BenchmarkError(
                    f"{meta_path}: missing aider_mode (required in --target-folder mode)"
                )
            ts = _derive_ts_slug_for_target_run(run_dir, meta)
            run_entries.append((run_dir, ts, mode))
    else:
        root = DEFAULT_RESULTS_ROOT
        if not root.is_dir():
            raise BenchmarkError(f"results directory is not a directory: {root}")
        out_base = root
        for child in root.iterdir():
            if not child.is_dir():
                continue
            m = _RUN_DIR_RE.match(child.name)
            if not m:
                continue
            ts, mode, _ctx_label, _predict_label = m.groups()
            run_entries.append((child, ts, mode))

    run_entries.sort(key=lambda t: (t[1], str(t[0].resolve())))
    if not run_entries:
        if args.target_folder is not None:
            raise BenchmarkError(
                f"no immediate child directories with run-meta.txt + metrics.csv "
                f"+ suite_runs.csv under {args.target_folder}"
            )
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
        out_path = out_base / f"aider-stats-{timestamp_slug()}.csv"

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
