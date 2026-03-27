import json
import os
import time
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

from common import (
    SUITE_TIMEOUT,
    BenchmarkError,
    Metric,
    ModelContext,
    ensure_commands_exist,
)
from suite_common import run_logged_command


def _summarize_aider_results(run_dir: Path) -> Dict[str, float]:
    result_files = sorted(run_dir.glob("*/exercises/practice/*/.aider.results.json"))
    if not result_files:
        raise BenchmarkError(f"could not find Aider result files under {run_dir}")

    rows: List[Dict[str, object]] = []
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)

    if not rows:
        raise BenchmarkError(f"could not parse Aider result files under {run_dir}")

    completed = len(rows)
    tries = max(len(_cast_list(r.get("tests_outcomes"))) for r in rows)
    if tries <= 0:
        raise BenchmarkError(f"Aider results had no test outcomes under {run_dir}")

    pass_counts: List[int] = [0 for _ in range(tries)]
    malformed_case_count = 0
    total_cost = 0.0
    total_duration = 0.0
    error_outputs = 0.0
    syntax_errors = 0.0
    indentation_errors = 0.0
    exhausted_context_windows = 0.0
    test_timeouts = 0.0
    prompt_tokens = 0.0
    completion_tokens = 0.0

    for row in rows:
        outcomes = _cast_list(row.get("tests_outcomes"))
        final_passed = bool(outcomes and outcomes[-1] is True)
        if final_passed:
            for idx in range(len(outcomes) - 1, tries):
                pass_counts[idx] += 1

        if _to_float(row.get("num_malformed_responses")) > 0:
            malformed_case_count += 1

        total_cost += _to_float(row.get("cost"))
        total_duration += _to_float(row.get("duration"))
        error_outputs += _to_float(row.get("num_error_outputs"))
        syntax_errors += _to_float(row.get("syntax_errors"))
        indentation_errors += _to_float(row.get("indentation_errors"))
        exhausted_context_windows += _to_float(row.get("num_exhausted_context_windows"))
        test_timeouts += _to_float(row.get("test_timeouts"))
        prompt_tokens += _to_float(row.get("prompt_tokens"))
        completion_tokens += _to_float(row.get("completion_tokens"))

    metrics: Dict[str, float] = {}
    for idx, passed in enumerate(pass_counts, start=1):
        metrics[f"pass_rate_{idx}"] = passed / completed
    metrics["pass_rate"] = pass_counts[-1] / completed
    metrics["completed_cases"] = float(completed)
    metrics["percent_cases_well_formed"] = 1.0 - (malformed_case_count / completed)
    metrics["total_cost"] = total_cost
    metrics["seconds_per_case"] = total_duration / completed
    metrics["error_outputs"] = error_outputs
    metrics["syntax_errors"] = syntax_errors
    metrics["indentation_errors"] = indentation_errors
    metrics["exhausted_context_windows"] = exhausted_context_windows
    metrics["test_timeouts"] = test_timeouts
    metrics["prompt_tokens"] = prompt_tokens
    metrics["completion_tokens"] = completion_tokens
    return metrics


def _cast_list(value: object) -> List[bool]:
    if not isinstance(value, list):
        return []
    return [bool(item) for item in value]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _validate_aider_setup(aider_repo_dir: Path) -> None:
    if not aider_repo_dir.is_dir():
        raise BenchmarkError(f"Aider repo directory not found: {aider_repo_dir}")

    benchmark_py = aider_repo_dir / "benchmark" / "benchmark.py"
    if not benchmark_py.is_file():
        raise BenchmarkError(f"Aider benchmark script not found: {benchmark_py}")

    exercises_dir = "polyglot-benchmark"
    exercises_host_dir = aider_repo_dir / "tmp.benchmarks" / exercises_dir
    if not exercises_host_dir.is_dir():
        raise BenchmarkError(
            f"Aider exercises directory not found: {exercises_host_dir}. "
            f"Clone {exercises_dir} under tmp.benchmarks first."
        )

def run_aider(ctx: ModelContext, full_mode: bool) -> Tuple[str, str, List[Metric], str]:
    ensure_commands_exist(["docker"])

    aider_repo_dir = Path(ctx.args.aider_repo_dir).expanduser().resolve()
    _validate_aider_setup(aider_repo_dir)

    rundir_prefix = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    rundir_name = f"{rundir_prefix}-run"
    rundir_path = aider_repo_dir / "tmp.benchmarks" / rundir_name

    suite_dir = ctx.model_raw_dir / "aider"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # This might be needed if we want to collect stats
    # benchmark_root_host_dir = suite_dir / "bench"
    # benchmark_root_host_dir.mkdir(parents=True, exist_ok=True)

    # aider_model_id = _resolve_aider_model_id(ctx)
    num_tests = -1 if full_mode else int(ctx.args.aider_limit)

    # Container launch command
    cmd: List[str] = [
        "docker",
        "run",
        "--rm", # remove on exit
        "--memory",
        "8g", # Docker memory limit passed to Aider benchmark container
        "--memory-swap",
        "8g", # If same as memory - effectively means no swap allowed
        "--add-host",
        "host.docker.internal:host-gateway", # to access local OpenAI-compatible endpoint
        "-v",
        f"{aider_repo_dir}:/aider", # not sure if needed if no changes in repo after container is built
        "-v",
        f"{aider_repo_dir}/tmp.benchmarks:/benchmarks", # exercises directory - here the outputs will be
        "-e",
        "AIDER_DOCKER=1",
        "-e",
        "AIDER_BENCHMARK_DIR=/benchmarks",
        "-e",
        f"OPENAI_API_BASE=http://host.docker.internal:{ctx.args.server_port}/v1",
        "-e",
        f"OPENAI_API_KEY=local-benchmark",
        "aider-benchmark", # docker image name
    ]

    # Command to run inside the container
    cmd.extend([
        "python3",
        "/aider/benchmark/benchmark.py",
        rundir_name, # run directory name
        "--model",
        f"openai/{ctx.model_slug}", # try instead of model id (aider does not have many supported OOB)
        "--edit-format",
        "whole",
        "--threads",
        "1",
        "--num-tests",
        str(num_tests),
        "--exercises-dir",
        "polyglot-benchmark",
    ])

    stdout_path = suite_dir / "run.stdout.log"
    stderr_path = suite_dir / "run.stderr.log"
    started = time.perf_counter()
    proc = run_logged_command(
        cmd=cmd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=SUITE_TIMEOUT,
        env=os.environ.copy(),
    )
    runtime_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise BenchmarkError(f"Aider benchmark failed (see {stderr_path})")

    metrics = _summarize_aider_results(rundir_path)

    metric_rows = [
        Metric(
            timestamp=ctx.ts_slug,
            model_path=ctx.model_path,
            model_name=ctx.model_slug,
            suite="aider",
            metric_name=metric_name,
            metric_value=f"{metric_value:.6f}",
            metric_stderr="",
            limit=ctx.args.aider_limit,
            status="ok",
            error_note="",
        )
        for metric_name, metric_value in sorted(metrics.items())
    ]

    primary_name = "pass_rate"
    primary_value = f"{metrics[primary_name]:.6f}"
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"
