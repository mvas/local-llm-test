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
    expand_model_path,
)
from suite_common import run_logged_command


def _read_aider_model_map_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise BenchmarkError(
                f"invalid Aider model map entry at {path}:{line_no}; "
                "expected MODEL_PATH=AIDER_MODEL_ID"
            )
        model_text, aider_model_id = line.split("=", 1)
        model_path = expand_model_path(model_text)
        aider_model_id = aider_model_id.strip()
        if not aider_model_id:
            raise BenchmarkError(f"empty Aider model id at {path}:{line_no}")
        mapping[model_path] = aider_model_id
    return mapping


def _resolve_aider_model_id(ctx: ModelContext) -> str:
    map_file = str(getattr(ctx.args, "aider_model_map_file", "")).strip()
    if not map_file:
        return ctx.model_slug
    map_path = Path(map_file).expanduser()
    if not map_path.is_file():
        raise BenchmarkError(f"Aider model map file not found: {map_path}")
    mapping = _read_aider_model_map_file(map_path)
    aider_model_id = mapping.get(ctx.model_path, "")
    if not aider_model_id:
        raise BenchmarkError(
            f"no Aider model id mapping for model path: {ctx.model_path} "
            f"(in {map_path})"
        )
    return aider_model_id


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

        if float(row.get("num_malformed_responses", 0) or 0) > 0:
            malformed_case_count += 1

        total_cost += float(row.get("cost", 0) or 0)
        total_duration += float(row.get("duration", 0) or 0)
        error_outputs += float(row.get("num_error_outputs", 0) or 0)
        syntax_errors += float(row.get("syntax_errors", 0) or 0)
        indentation_errors += float(row.get("indentation_errors", 0) or 0)
        exhausted_context_windows += float(row.get("num_exhausted_context_windows", 0) or 0)
        test_timeouts += float(row.get("test_timeouts", 0) or 0)
        prompt_tokens += float(row.get("prompt_tokens", 0) or 0)
        completion_tokens += float(row.get("completion_tokens", 0) or 0)

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


def run_aider(ctx: ModelContext, full_mode: bool) -> Tuple[str, str, List[Metric], str]:
    ensure_commands_exist(["docker"])


    aider_repo_dir = Path(ctx.args.aider_repo_dir).expanduser().resolve()
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

    rundir_prefix = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    suite_dir = ctx.model_raw_dir / "aider"
    suite_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root_host_dir = suite_dir / "bench"
    benchmark_root_host_dir.mkdir(parents=True, exist_ok=True)
    run_dir_name = f"{ctx.ts_slug}--{ctx.model_slug}--aider"
    run_dir_container = Path("/benchmarks") / run_dir_name
    run_dir_host = benchmark_root_host_dir / run_dir_name

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
        "python",
        "/aider/benchmark/benchmark.py",
        f"{rundir_prefix}-run", # run directory name
        "--model",
        f"openai/{ctx.model_slug}", # try instead of model id (aider does not have many supported OOB)
        "--edit-format",
        "whole",
        "--threads",
        "1",
        # "--tries",
        # "2",
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

    metrics = _summarize_aider_results(run_dir_host)
    primary_name = "pass_rate"
    primary_value = f"{metrics[primary_name]:.6f}"
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
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"
