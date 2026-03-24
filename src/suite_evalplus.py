import ast
import gzip
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Dict, List, Tuple
from common import EVALPLUS_EVALUATE_BIN, SUITE_TIMEOUT, BenchmarkError, Metric, ModelContext, ensure_commands_exist
from suite_common import pick_primary_metric, run_logged_command



def _create_humaneval_subset(limit: int, output_path: Path) -> None:
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise BenchmarkError(
            "EvalPlus Python package is required to create a HumanEval+ subset"
        ) from exc

    problems = list(get_human_eval_plus().items())
    if not problems:
        raise BenchmarkError("EvalPlus returned no HumanEval+ tasks")
    subset = problems[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for task_id, payload in subset:
            row = {"task_id": task_id}
            row.update(payload)
            handle.write(json.dumps(row) + "\n")


def _create_mbpp_subset(limit: int, output_path: Path) -> None:
    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:
        raise BenchmarkError(
            "EvalPlus Python package is required to create an MBPP+ subset"
        ) from exc

    problems = list(get_mbpp_plus().items())
    if not problems:
        raise BenchmarkError("EvalPlus returned no MBPP+ tasks")
    subset = problems[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for task_id, payload in subset:
            row = {"task_id": task_id}
            row.update(payload)
            handle.write(json.dumps(row) + "\n")


def _parse_evalplus_stdout(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    base_match = re.search(r"Base\s*\n(\{[^\n]+\})", text, flags=re.MULTILINE)
    plus_match = re.search(r"Base \+ Extra\s*\n(\{[^\n]+\})", text, flags=re.MULTILINE)
    if base_match:
        payload = ast.literal_eval(base_match.group(1))
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    metrics[f"base.{key}"] = float(value)
    if plus_match:
        payload = ast.literal_eval(plus_match.group(1))
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    metrics[f"plus.{key}"] = float(value)
    if metrics:
        return metrics

    # Newer EvalPlus output format:
    # humaneval (base tests)
    # pass@1:    0.123
    # ...
    # humaneval+ (base + extra tests)
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        if line.endswith("(base tests)"):
            section = "base"
            continue
        if line.endswith("(base + extra tests)"):
            section = "plus"
            continue
        metric_match = re.match(r"^(pass@\d+):\s*([0-9]*\.?[0-9]+)$", line)
        if metric_match and section:
            metric_name = metric_match.group(1)
            metric_value = float(metric_match.group(2))
            metrics[f"{section}.{metric_name}"] = metric_value

    if not metrics:
        raise BenchmarkError("could not parse EvalPlus results from stdout")
    return metrics


def _raise_if_evalplus_stderr_has_errors(stderr_path: Path) -> None:
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if not stderr_text.strip():
        return

    lower_text = stderr_text.lower()
    fatal_markers = (
        "traceback (most recent call last):",
        "valueerror: current limit exceeds maximum limit",
        "process process-",
    )
    if any(marker in lower_text for marker in fatal_markers):
        raise BenchmarkError(
            f"EvalPlus reported runtime errors in stderr (see {stderr_path})"
        )


def run_humaneval(ctx: ModelContext) -> Tuple[str, str, List[Metric], str]:
    return run_humaneval_suite(ctx=ctx,
        suite_name="humaneval",
        limit=ctx.args.humaneval_limit,
        create_subset_func=_create_humaneval_subset,
        env_var_name="HUMANEVAL_OVERRIDE_PATH")


def run_mbpp(ctx: ModelContext) -> Tuple[str, str, List[Metric], str]:
    return run_humaneval_suite(
        ctx=ctx,
        suite_name="mbpp",
        limit=ctx.args.mbpp_limit,
        create_subset_func=_create_mbpp_subset,
        env_var_name="MBPP_OVERRIDE_PATH")


def run_humaneval_suite(
    ctx: ModelContext,
    suite_name: str,
    limit: int,
    create_subset_func: Callable[[int, Path], None],
    env_var_name: str
) -> Tuple[str, str, List[Metric], str]:
    ensure_commands_exist([EVALPLUS_EVALUATE_BIN])

    suite_dir = ctx.model_raw_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Create subset of HumanEval+ problems
    subset_path = suite_dir / f"{suite_name}_subset.jsonl.gz"
    create_subset_func(limit, subset_path)

    env = os.environ.copy()
    env[env_var_name] = str(subset_path)
    env.setdefault("OPENAI_API_KEY", "local-benchmark")
    # Avoid RLIMIT failures on some macOS setups during EvalPlus worker execution.
    env.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    base_url = f"http://{ctx.args.server_host}:{ctx.args.server_port}/v1"

    eval_stdout = suite_dir / "evaluate.stdout.log"
    eval_stderr = suite_dir / "evaluate.stderr.log"
    eval_cmd = [
        EVALPLUS_EVALUATE_BIN,
        "--dataset",
        suite_name,
        "--model",
        ctx.model_slug,
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--root",
        str(suite_dir / "results"),
        "--greedy",
    ]

    started = time.perf_counter()
    eval_proc = run_logged_command(
        eval_cmd,
        eval_stdout,
        eval_stderr,
        timeout_s=SUITE_TIMEOUT,
        env=env,
    )
    runtime_s = time.perf_counter() - started
    if eval_proc.returncode != 0:
        raise BenchmarkError(f"EvalPlus evaluation failed (see {eval_stderr})")
    _raise_if_evalplus_stderr_has_errors(eval_stderr)

    metrics = _parse_evalplus_stdout(eval_stdout.read_text(encoding="utf-8"))

    metric_rows = [Metric(timestamp=ctx.ts_slug,
                            model_path=ctx.model_path,
                            model_name=ctx.model_slug,
                            suite=suite_name,
                            metric_name=metric_name,
                            metric_value=f"{metric_value:.6f}",
                            metric_stderr="",
                            limit=limit,
                            status="ok",
                            error_note="",
                        ) for metric_name, metric_value in sorted(metrics.items())]
    primary_name, primary_value = pick_primary_metric(suite_name, metrics)
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"
