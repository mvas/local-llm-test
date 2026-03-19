import ast
import gzip
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Tuple
from common import BenchmarkError, Metric, ModelContext, ensure_commands_exist
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
    if not metrics:
        raise BenchmarkError("could not parse EvalPlus results from stdout")
    return metrics


def _find_latest_matching(root: Path, pattern: str) -> Path:
    candidates = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise BenchmarkError(f"could not find {pattern} under {root}")
    return candidates[0]


def run_humaneval_suite(
    ctx: ModelContext,
) -> Tuple[str, str, List[Metric], str]:
    ensure_commands_exist([ctx.args.evalplus_codegen_bin, ctx.args.evalplus_evaluate_bin])

    suite_dir = ctx.model_raw_dir / "humaneval"

    suite_dir.mkdir(parents=True, exist_ok=True)
    subset_path = suite_dir / "humaneval_subset.jsonl.gz"
    _create_humaneval_subset(ctx.args.humaneval_limit, subset_path)

    env = os.environ.copy()
    env["HUMANEVAL_OVERRIDE_PATH"] = str(subset_path)
    env.setdefault("OPENAI_API_KEY", "local-benchmark")
    base_url = f"http://{ctx.args.server_host}:{ctx.args.server_port}/v1"

    codegen_stdout = suite_dir / "codegen.stdout.log"
    codegen_stderr = suite_dir / "codegen.stderr.log"
    codegen_cmd = [
        ctx.args.evalplus_codegen_bin,
        "--model",
        ctx.model_slug,
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--dataset",
        "humaneval",
        "--root",
        str(suite_dir / "codegen"),
        "--greedy",
    ]

    started = time.perf_counter()
    codegen_proc = run_logged_command(
        codegen_cmd,
        codegen_stdout,
        codegen_stderr,
        timeout_s=ctx.args.suite_timeout_s,
        env=env,
    )
    if codegen_proc.returncode != 0:
        raise BenchmarkError(f"EvalPlus code generation failed (see {codegen_stderr})")

    samples_path = _find_latest_matching(suite_dir / "codegen", "*.jsonl")
    eval_stdout = suite_dir / "evaluate.stdout.log"
    eval_stderr = suite_dir / "evaluate.stderr.log"
    eval_cmd = [
        ctx.args.evalplus_evaluate_bin,
        "--dataset",
        "humaneval",
        "--samples",
        str(samples_path),
    ]

    eval_proc = run_logged_command(
        eval_cmd,
        eval_stdout,
        eval_stderr,
        timeout_s=ctx.args.suite_timeout_s,
        env=env,
    )
    runtime_s = time.perf_counter() - started
    if eval_proc.returncode != 0:
        raise BenchmarkError(f"EvalPlus evaluation failed (see {eval_stderr})")

    metrics = _parse_evalplus_stdout(eval_stdout.read_text(encoding="utf-8"))

    metric_rows = [Metric(timestamp=ctx.ts_slug,
                            model_path=ctx.model_path,
                            model_name=ctx.model_slug,
                            suite="humaneval",
                            metric_name=metric_name,
                            metric_value=f"{metric_value:.6f}",
                            metric_stderr="",
                            limit=ctx.args.humaneval_limit,
                            status="ok",
                            error_note="",
                        ) for metric_name, metric_value in sorted(metrics.items())]
    primary_name, primary_value = pick_primary_metric("humaneval", metrics)
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"


