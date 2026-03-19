import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
from common import BenchmarkError, Metric, ModelContext
from suite_common import pick_primary_metric, run_logged_command


def _flatten_numeric_metrics(prefix: str, value: object, out: Dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_metrics(next_prefix, inner, out)


def _extract_lm_eval_metrics(payload: Dict[str, object], suite_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    groups = payload.get("groups")
    if isinstance(groups, dict) and isinstance(groups.get(suite_name), dict):
        _flatten_numeric_metrics("", groups[suite_name], metrics)
        if metrics:
            return metrics

    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get(suite_name), dict):
        _flatten_numeric_metrics("", results[suite_name], metrics)
        if metrics:
            return metrics

    if isinstance(results, dict):
        for task_name, task_metrics in results.items():
            if isinstance(task_metrics, dict) and (
                task_name == suite_name or str(task_name).startswith(f"{suite_name}_")
            ):
                _flatten_numeric_metrics(str(task_name), task_metrics, metrics)

    if not metrics:
        raise BenchmarkError(f"no numeric metrics found for suite {suite_name}")
    return metrics


def _find_result_payload(root: Path) -> Dict[str, object]:
    candidates = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and ("results" in payload or "groups" in payload):
            return payload
    raise BenchmarkError(f"could not find lm-eval JSON results under {root}")


def run_lm_eval_suite(
    ctx: ModelContext,
    tokenizer_id: str,
    suite_name: str,
    limit: int,
    fewshot: int,
) -> Tuple[str, str, List[Metric], str]:
    suite_dir = ctx.model_raw_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = suite_dir / "stdout.log"
    stderr_path = suite_dir / "stderr.log"

    base_url = f"http://{ctx.args.server_host}:{ctx.args.server_port}/v1/completions"
    model_args = ",".join(
        [
            f"model={ctx.model_slug}",
            f"base_url={base_url}",
            f"num_concurrent={ctx.args.num_concurrent}",
            f"max_retries={ctx.args.max_retries}",
            "tokenized_requests=False",
            f"max_length={ctx.args.ctx}",
            f"max_gen_toks={ctx.args.max_gen_toks}",
            f"seed={ctx.args.seed}",
        ]
    )
    if tokenizer_id:
        model_args += f",tokenizer={tokenizer_id},tokenizer_backend=huggingface"

    cmd = [
        ctx.args.lm_eval_bin,
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        suite_name,
        "--num_fewshot",
        str(fewshot),
        "--limit",
        str(limit),
        "--output_path",
        str(suite_dir),
    ]

    started = time.perf_counter()
    proc = run_logged_command(cmd, stdout_path, stderr_path, timeout_s=ctx.args.suite_timeout_s)
    runtime_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise BenchmarkError(f"lm-eval failed for {suite_name} (see {stderr_path})")

    payload = _find_result_payload(suite_dir)
    metrics = _extract_lm_eval_metrics(payload, suite_name)

    metric_rows = []
    for metric_name, metric_value in sorted(metrics.items()):
        metric_stderr = ""
        if metric_name.endswith("_stderr,none"):
            continue
        stderr_key = f"{metric_name}_stderr,none"
        if stderr_key in metrics:
            metric_stderr = f"{metrics[stderr_key]:.6f}"
        newRow = Metric(
                timestamp=ctx.ts_slug,
                model_path=ctx.model_path,
                model_name=ctx.model_slug,
                suite=suite_name,
                metric_name=metric_name,
                metric_value=f"{metric_value:.6f}",
                metric_stderr=metric_stderr,
                limit=limit,
                status="ok",
                error_note="",
            )
        metric_rows.append(newRow)

    primary_name, primary_value = pick_primary_metric(suite_name, metrics)
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"

