import os
from pathlib import Path
import re
import time
from typing import List, Tuple
from common import BenchmarkError, Metric, ModelContext
from suite_common import pick_primary_metric, run_logged_command


def run_external_template_suite(
    suite_name: str,
    template: str,
    ctx: ModelContext,
    limit: int,
) -> Tuple[str, str, List[Metric], str]:
    if not template:
        raise BenchmarkError(f"{suite_name} command template was not provided")

    suite_dir = ctx.model_raw_dir / suite_name

    suite_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = suite_dir / "stdout.log"
    stderr_path = suite_dir / "stderr.log"
    base_url = f"http://{ctx.args.server_host}:{ctx.args.server_port}/v1"
    command_text = template.format(
        model_name=ctx.model_slug,
        model_path=ctx.model_path,
        base_url=base_url,
        host=ctx.args.server_host,
        port=ctx.args.server_port,
        suite_dir=suite_dir,
        limit=limit,
        ctx=ctx.args.ctx,
        seed=ctx.args.seed,
    )

    started = time.perf_counter()
    proc = run_logged_command(
        [command_text],
        stdout_path,
        stderr_path,
        timeout_s=ctx.args.suite_timeout_s,
        env=os.environ.copy(),
        shell=True,
    )
    runtime_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise BenchmarkError(f"{suite_name} command failed (see {stderr_path})")

    raw_metrics = re.findall(r"([A-Za-z0-9_@./+-]+)=([0-9]+(?:\.[0-9]+)?)", stdout_path.read_text(encoding="utf-8"))
    metrics = {name: float(value) for name, value in raw_metrics}
    if not metrics:
        metrics = {"exit_code": 0.0}

    primary_name, primary_value = pick_primary_metric(suite_name, metrics)
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
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"

