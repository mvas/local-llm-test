#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, fields
import gzip
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
from typing import Dict, List, Optional, Tuple

from common import ( 
    now_iso,
    timestamp_slug,
    slugify_model,
    expand_model_path,
    read_models_file,
    BenchmarkError,
    ensure_commands_exist,
    write_csv,
    append_csv_row,
    wait_for_health
)


DEFAULTS = {
    "ngl": 99,
    "ctx": 8192,
    "temp": 0.0,
    "top_p": 1.0,
    "seed": 1234,
    "max_gen_toks": 512,
    "num_concurrent": 1,
    "max_retries": 3,
    "suite_timeout_s": 3600,
    "server_host": "127.0.0.1",
    "server_port": 8082,
    "out_dir_base": "results/performance",
    "llama_server_bin": os.environ.get("LLAMA_SERVER_BIN", "llama-server"),
    "lm_eval_bin": os.environ.get("LM_EVAL_BIN", "lm_eval"),
    "evalplus_codegen_bin": os.environ.get("EVALPLUS_CODEGEN_BIN", "evalplus.codegen"),
    "evalplus_evaluate_bin": os.environ.get("EVALPLUS_EVALUATE_BIN", "evalplus.evaluate"),
    # Quick-suite defaults sized to keep the total runtime reasonable on local hardware.
    "gsm8k_limit": 30,
    "gsm8k_fewshot": 4,
    "mmlu_limit": 100,
    "mmlu_fewshot": 5,
    "ifeval_limit": 40,
    "ifeval_fewshot": 0,
    "humaneval_limit": 20,
    "run_humaneval": True,
}


LM_EVAL_SUITES = (
    ("gsm8k", "gsm8k", DEFAULTS["gsm8k_limit"], DEFAULTS["gsm8k_fewshot"]),
    ("mmlu", "mmlu", DEFAULTS["mmlu_limit"], DEFAULTS["mmlu_fewshot"]),
    ("ifeval", "ifeval", DEFAULTS["ifeval_limit"], DEFAULTS["ifeval_fewshot"]),
)

@dataclass(frozen=True)
class MetricRow:
    timestamp: str
    model_path: str
    model_name: str
    suite: str
    task_name: str
    metric_name: str
    metric_value: str
    metric_stderr: str
    limit: int
    status: str
    error_note: str

    @classmethod
    def headers(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryRow:
    timestamp: str
    model_path: str
    model_name: str
    ctx: int
    ngl: int
    temperature: float
    top_p: float
    seed: int
    gsm8k_primary_metric: str
    mmlu_primary_metric: str
    ifeval_primary_metric: str
    humaneval_plus_pass_at_1: str
    bfcl_primary_metric: str
    aider_primary_metric: str
    status: str
    error_note: str

    @classmethod
    def headers(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteRunRow:
    timestamp: str
    model_path: str
    model_name: str
    suite: str
    limit: int
    status: str
    runtime_s: str
    primary_metric_name: str
    primary_metric_value: str
    error_note: str

    @classmethod
    def headers(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


METRIC_FIELDS = MetricRow.headers()
SUMMARY_FIELDS = SummaryRow.headers()
SUITE_RUN_FIELDS = SuiteRunRow.headers()


def read_tokenizer_map_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise BenchmarkError(
                f"invalid tokenizer map entry at {path}:{line_no}; expected MODEL_PATH=TOKENIZER_ID"
            )
        model_text, tokenizer_id = line.split("=", 1)
        model_path = expand_model_path(model_text)
        tokenizer_id = tokenizer_id.strip()
        if not tokenizer_id:
            raise BenchmarkError(f"empty tokenizer id at {path}:{line_no}")
        mapping[model_path] = tokenizer_id
    return mapping


def write_meta(path: Path, args: argparse.Namespace, resolved_models: List[str]) -> None:
    lines = [
        f"run_timestamp_utc={now_iso()}",
        f"models_file={args.models_file}",
        f"tokenizer_map_file={args.tokenizer_map_file or ''}",
        f"models_count={len(resolved_models)}",
        f"ctx={args.ctx}",
        f"ngl={args.ngl}",
        f"temperature={args.temp}",
        f"top_p={args.top_p}",
        f"seed={args.seed}",
        f"max_gen_toks={args.max_gen_toks}",
        f"server_host={args.server_host}",
        f"server_port={args.server_port}",
        f"llama_server_bin={args.llama_server_bin}",
        f"lm_eval_bin={args.lm_eval_bin}",
        f"evalplus_codegen_bin={args.evalplus_codegen_bin}",
        f"evalplus_evaluate_bin={args.evalplus_evaluate_bin}",
        f"gsm8k_limit={args.gsm8k_limit}",
        f"gsm8k_fewshot={args.gsm8k_fewshot}",
        f"mmlu_limit={args.mmlu_limit}",
        f"mmlu_fewshot={args.mmlu_fewshot}",
        f"ifeval_limit={args.ifeval_limit}",
        f"ifeval_fewshot={args.ifeval_fewshot}",
        f"humaneval_limit={args.humaneval_limit}",
        f"run_humaneval={args.run_humaneval}",
        f"bfcl_command_template={args.bfcl_command_template or ''}",
        f"aider_command_template={args.aider_command_template or ''}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flatten_numeric_metrics(prefix: str, value: object, out: Dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric_metrics(next_prefix, inner, out)


def find_result_json(root: Path) -> Path:
    candidates = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and ("results" in payload or "groups" in payload):
            return candidate
    raise BenchmarkError(f"could not find lm-eval JSON results under {root}")


def extract_lm_eval_metrics(payload: Dict[str, object], suite_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    groups = payload.get("groups")
    if isinstance(groups, dict) and isinstance(groups.get(suite_name), dict):
        flatten_numeric_metrics("", groups[suite_name], metrics)
        if metrics:
            return metrics

    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get(suite_name), dict):
        flatten_numeric_metrics("", results[suite_name], metrics)
        if metrics:
            return metrics

    if isinstance(results, dict):
        for task_name, task_metrics in results.items():
            if isinstance(task_metrics, dict) and (
                task_name == suite_name or str(task_name).startswith(f"{suite_name}_")
            ):
                flatten_numeric_metrics(str(task_name), task_metrics, metrics)

    if not metrics:
        raise BenchmarkError(f"no numeric metrics found for suite {suite_name}")
    return metrics


def pick_primary_metric(suite_name: str, metrics: Dict[str, float]) -> Tuple[str, str]:
    preferred_by_suite = {
        "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"],
        "mmlu": ["acc,none", "acc_norm,none"],
        "ifeval": [
            "prompt_level_strict_acc,none",
            "inst_level_strict_acc,none",
            "prompt_level_loose_acc,none",
        ],
        "humaneval": ["plus.pass@1", "base.pass@1"],
        "bfcl": ["score", "accuracy", "pass@1"],
        "aider": ["score", "pass_rate", "success_rate"],
    }
    for candidate in preferred_by_suite.get(suite_name, []):
        if candidate in metrics:
            return candidate, f"{metrics[candidate]:.6f}"

    for key in sorted(metrics):
        return key, f"{metrics[key]:.6f}"
    return "", ""


def run_logged_command(
    cmd: List[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        return subprocess.run(
            cmd if not shell else cmd[0],
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            timeout=timeout_s,
            env=env,
            shell=shell,
            check=False,
        )


def start_server(args: argparse.Namespace, model_path: str, server_log: Path) -> subprocess.Popen[str]:
    cmd = [
        args.llama_server_bin,
        "-m",
        model_path,
        "-ngl",
        str(args.ngl),
        "-c",
        str(args.ctx),
        "--reasoning",
        "off",
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]
    with server_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return proc


def cleanup_server(server_proc: Optional[subprocess.Popen[str]]) -> None:
    if server_proc is None or server_proc.poll() is not None:
        return
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=5)


def run_lm_eval_suite(
    args: argparse.Namespace,
    model_slug: str,
    tokenizer_id: str,
    suite_name: str,
    task_name: str,
    limit: int,
    fewshot: int,
    suite_dir: Path,
) -> Tuple[Dict[str, float], str, str]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = suite_dir / "stdout.log"
    stderr_path = suite_dir / "stderr.log"

    base_url = f"http://{args.server_host}:{args.server_port}/v1/completions"
    model_args = ",".join(
        [
            f"model={model_slug}",
            f"base_url={base_url}",
            f"num_concurrent={args.num_concurrent}",
            f"max_retries={args.max_retries}",
            "tokenized_requests=False",
            f"max_length={args.ctx}",
            f"max_gen_toks={args.max_gen_toks}",
            f"seed={args.seed}",
        ]
    )
    if tokenizer_id:
        model_args += f",tokenizer={tokenizer_id},tokenizer_backend=huggingface"

    cmd = [
        args.lm_eval_bin,
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        task_name,
        "--num_fewshot",
        str(fewshot),
        "--limit",
        str(limit),
        "--output_path",
        str(suite_dir),
    ]

    started = time.perf_counter()
    proc = run_logged_command(cmd, stdout_path, stderr_path, timeout_s=args.suite_timeout_s)
    runtime_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise BenchmarkError(f"lm-eval failed for {suite_name} (see {stderr_path})")

    result_json = find_result_json(suite_dir)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    metrics = extract_lm_eval_metrics(payload, suite_name)
    return metrics, f"{runtime_s:.3f}", ""


def create_humaneval_subset(limit: int, output_path: Path) -> None:
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


def find_latest_matching(root: Path, pattern: str) -> Path:
    candidates = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise BenchmarkError(f"could not find {pattern} under {root}")
    return candidates[0]


def parse_evalplus_stdout(text: str) -> Dict[str, float]:
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


def run_humaneval_suite(
    args: argparse.Namespace,
    model_slug: str,
    suite_dir: Path,
) -> Tuple[Dict[str, float], str, str]:
    ensure_commands_exist([args.evalplus_codegen_bin, args.evalplus_evaluate_bin])
    suite_dir.mkdir(parents=True, exist_ok=True)
    subset_path = suite_dir / "humaneval_subset.jsonl.gz"
    create_humaneval_subset(args.humaneval_limit, subset_path)

    env = os.environ.copy()
    env["HUMANEVAL_OVERRIDE_PATH"] = str(subset_path)
    env.setdefault("OPENAI_API_KEY", "local-benchmark")
    base_url = f"http://{args.server_host}:{args.server_port}/v1"

    codegen_stdout = suite_dir / "codegen.stdout.log"
    codegen_stderr = suite_dir / "codegen.stderr.log"
    codegen_cmd = [
        args.evalplus_codegen_bin,
        "--model",
        model_slug,
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
        timeout_s=args.suite_timeout_s,
        env=env,
    )
    if codegen_proc.returncode != 0:
        raise BenchmarkError(f"EvalPlus code generation failed (see {codegen_stderr})")

    samples_path = find_latest_matching(suite_dir / "codegen", "*.jsonl")
    eval_stdout = suite_dir / "evaluate.stdout.log"
    eval_stderr = suite_dir / "evaluate.stderr.log"
    eval_cmd = [
        args.evalplus_evaluate_bin,
        "--dataset",
        "humaneval",
        "--samples",
        str(samples_path),
    ]

    eval_proc = run_logged_command(
        eval_cmd,
        eval_stdout,
        eval_stderr,
        timeout_s=args.suite_timeout_s,
        env=env,
    )
    runtime_s = time.perf_counter() - started
    if eval_proc.returncode != 0:
        raise BenchmarkError(f"EvalPlus evaluation failed (see {eval_stderr})")

    metrics = parse_evalplus_stdout(eval_stdout.read_text(encoding="utf-8"))
    return metrics, f"{runtime_s:.3f}", ""


def run_external_template_suite(
    suite_name: str,
    template: str,
    model_path: str,
    model_slug: str,
    args: argparse.Namespace,
    suite_dir: Path,
    limit: int,
) -> Tuple[Dict[str, float], str, str]:
    if not template:
        raise BenchmarkError(f"{suite_name} command template was not provided")

    suite_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = suite_dir / "stdout.log"
    stderr_path = suite_dir / "stderr.log"
    base_url = f"http://{args.server_host}:{args.server_port}/v1"
    command_text = template.format(
        model_name=model_slug,
        model_path=model_path,
        base_url=base_url,
        host=args.server_host,
        port=args.server_port,
        suite_dir=suite_dir,
        limit=limit,
        ctx=args.ctx,
        seed=args.seed,
    )

    started = time.perf_counter()
    proc = run_logged_command(
        [command_text],
        stdout_path,
        stderr_path,
        timeout_s=args.suite_timeout_s,
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
    return metrics, f"{runtime_s:.3f}", ""


def safe_median(values: List[float]) -> str:
    if not values:
        return ""
    return f"{statistics.median(values):.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick local quality benchmark against llama.cpp-served GGUF models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("models_file", help="Path to models list file (same format as models/models.txt).")
    parser.add_argument(
        "--tokenizer-map-file",
        default="",
        help="Optional MODEL_PATH=TOKENIZER_ID map for lm-eval tasks; recommended for MMLU/loglikelihood runs.",
    )
    parser.add_argument("--out-dir-base", default=DEFAULTS["out_dir_base"])
    parser.add_argument("--ngl", type=int, default=DEFAULTS["ngl"])
    parser.add_argument("--ctx", type=int, default=DEFAULTS["ctx"])
    parser.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    parser.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULTS["max_gen_toks"])
    parser.add_argument("--num-concurrent", type=int, default=DEFAULTS["num_concurrent"])
    parser.add_argument("--max-retries", type=int, default=DEFAULTS["max_retries"])
    parser.add_argument("--suite-timeout-s", type=int, default=DEFAULTS["suite_timeout_s"])
    parser.add_argument("--server-host", default=DEFAULTS["server_host"])
    parser.add_argument("--server-port", type=int, default=DEFAULTS["server_port"])
    parser.add_argument("--llama-server-bin", default=DEFAULTS["llama_server_bin"])
    parser.add_argument("--lm-eval-bin", default=DEFAULTS["lm_eval_bin"])
    parser.add_argument("--evalplus-codegen-bin", default=DEFAULTS["evalplus_codegen_bin"])
    parser.add_argument("--evalplus-evaluate-bin", default=DEFAULTS["evalplus_evaluate_bin"])
    parser.add_argument("--gsm8k-limit", type=int, default=DEFAULTS["gsm8k_limit"])
    parser.add_argument("--gsm8k-fewshot", type=int, default=DEFAULTS["gsm8k_fewshot"])
    parser.add_argument("--mmlu-limit", type=int, default=DEFAULTS["mmlu_limit"])
    parser.add_argument("--mmlu-fewshot", type=int, default=DEFAULTS["mmlu_fewshot"])
    parser.add_argument("--ifeval-limit", type=int, default=DEFAULTS["ifeval_limit"])
    parser.add_argument("--ifeval-fewshot", type=int, default=DEFAULTS["ifeval_fewshot"])
    parser.add_argument("--humaneval-limit", type=int, default=DEFAULTS["humaneval_limit"])
    parser.add_argument(
        "--run-humaneval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_humaneval"],
        help="Run EvalPlus HumanEval+ subset if EvalPlus is installed.",
    )
    parser.add_argument(
        "--bfcl-command-template",
        default="",
        help=(
            "Optional shell command template for BFCL. Available placeholders: "
            "{model_name}, {model_path}, {base_url}, {host}, {port}, {suite_dir}, {limit}, {ctx}, {seed}"
        ),
    )
    parser.add_argument("--bfcl-limit", type=int, default=20)
    parser.add_argument(
        "--aider-command-template",
        default="",
        help=(
            "Optional shell command template for Aider benchmark subset. Available placeholders: "
            "{model_name}, {model_path}, {base_url}, {host}, {port}, {suite_dir}, {limit}, {ctx}, {seed}"
        ),
    )
    parser.add_argument("--aider-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_file = Path(args.models_file)
    if not models_file.is_file():
        raise BenchmarkError(f"models file not found: {models_file}")

    ensure_commands_exist([args.llama_server_bin, args.lm_eval_bin])
    models = read_models_file(models_file)
    if not models:
        raise BenchmarkError(f"no model entries found in {models_file}")
    tokenizer_map: Dict[str, str] = {}
    if args.tokenizer_map_file:
        tokenizer_path = Path(args.tokenizer_map_file)
        if not tokenizer_path.is_file():
            raise BenchmarkError(f"tokenizer map file not found: {tokenizer_path}")
        tokenizer_map = read_tokenizer_map_file(tokenizer_path)

    ts_slug = timestamp_slug()
    run_dir = Path(args.out_dir_base) / ts_slug
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = run_dir / "summary.csv"
    suite_runs_csv = run_dir / "suite_runs.csv"
    metrics_csv = run_dir / "metrics.csv"
    meta_file = run_dir / "run-meta.txt"
    write_csv(summary_csv, SUMMARY_FIELDS)
    write_csv(suite_runs_csv, SUITE_RUN_FIELDS)
    write_csv(metrics_csv, METRIC_FIELDS)
    write_meta(meta_file, args, models)

    print(f"Run directory: {run_dir}")
    print()

    server_proc: Optional[subprocess.Popen[str]] = None

    def signal_cleanup(*_: object) -> None:
        cleanup_server(server_proc)

    signal.signal(signal.SIGINT, signal_cleanup)
    signal.signal(signal.SIGTERM, signal_cleanup)

    for index, model_path in enumerate(models, start=1):
        model_slug = slugify_model(model_path)
        model_raw_dir = raw_dir / model_slug
        model_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}] Model: {model_path}")

        if not Path(model_path).is_file():
            err = "model file not found"
            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                SummaryRow(
                    timestamp=ts_slug,
                    model_path=model_path,
                    model_name=model_slug,
                    ctx=args.ctx,
                    ngl=args.ngl,
                    temperature=args.temp,
                    top_p=args.top_p,
                    seed=args.seed,
                    gsm8k_primary_metric="",
                    mmlu_primary_metric="",
                    ifeval_primary_metric="",
                    humaneval_plus_pass_at_1="",
                    bfcl_primary_metric="",
                    aider_primary_metric="",
                    status="failed",
                    error_note=err,
                ).to_dict(),
            )
            print(f"  - ERROR: {err}")
            print()
            continue

        summary_values: Dict[str, str] = {
            "gsm8k_primary_metric": "",
            "mmlu_primary_metric": "",
            "ifeval_primary_metric": "",
            "humaneval_plus_pass_at_1": "",
            "bfcl_primary_metric": "",
            "aider_primary_metric": "",
        }
        model_status = "ok"
        model_errors: List[str] = []

        try:
            server_log = model_raw_dir / "server.log"
            server_proc = start_server(args, model_path, server_log)
            if not wait_for_health(args.server_host, args.server_port, timeout_s=180):
                raise BenchmarkError("llama-server did not become ready")

            for suite_name, task_name, default_limit, default_fewshot in LM_EVAL_SUITES:
                limit = getattr(args, f"{suite_name}_limit", default_limit)
                fewshot = getattr(args, f"{suite_name}_fewshot", default_fewshot)
                suite_dir = model_raw_dir / suite_name
                try:
                    metrics, runtime_s, _ = run_lm_eval_suite(
                        args=args,
                        model_slug=model_slug,
                        tokenizer_id=tokenizer_map.get(model_path, ""),
                        suite_name=suite_name,
                        task_name=task_name,
                        limit=limit,
                        fewshot=fewshot,
                        suite_dir=suite_dir,
                    )
                    primary_name, primary_value = pick_primary_metric(suite_name, metrics)
                    summary_values[f"{suite_name}_primary_metric"] = primary_value
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite=suite_name,
                            limit=limit,
                            status="ok",
                            runtime_s=runtime_s,
                            primary_metric_name=primary_name,
                            primary_metric_value=primary_value,
                            error_note="",
                        ).to_dict(),
                    )
                    for metric_name, metric_value in sorted(metrics.items()):
                        metric_stderr = ""
                        if metric_name.endswith("_stderr,none"):
                            continue
                        stderr_key = f"{metric_name}_stderr,none"
                        if stderr_key in metrics:
                            metric_stderr = f"{metrics[stderr_key]:.6f}"
                        append_csv_row(
                            metrics_csv,
                            METRIC_FIELDS,
                            MetricRow(
                                timestamp=ts_slug,
                                model_path=model_path,
                                model_name=model_slug,
                                suite=suite_name,
                                task_name=task_name,
                                metric_name=metric_name,
                                metric_value=f"{metric_value:.6f}",
                                metric_stderr=metric_stderr,
                                limit=limit,
                                status="ok",
                                error_note="",
                            ).to_dict(),
                        )
                    print(f"  - {suite_name}: {primary_name}={primary_value}")
                except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
                    model_status = "partial"
                    model_errors.append(f"{suite_name}: {exc}")
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite=suite_name,
                            limit=limit,
                            status="failed",
                            runtime_s="",
                            primary_metric_name="",
                            primary_metric_value="",
                            error_note=str(exc),
                        ).to_dict(),
                    )
                    print(f"  - {suite_name}: ERROR: {exc}")

            if args.run_humaneval:
                try:
                    suite_dir = model_raw_dir / "humaneval"
                    metrics, runtime_s, _ = run_humaneval_suite(args, model_slug, suite_dir)
                    primary_name, primary_value = pick_primary_metric("humaneval", metrics)
                    summary_values["humaneval_plus_pass_at_1"] = primary_value
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="humaneval",
                            limit=args.humaneval_limit,
                            status="ok",
                            runtime_s=runtime_s,
                            primary_metric_name=primary_name,
                            primary_metric_value=primary_value,
                            error_note="",
                        ).to_dict(),
                    )
                    for metric_name, metric_value in sorted(metrics.items()):
                        append_csv_row(
                            metrics_csv,
                            METRIC_FIELDS,
                            MetricRow(
                                timestamp=ts_slug,
                                model_path=model_path,
                                model_name=model_slug,
                                suite="humaneval",
                                task_name="humaneval",
                                metric_name=metric_name,
                                metric_value=f"{metric_value:.6f}",
                                metric_stderr="",
                                limit=args.humaneval_limit,
                                status="ok",
                                error_note="",
                            ).to_dict(),
                        )
                    print(f"  - humaneval: {primary_name}={primary_value}")
                except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
                    model_status = "partial"
                    model_errors.append(f"humaneval: {exc}")
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="humaneval",
                            limit=args.humaneval_limit,
                            status="failed",
                            runtime_s="",
                            primary_metric_name="",
                            primary_metric_value="",
                            error_note=str(exc),
                        ).to_dict(),
                    )
                    print(f"  - humaneval: ERROR: {exc}")

            if args.bfcl_command_template:
                try:
                    suite_dir = model_raw_dir / "bfcl"
                    metrics, runtime_s, _ = run_external_template_suite(
                        suite_name="bfcl",
                        template=args.bfcl_command_template,
                        model_path=model_path,
                        model_slug=model_slug,
                        args=args,
                        suite_dir=suite_dir,
                        limit=args.bfcl_limit,
                    )
                    primary_name, primary_value = pick_primary_metric("bfcl", metrics)
                    summary_values["bfcl_primary_metric"] = primary_value
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="bfcl",
                            limit=args.bfcl_limit,
                            status="ok",
                            runtime_s=runtime_s,
                            primary_metric_name=primary_name,
                            primary_metric_value=primary_value,
                            error_note="",
                        ).to_dict(),
                    )
                    for metric_name, metric_value in sorted(metrics.items()):
                        append_csv_row(
                            metrics_csv,
                            METRIC_FIELDS,
                            MetricRow(
                                timestamp=ts_slug,
                                model_path=model_path,
                                model_name=model_slug,
                                suite="bfcl",
                                task_name="bfcl",
                                metric_name=metric_name,
                                metric_value=f"{metric_value:.6f}",
                                metric_stderr="",
                                limit=args.bfcl_limit,
                                status="ok",
                                error_note="",
                            ).to_dict(),
                        )
                    print(f"  - bfcl: {primary_name}={primary_value}")
                except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
                    model_status = "partial"
                    model_errors.append(f"bfcl: {exc}")
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="bfcl",
                            limit=args.bfcl_limit,
                            status="failed",
                            runtime_s="",
                            primary_metric_name="",
                            primary_metric_value="",
                            error_note=str(exc),
                        ).to_dict(),
                    )
                    print(f"  - bfcl: ERROR: {exc}")

            if args.aider_command_template:
                try:
                    suite_dir = model_raw_dir / "aider"
                    metrics, runtime_s, _ = run_external_template_suite(
                        suite_name="aider",
                        template=args.aider_command_template,
                        model_path=model_path,
                        model_slug=model_slug,
                        args=args,
                        suite_dir=suite_dir,
                        limit=args.aider_limit,
                    )
                    primary_name, primary_value = pick_primary_metric("aider", metrics)
                    summary_values["aider_primary_metric"] = primary_value
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="aider",
                            limit=args.aider_limit,
                            status="ok",
                            runtime_s=runtime_s,
                            primary_metric_name=primary_name,
                            primary_metric_value=primary_value,
                            error_note="",
                        ).to_dict(),
                    )
                    for metric_name, metric_value in sorted(metrics.items()):
                        append_csv_row(
                            metrics_csv,
                            METRIC_FIELDS,
                            MetricRow(
                                timestamp=ts_slug,
                                model_path=model_path,
                                model_name=model_slug,
                                suite="aider",
                                task_name="aider",
                                metric_name=metric_name,
                                metric_value=f"{metric_value:.6f}",
                                metric_stderr="",
                                limit=args.aider_limit,
                                status="ok",
                                error_note="",
                            ).to_dict(),
                        )
                    print(f"  - aider: {primary_name}={primary_value}")
                except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
                    model_status = "partial"
                    model_errors.append(f"aider: {exc}")
                    append_csv_row(
                        suite_runs_csv,
                        SUITE_RUN_FIELDS,
                        SuiteRunRow(
                            timestamp=ts_slug,
                            model_path=model_path,
                            model_name=model_slug,
                            suite="aider",
                            limit=args.aider_limit,
                            status="failed",
                            runtime_s="",
                            primary_metric_name="",
                            primary_metric_value="",
                            error_note=str(exc),
                        ).to_dict(),
                    )
                    print(f"  - aider: ERROR: {exc}")

        except (BenchmarkError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
            model_status = "failed"
            model_errors.append(str(exc))
            print(f"  - ERROR: {exc}")
        finally:
            cleanup_server(server_proc)
            server_proc = None
            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                SummaryRow(
                    timestamp=ts_slug,
                    model_path=model_path,
                    model_name=model_slug,
                    ctx=args.ctx,
                    ngl=args.ngl,
                    temperature=args.temp,
                    top_p=args.top_p,
                    seed=args.seed,
                    gsm8k_primary_metric=summary_values["gsm8k_primary_metric"],
                    mmlu_primary_metric=summary_values["mmlu_primary_metric"],
                    ifeval_primary_metric=summary_values["ifeval_primary_metric"],
                    humaneval_plus_pass_at_1=summary_values["humaneval_plus_pass_at_1"],
                    bfcl_primary_metric=summary_values["bfcl_primary_metric"],
                    aider_primary_metric=summary_values["aider_primary_metric"],
                    status=model_status,
                    error_note=" | ".join(model_errors),
                ).to_dict(),
            )
            print()

    print("Done.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Suite runs CSV: {suite_runs_csv}")
    print(f"Metrics CSV: {metrics_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
