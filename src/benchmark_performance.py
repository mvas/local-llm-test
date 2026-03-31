#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import signal
import subprocess
import sys
import urllib.error
from typing import Callable, Dict, List, Optional, Tuple, cast

from common import (
    LLAMA_SERVER_BIN,
    LM_EVAL_BIN,
    now_iso,
    timestamp_slug,
    slugify_model,
    expand_model_path,
    read_models_file,
    BenchmarkError,
    ensure_commands_exist,
    write_csv,
    append_csv_row,
    start_llama_server,
    cleanup_server,
    Metric,
    ModelContext,
)
from suite_evalplus import run_humaneval, run_mbpp
from suite_bfcl import run_bfcl
from suite_aider import run_aider
from suite_lm_eval import run_lm_eval_suite


DEFAULTS = {
    "ngl": 99,
    "ctx": 8192,
    "temp": 0.0,
    "top_p": 1.0,
    "seed": 1234,
    "server_host": "127.0.0.1",
    "server_port": 8082,
    "out_dir_base": "results/performance",
    # Quick-suite defaults sized to keep the total runtime reasonable on local hardware.
    "run_humaneval": False,
    "run_mbpp": False,
    "run_bfcl": False,
    "run_aider": False,
    "run_lm_evals": False,
    "full_mode": False,
    "bfcl_timeout": 5400,
}

config_fast = {
    "gsm8k": {
        "task": "gsm8k_cot_llama",  # designed for modern instruct/chat formatting
        "use_chat": True, # Uses chat API, better suited for instruct models
        "limit": 30, #30,
        "shots": 4,
        # "context": 8192,
        "model_args": [
            f"max_gen_toks=1024",
            f"seed=1234",
        ],
        "cmd_args": [
            "--apply_chat_template",
            "--fewshot_as_multiturn",
            "--gen_kwargs",
            "temperature=0.0,top_p=1.0",
        ]
    },
    "mmlu": {
        "task": "mmlu",
        "use_chat": False,
        "use_custom_lm_eval_backend": "llamacpp-native-mc",
        "limit": 20, # 100,
        "shots": 5,
        # "context": 8192, # can be 4096
        "model_args": [
            "n_probs=100",
        ]
    },
    "ifeval": {
        "task": "ifeval",
        "use_chat": True,
        "limit": 30, # 50,
        "shots": 0,
        "model_args": [
            f"max_length=8192",
            f"max_gen_toks=2048",
            f"seed=1234",
        ],
        "cmd_args": [
            "--apply_chat_template",
            "--gen_kwargs",
            "temperature=0.0,top_p=1.0",
        ]
    }
}

config_full = {
    "gsm8k": {
        "task": "gsm8k_cot_llama",  
        "use_chat": True, # Uses chat API, better suited for instruct models
        "limit": 10, # TODO:TEMPORARY!!,
        "shots": 8,
        # "context": 8192,
        "model_args": [
            f"max_gen_toks=1024",
            f"seed=1234",
        ],
        "cmd_args": [
            "--apply_chat_template",
            "--fewshot_as_multiturn",
            "--gen_kwargs",
            "temperature=0.0,top_p=1.0",
        ]
    },
    "mmlu": {
        "task": "mmlu_pro",
        "use_chat": False,
        "use_custom_lm_eval_backend": "llamacpp-native-mc",
        "limit": 10, # TODO:TEMPORARY!!,
        "shots": 5,
        "model_args": [
            f"n_probs=100",
        ],
    },
    "ifeval": {
        "task": "ifeval",
        "use_chat": True,
        "limit": 10, # TODO:TEMPORARY!!,
        "shots": 0,
        "model_args": [
            f"max_length=8192",
            f"max_gen_toks=2048",
            f"seed=1234",
        ],
        "cmd_args": [
            "--apply_chat_template",
            "--gen_kwargs",
            "temperature=0.0,top_p=1.0",
        ]
    }
}

@dataclass(frozen=True)
class Summary:
    timestamp: str
    model_path: str
    model_name: str
    ctx: int
    ngl: int
    temp: float
    top_p: float
    seed: int
    gsm8k_primary_metric: str = ""
    mmlu_primary_metric: str = ""
    ifeval_primary_metric: str = ""
    humaneval_plus_pass_at_1: str = ""
    mbpp_plus_pass_at_1: str = ""
    bfcl_primary_metric: str = ""
    aider_primary_metric: str = ""
    status: str = ""
    error_note: str = ""

    @classmethod
    def headers(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteRun:
    timestamp: str
    model_path: str
    model_name: str
    suite: str
    limit: int
    status: str
    runtime_s: str = ""
    primary_metric_name: str = ""
    primary_metric_value: str = ""
    error_note: str = ""

    @classmethod
    def headers(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


METRIC_FIELDS = Metric.headers()
SUMMARY_FIELDS = Summary.headers()
SUITE_RUN_FIELDS = SuiteRun.headers()


def _read_tokenizer_map_file(path: Path) -> Dict[str, str]:
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


def _write_meta(path: Path, args: argparse.Namespace, resolved_models: List[str]) -> None:
    lines = [
        f"run_timestamp_utc={now_iso()}",
        f"models_file={args.models_file}",
        f"tokenizer_map_file={args.tokenizer_map_file or ''}",
        f"models_count={len(resolved_models)}",
        f"ctx={args.ctx}",
        f"ngl={args.ngl}",
        f"default_temp={args.default_temp}",
        f"default_top_p={args.default_top_p}",
        f"default_seed={args.default_seed}",
        f"server_host={args.server_host}",
        f"server_port={args.server_port}",
        f"humaneval_limit={args.humaneval_limit}",
        f"mbpp_limit={args.mbpp_limit}",
        f"run_humaneval={args.run_humaneval}",
        f"run_mbpp={args.run_mbpp}",
        f"run_bfcl={args.run_bfcl}",
        f"bfcl_limit={args.bfcl_limit}",
        f"bfcl_timeout={args.bfcl_timeout}",
        f"bfcl_model_id_map_file={args.bfcl_model_id_map_file}",
        f"run_lm_evals={args.run_lm_evals}",
        f"full_mode={args.full_mode}",
        f"run_aider={args.run_aider}",
        f"aider_limit={args.aider_limit}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick local quality benchmark against llama.cpp-served GGUF models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("models_file", help="Path to models list file (same format as models/models.txt).")
    parser.add_argument(
        "-t",
        "--tokenizer-map-file",
        dest="tokenizer_map_file",
        default="",
        metavar="PATH",
        help="Optional MODEL_PATH=TOKENIZER_ID map for lm-eval tasks; recommended for MMLU/loglikelihood runs.",
    )
    parser.add_argument("--out-dir-base", default=DEFAULTS["out_dir_base"])
    parser.add_argument("--ngl", type=int, default=DEFAULTS["ngl"])
    parser.add_argument("--ctx", type=int, default=DEFAULTS["ctx"])
    parser.add_argument("--default-temp", type=float, default=DEFAULTS["temp"])
    parser.add_argument("--default-top-p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--default-seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--server-host", default=DEFAULTS["server_host"])
    parser.add_argument("--server-port", type=int, default=DEFAULTS["server_port"])
    parser.add_argument("--full-mode", action=argparse.BooleanOptionalAction, default=DEFAULTS["full_mode"])
    parser.add_argument(
        "--run-lm-evals",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_lm_evals"],
        help="Run LM-Eval suites if LM-Eval is installed.",
    )
    parser.add_argument(
        "--run-humaneval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_humaneval"],
        help="Run EvalPlus HumanEval+ subset if EvalPlus is installed.",
    )
    parser.add_argument("--humaneval-limit", type=int, default=40)
    parser.add_argument(
        "--run-mbpp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_mbpp"],
        help="Run EvalPlus MBPP+ subset if EvalPlus is installed.",
    )
    parser.add_argument("--mbpp-limit", type=int, default=20)
    parser.add_argument(
        "--run-bfcl",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_bfcl"],
        help="Run BFCL benchmark using the dedicated BFCL environment.",
    )
    parser.add_argument("--bfcl-limit", type=int, default=20)
    parser.add_argument(
        "--bfcl-timeout",
        type=int,
        default=DEFAULTS["bfcl_timeout"],
        metavar="SECONDS",
        help="Timeout in seconds for the BFCL generate step. Reasoning models may need more time.",
    )
    parser.add_argument(
        "--bfcl-model-id-map-file",
        default="models/bfcl-model-ids.txt",
        metavar="PATH",
        help="MODEL_PATH=BFCL_MODEL_ID map used when --bfcl-model-id is not set.",
    )
    parser.add_argument(
        "--run-aider",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_aider"],
        help="Run Aider benchmark via aider-benchmark Docker image.",
    )
    parser.add_argument("--aider-limit", type=int, default=20)
    return parser.parse_args()


def _get_lm_eval_cmd(suite_config: dict, ctx: ModelContext, host:str, port: int, tokenizer_id: str) -> List[str]:
    custom_backend = str(suite_config.get("use_custom_lm_eval_backend", ""))
    use_chat = bool(suite_config.get("use_chat", False))

    base_url = f"http://{host}:{port}" 
    if not custom_backend:
        base_url += "/v1/chat/completions" if use_chat else "/v1/completions"

    extra_model_args = cast(List[str], suite_config.get("model_args") or [])
    model_args = ",".join([
        f"model={ctx.model_slug}",
        f"base_url={base_url}",
        "num_concurrent=1",
        "max_retries=3",
        "tokenized_requests=False",
        *extra_model_args,
    ])
    if tokenizer_id:
        model_args += f",tokenizer={tokenizer_id},tokenizer_backend=huggingface"

    if custom_backend:
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("lm_eval_with_local_models.py")),
            "run",
            "--model",
            custom_backend,
        ]
    else:
        cmd = [
            LM_EVAL_BIN,
            "run",
            "--model",
            "local-chat-completions" if use_chat else "local-completions",
        ]
    cmd.extend([
        "--model_args",
        model_args,
        "--tasks",
        str(suite_config["task"]),
        "--num_fewshot",
        str(suite_config.get("shots", 0)),
    ])
    cmd.extend(cast(List[str], suite_config.get("cmd_args", [])))
    limit = suite_config.get("limit", 0)
    if limit > 0:
        cmd.append("--limit")
        cmd.append(str(limit))
    return cmd


def _benchmark_model(
    ctx: ModelContext,
    args: argparse.Namespace,
    summary_csv: Path,
    suite_runs_csv: Path,
    metrics_csv: Path,
    tokenizer_map: Dict[str, str],
) -> None:

    def run_benchmark_suite(suite_name: str, limit: int, runner: Callable[[], Tuple[str, str, List[Metric], str]]) -> Tuple[bool, str]:
        print(f"  - Running {suite_name} (limit={limit})", end=". ", flush=True)
        try:
            primary_name, primary_value, metrics, runtime_s = runner()
            append_csv_row(
                    suite_runs_csv,
                    SUITE_RUN_FIELDS,
                    SuiteRun(
                        timestamp=ctx.ts_slug,
                        model_path=ctx.model_path,
                        model_name=ctx.model_slug,
                        suite=suite_name,
                        limit=limit,
                        status="ok",
                        runtime_s=runtime_s,
                        primary_metric_name=primary_name,
                        primary_metric_value=primary_value,
                    ).to_dict(),
                )
            for row in metrics:
                append_csv_row(metrics_csv, METRIC_FIELDS, row.to_dict())
            print(f"Results: {primary_name}={primary_value}")
            return True, primary_value
        except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
            model_errors.append(f"{suite_name}: {exc}")
            append_csv_row(
                suite_runs_csv,
                SUITE_RUN_FIELDS,
                SuiteRun(
                    timestamp=ctx.ts_slug,
                    model_path=ctx.model_path,
                    model_name=ctx.model_slug,
                    suite=suite_name,
                    limit=limit,
                    status="failed",
                    error_note=str(exc),
                ).to_dict(),
            )
            print(f"ERROR: {exc}")
            return False, ""

    summary_values: Dict[str, str] = {
        "gsm8k_primary_metric": "",
        "mmlu_primary_metric": "",
        "ifeval_primary_metric": "",
        "humaneval_plus_pass_at_1": "",
        "mbpp_plus_pass_at_1": "",
        "bfcl_primary_metric": "",
        "aider_primary_metric": "",
    }

    model_status = "ok"
    model_errors: List[str] = []
    tokenizer_id = tokenizer_map.get(ctx.model_path, "")
    try:
        server_log = ctx.model_raw_dir / "server.log"
        global server_proc
        server_proc = start_llama_server(
            server_log, ctx.model_path, args.ngl, args.ctx,
            args.server_host, args.server_port, None,
            args.default_temp, args.default_top_p, args.default_seed)

        # Run LM-Eval suites
        if args.run_lm_evals:
            suites = config_full if args.full_mode else config_fast
            for suite_name, suite_config in suites.items():
                cmd = _get_lm_eval_cmd(suite_config, ctx, args.server_host, args.server_port, tokenizer_id)
                limit = suite_config.get("limit", 0)

                success, primary_value = run_benchmark_suite(
                    suite_name, limit, lambda: run_lm_eval_suite(
                        ctx=ctx,
                        suite_name=suite_name,
                        limit=limit,
                        cmd=cmd,
                    ))
                summary_values[f"{suite_name}_primary_metric"] = primary_value
                if not success:
                    model_status = "partial"

        # Run EvalPlus HumanEval+ subset
        if args.run_humaneval:
            success, primary_value = run_benchmark_suite(
                "humaneval", args.humaneval_limit, lambda: run_humaneval(
                    ctx=ctx,
                    host=args.server_host,
                    port=args.server_port,
                    limit=args.humaneval_limit,))
            summary_values["humaneval_plus_pass_at_1"] = primary_value
            if not success:
                model_status = "partial"

        # Run EvalPlus MBPP+ subset
        if args.run_mbpp:
            success, primary_value = run_benchmark_suite(
                "mbpp", args.mbpp_limit, lambda: run_mbpp(
                    ctx=ctx,
                    host=args.server_host,
                    port=args.server_port,
                    limit=args.mbpp_limit,))
            summary_values["mbpp_plus_pass_at_1"] = primary_value
            if not success:
                model_status = "partial"

        # Run BFCL benchmark subset
        if args.run_bfcl:
            success, primary_value = run_benchmark_suite(
                "bfcl", args.bfcl_limit, lambda: run_bfcl(
                    ctx=ctx,
                    host=args.server_host,
                    port=args.server_port,
                    full_mode=args.full_mode,
                    limit=args.bfcl_limit,
                    map_file=args.bfcl_model_id_map_file,
                    timeout_s=args.bfcl_timeout,
                ),
            )
            summary_values["bfcl_primary_metric"] = primary_value
            if not success:
                model_status = "partial"

        # Run Aider benchmark subset
        if args.run_aider:
            success, primary_value = run_benchmark_suite(
                "aider", args.aider_limit, lambda: run_aider(
                    ctx=ctx,
                    port=args.server_port,
                    full_mode=args.full_mode,
                    limit=args.aider_limit,
                ),
            )
            summary_values["aider_primary_metric"] = primary_value
            if not success:
                model_status = "partial"

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
            Summary(
                timestamp=ctx.ts_slug,
                model_path=ctx.model_path,
                model_name=ctx.model_slug,
                ctx=args.ctx,
                ngl=args.ngl,
                temp=args.default_temp,
                top_p=args.default_top_p,
                seed=args.default_seed,
                gsm8k_primary_metric=summary_values["gsm8k_primary_metric"],
                mmlu_primary_metric=summary_values["mmlu_primary_metric"],
                ifeval_primary_metric=summary_values["ifeval_primary_metric"],
                humaneval_plus_pass_at_1=summary_values["humaneval_plus_pass_at_1"],
                mbpp_plus_pass_at_1=summary_values["mbpp_plus_pass_at_1"],
                bfcl_primary_metric=summary_values["bfcl_primary_metric"],
                aider_primary_metric=summary_values["aider_primary_metric"],
                status=model_status,
                error_note=" | ".join(model_errors),
            ).to_dict(),
        )
        print()

server_proc: Optional[subprocess.Popen[str]] = None


def main() -> int:
    ensure_commands_exist([LLAMA_SERVER_BIN, LM_EVAL_BIN])

    args = _parse_args()

    # Read models
    models_file = Path(args.models_file)
    if not models_file.is_file():
        raise BenchmarkError(f"models file not found: {models_file}")
    models = read_models_file(models_file)
    if not models:
        raise BenchmarkError(f"no model entries found in {models_file}")

    # Read tokenizers
    tokenizer_map: Dict[str, str] = {}
    if args.tokenizer_map_file:
        print(f"Reading tokenizers from: {args.tokenizer_map_file}")
        tokenizer_path = Path(args.tokenizer_map_file)
        if not tokenizer_path.is_file():
            raise BenchmarkError(f"tokenizer map file not found: {tokenizer_path}")
        tokenizer_map = _read_tokenizer_map_file(tokenizer_path)

    # Create run folders and files
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
    _write_meta(meta_file, args, models)

    print(f"Running in folder: {run_dir}")
    print()

    def signal_cleanup(*_: object) -> None:
        cleanup_server(server_proc)
    signal.signal(signal.SIGINT, signal_cleanup)
    signal.signal(signal.SIGTERM, signal_cleanup)

    for index, model_path in enumerate(models, start=1):
        model_slug = slugify_model(model_path)
        model_raw_dir = raw_dir / model_slug
        model_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}] Benchmarking model: {model_path}")

        if not Path(model_path).is_file():
            err = "model file not found"
            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                Summary(
                    timestamp=ts_slug,
                    model_path=model_path,
                    model_name=model_slug,
                    ctx=args.ctx,
                    ngl=args.ngl,
                    temp=args.default_temp,
                    top_p=args.default_top_p,
                    seed=args.default_seed,
                    status="failed",
                    error_note=err,
                ).to_dict(),
            )
            print(f"  - ERROR: {err}")
            print()
            continue
        ctx = ModelContext(
            ts_slug=ts_slug,
            model_path=model_path,
            model_slug=model_slug,
            model_raw_dir=model_raw_dir,
        )
        _benchmark_model(ctx, args, summary_csv, suite_runs_csv, metrics_csv, tokenizer_map)

    print("Benchmark complete.")
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
