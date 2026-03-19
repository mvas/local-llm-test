#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import urllib.error
from typing import Callable, Dict, List, Optional, Tuple

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
    start_server,
    cleanup_server,
    Metric,
    ModelContext,
)
from suite_evalplus import run_humaneval_suite
from suite_lm_eval import run_lm_eval_suite
from suite_templated import run_external_template_suite


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
    "run_humaneval": False,
    "run_bfcl": False,
    "run_aider": False,
}

# LM-Eval suites with default limit and fewshot
LM_EVAL_SUITES = (
    ("gsm8k", DEFAULTS["gsm8k_limit"], DEFAULTS["gsm8k_fewshot"]),
    ("mmlu", DEFAULTS["mmlu_limit"], DEFAULTS["mmlu_fewshot"]),
    ("ifeval", DEFAULTS["ifeval_limit"], DEFAULTS["ifeval_fewshot"]),
)

@dataclass(frozen=True)
class Summary:
    timestamp: str
    model_path: str
    model_name: str
    ctx: int
    ngl: int
    temperature: float
    top_p: float
    seed: int
    gsm8k_primary_metric: str = ""
    mmlu_primary_metric: str = ""
    ifeval_primary_metric: str = ""
    humaneval_plus_pass_at_1: str = ""
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
    runtime_s: str
    primary_metric_name: str
    primary_metric_value: str
    error_note: str

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


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--run-humaneval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_humaneval"],
        help="Run EvalPlus HumanEval+ subset if EvalPlus is installed.",
    )
    parser.add_argument("--humaneval-limit", type=int, default=DEFAULTS["humaneval_limit"])
    parser.add_argument(
        "--run-bfcl",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_bfcl"],
        help="Run BFCL benchmark subset if BFCL is installed.",
    )
    parser.add_argument("--bfcl-limit", type=int, default=20)
    parser.add_argument(
        "--run-aider",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["run_aider"],
        help="Run Aider benchmark subset if Aider is installed.",
    )
    parser.add_argument("--aider-limit", type=int, default=20)
    # parser.add_argument(
    #     "--bfcl-command-template",
    #     default="",
    #     help=(
    #         "Optional shell command template for BFCL. Available placeholders: "
    #         "{model_name}, {model_path}, {base_url}, {host}, {port}, {suite_dir}, {limit}, {ctx}, {seed}"
    #     ),
    # )
    # parser.add_argument(
    #     "--aider-command-template",
    #     default="",
    #     help=(
    #         "Optional shell command template for Aider benchmark subset. Available placeholders: "
    #         "{model_name}, {model_path}, {base_url}, {host}, {port}, {suite_dir}, {limit}, {ctx}, {seed}"
    #     ),
    # )
    return parser.parse_args()


def _benchmark_model(ctx: ModelContext,
    summary_csv: Path, suite_runs_csv: Path, metrics_csv: Path, tokenizer_map: Dict[str, str]) -> None:

    summary_values: Dict[str, str] = {
        "gsm8k_primary_metric": "",
        "mmlu_primary_metric": "",
        "ifeval_primary_metric": "",
        "humaneval_plus_pass_at_1": "",
        "bfcl_primary_metric": "",
        "aider_primary_metric": "",
    }

    def getSuiteRunRow(suite: str, limit: int, status: str, runtime_s: str = "", primary_metric_name: str = "", primary_metric_value: str = "", error_note: str = "") -> SuiteRun:
        return SuiteRun(
                        timestamp=ctx.ts_slug,
                        model_path=ctx.model_path,
                        model_name=ctx.model_slug,
                        suite=suite,
                        limit=limit,
                        status=status,
                        runtime_s=runtime_s,
                        primary_metric_name=primary_metric_name,
                        primary_metric_value=primary_metric_value,
                        error_note=error_note,
                    )

    def run_benchmark_suite(suite_name: str, limit: int, runner: Callable[[], Tuple[str, str, List[Metric], str]]) -> Tuple[bool, str]:
        try:
            primary_name, primary_value, metrics, runtime_s = runner()
            append_csv_row(
                    suite_runs_csv,
                    SUITE_RUN_FIELDS,
                    getSuiteRunRow(suite=suite_name, limit=limit, status="ok", runtime_s=runtime_s, primary_metric_name=primary_name, primary_metric_value=primary_value).to_dict(),
                )
            for row in metrics:
                append_csv_row(metrics_csv, METRIC_FIELDS, row.to_dict())
            print(f"  - {suite_name}: {primary_name}={primary_value}")
            return True, primary_value
        except (BenchmarkError, subprocess.SubprocessError, TimeoutError) as exc:
            model_errors.append(f"{suite_name}: {exc}")
            append_csv_row(
                suite_runs_csv,
                SUITE_RUN_FIELDS,
                getSuiteRunRow(suite=suite_name, limit=limit, status="failed", error_note=str(exc)).to_dict(),
            )
            print(f"  - {suite_name}: ERROR: {exc}")
            return False, ""

    model_status = "ok"
    model_errors: List[str] = []
    try:
        server_log = ctx.model_raw_dir / "server.log"
        global server_proc
        server_proc = start_server(
            ctx.args.llama_server_bin, server_log, ctx.model_path, ctx.args.ngl, ctx.args.ctx,
            ctx.args.server_host, ctx.args.server_port, None, ctx.args.temp, ctx.args.top_p, ctx.args.seed)

        # Run LM-Eval suites
        for suite_name, default_limit, default_fewshot in LM_EVAL_SUITES:
            limit = getattr(ctx.args, f"{suite_name}_limit", default_limit)
            fewshot = getattr(ctx.args, f"{suite_name}_fewshot", default_fewshot)

            success, primary_value = run_benchmark_suite(suite_name, limit, lambda: run_lm_eval_suite(
                ctx=ctx,
                tokenizer_id=tokenizer_map.get(ctx.model_path, ""),
                suite_name=suite_name,
                limit=limit,
                fewshot=fewshot,
            ))
            summary_values[f"{suite_name}_primary_metric"] = primary_value
            if not success:
                model_status = "partial"

        # Run EvalPlus HumanEval+ subset
        if ctx.args.run_humaneval:
            success, primary_value = run_benchmark_suite(
                "humaneval", ctx.args.humaneval_limit, lambda: run_humaneval_suite(ctx=ctx))
            summary_values["humaneval_plus_pass_at_1"] = primary_value
            if not success:
                model_status = "partial"

        # Run BFCL benchmark subset
        if ctx.args.run_bfcl:
            success, primary_value = run_benchmark_suite("bfcl", ctx.args.bfcl_limit, lambda: run_external_template_suite(
                suite_name="bfcl",
                template=ctx.args.bfcl_command_template,
                ctx=ctx,
                limit=ctx.args.bfcl_limit,
            ))
            summary_values["bfcl_primary_metric"] = primary_value
            if not success:
                model_status = "partial"

        # Run Aider benchmark subset
        if ctx.args.run_aider:
            success, primary_value = run_benchmark_suite("aider", ctx.args.aider_limit, lambda: run_external_template_suite(
                suite_name="aider",
                template=ctx.args.aider_command_template,
                ctx=ctx,
                limit=ctx.args.aider_limit,
            ))
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
                ctx=ctx.args.ctx,
                ngl=ctx.args.ngl,
                temperature=ctx.args.temp,
                top_p=ctx.args.top_p,
                seed=ctx.args.seed,
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

server_proc: Optional[subprocess.Popen[str]] = None


def main() -> int:
    args = _parse_args()
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
        tokenizer_map = _read_tokenizer_map_file(tokenizer_path)

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

    print(f"Run directory: {run_dir}")
    print()

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
                Summary(
                    timestamp=ts_slug,
                    model_path=model_path,
                    model_name=model_slug,
                    ctx=args.ctx,
                    ngl=args.ngl,
                    temperature=args.temp,
                    top_p=args.top_p,
                    seed=args.seed,
                    status="failed",
                    error_note=err,
                ).to_dict(),
            )
            print(f"  - ERROR: {err}")
            print()
            continue
        context = ModelContext(
            args=args, ts_slug=ts_slug,
            model_path=model_path, model_slug=model_slug,
            model_raw_dir=model_raw_dir)
        _benchmark_model(context, summary_csv, suite_runs_csv, metrics_csv, tokenizer_map)

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
