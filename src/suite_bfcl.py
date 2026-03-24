import csv
import json
import os
from pathlib import Path
import re
import time
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


def _percent_or_float(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("empty numeric value")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _read_bfcl_model_id_map_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise BenchmarkError(
                f"invalid BFCL model id map entry at {path}:{line_no}; "
                "expected MODEL_PATH=BFCL_MODEL_ID"
            )
        model_text, bfcl_model_id = line.split("=", 1)
        model_path = expand_model_path(model_text)
        bfcl_model_id = bfcl_model_id.strip()
        if not bfcl_model_id:
            raise BenchmarkError(f"empty BFCL model id at {path}:{line_no}")
        mapping[model_path] = bfcl_model_id
    return mapping


def _resolve_bfcl_model_id(ctx: ModelContext) -> str:
    map_file = str(getattr(ctx.args, "bfcl_model_id_map_file", "")).strip()
    if not map_file:
        raise BenchmarkError(
            "--bfcl-model-id is required when --run-bfcl is enabled "
            "(or set --bfcl-model-id-map-file)"
        )
    map_path = Path(map_file).expanduser()
    if not map_path.is_file():
        raise BenchmarkError(f"BFCL model id map file not found: {map_path}")
    mapping = _read_bfcl_model_id_map_file(map_path)
    bfcl_model_id = mapping.get(ctx.model_path, "")
    if not bfcl_model_id:
        raise BenchmarkError(
            f"no BFCL model id mapping for model path: {ctx.model_path} "
            f"(in {map_path})"
        )
    return bfcl_model_id


def _write_subset_ids_file(
    bfcl_python: Path,
    output_path: Path,
    categories_csv: str,
    limit: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = r"""
import json
import sys
from pathlib import Path
from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

output_path = Path(sys.argv[1])
categories_csv = sys.argv[2]
limit = int(sys.argv[3])

categories = parse_test_category_argument([part.strip() for part in categories_csv.split(",") if part.strip()])
payload = {}
for category in categories:
    entries = load_dataset_entry(category)
    payload[category] = [entry["id"] for entry in entries[:limit]]

output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote subset IDs file: {output_path}")
"""
    proc = run_logged_command(
        [str(bfcl_python), "-c", script, str(output_path), categories_csv, str(limit)],
        stdout_path=output_path.with_suffix(".stdout.log"),
        stderr_path=output_path.with_suffix(".stderr.log"),
        timeout_s=SUITE_TIMEOUT,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise BenchmarkError(
            "failed to generate BFCL subset IDs file "
            f"(see {output_path.with_suffix('.stderr.log')})"
        )


def _parse_bfcl_metrics(score_dir: Path, bfcl_model_id: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    overall_csv = score_dir / "data_overall.csv"
    if overall_csv.is_file():
        with overall_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader, None)
            if first_row:
                field_map = {
                    "Overall Acc": "overall_acc",
                    "Non-Live AST Acc": "non_live_ast_acc",
                    "Live Acc": "live_acc",
                    "Multi Turn Acc": "multi_turn_acc",
                    "Relevance Detection": "relevance_detection",
                    "Irrelevance Detection": "irrelevance_detection",
                }
                for source_name, metric_name in field_map.items():
                    raw = first_row.get(source_name, "")
                    if not raw:
                        continue
                    try:
                        metrics[metric_name] = _percent_or_float(raw)
                    except ValueError:
                        continue

    model_score_dir = score_dir / bfcl_model_id.replace("/", "_")
    if model_score_dir.is_dir():
        pattern = re.compile(r"BFCL_v\d+_(.+)_score\.json$")
        for score_json in model_score_dir.rglob("*_score.json"):
            match = pattern.search(score_json.name)
            if not match:
                continue
            category = match.group(1)
            try:
                payload = json.loads(score_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                accuracy = payload[0].get("accuracy")
                if isinstance(accuracy, (int, float)):
                    metrics[f"{category}.accuracy"] = float(accuracy)

    if not metrics:
        raise BenchmarkError(f"could not parse BFCL metrics from {score_dir}")
    return metrics


def run_bfcl(ctx: ModelContext, full_mode: bool) -> Tuple[str, str, List[Metric], str]:
    bfcl_env_path = Path(".venv-bfcl")
    bfcl_bin = bfcl_env_path / "bin" / "bfcl"
    bfcl_python = bfcl_env_path / "bin" / "python"
    bfcl_model_id = _resolve_bfcl_model_id(ctx)
    ensure_commands_exist([str(bfcl_bin), str(bfcl_python)])

    suite_dir = ctx.model_raw_dir / "bfcl"
    suite_dir.mkdir(parents=True, exist_ok=True)
    result_dir_name = "result"
    score_dir_name = "score"
    result_dir = suite_dir / result_dir_name
    score_dir = suite_dir / score_dir_name

    categories_csv = "python"

    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(suite_dir)
    env["LOCAL_SERVER_ENDPOINT"] = ctx.args.server_host
    env["LOCAL_SERVER_PORT"] = str(ctx.args.server_port)
    env.setdefault(
        "REMOTE_OPENAI_BASE_URL",
        f"http://{ctx.args.server_host}:{ctx.args.server_port}/v1",
    )
    env.setdefault("REMOTE_OPENAI_API_KEY", "local-benchmark")
    # if ctx.args.bfcl_remote_tokenizer_path:
    #     env["REMOTE_OPENAI_TOKENIZER_PATH"] = ctx.args.bfcl_remote_tokenizer_path

    run_ids = False
    if ctx.args.bfcl_limit > 0:
        run_ids = True
        subset_ids_path = suite_dir / "test_case_ids_to_generate.json"
        _write_subset_ids_file(
            bfcl_python=bfcl_python,
            output_path=subset_ids_path,
            categories_csv=categories_csv,
            limit=ctx.args.bfcl_limit,
        )

    generate_stdout = suite_dir / "generate.stdout.log"
    generate_stderr = suite_dir / "generate.stderr.log"
    generate_cmd = [
        str(bfcl_bin),
        "generate",
        "--model",
        bfcl_model_id,
        "--skip-server-setup",
        "--temperature",
        "0.001",
        "--num-threads",
        "1",
        "--result-dir",
        result_dir_name,
    ]
    if not full_mode:
        # Available categories
        # https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md
        generate_cmd.append("--test-category")
        generate_cmd.append(categories_csv)
    if run_ids:
        generate_cmd.append("--run-ids")

    started = time.perf_counter()
    generate_proc = run_logged_command(
        generate_cmd,
        stdout_path=generate_stdout,
        stderr_path=generate_stderr,
        timeout_s=SUITE_TIMEOUT,
        env=env,
    )
    if generate_proc.returncode != 0:
        raise BenchmarkError(f"BFCL generate failed (see {generate_stderr})")

    evaluate_stdout = suite_dir / "evaluate.stdout.log"
    evaluate_stderr = suite_dir / "evaluate.stderr.log"
    evaluate_cmd = [
        str(bfcl_bin),
        "evaluate",
        "--model",
        bfcl_model_id,
        "--result-dir",
        result_dir_name,
        "--score-dir",
        score_dir_name,
    ]
    if not full_mode:
        # Available categories
        # https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md
        evaluate_cmd.append("--test-category")
        evaluate_cmd.append(categories_csv)
    if run_ids:
        evaluate_cmd.append("--partial-eval")

    evaluate_proc = run_logged_command(
        evaluate_cmd,
        stdout_path=evaluate_stdout,
        stderr_path=evaluate_stderr,
        timeout_s=SUITE_TIMEOUT,
        env=env,
    )
    runtime_s = time.perf_counter() - started
    if evaluate_proc.returncode != 0:
        raise BenchmarkError(f"BFCL evaluate failed (see {evaluate_stderr})")

    metrics = _parse_bfcl_metrics(score_dir=score_dir, bfcl_model_id=bfcl_model_id)
    primary_name = "overall_acc" if "overall_acc" in metrics else sorted(metrics)[0]
    primary_value = f"{metrics[primary_name]:.6f}"
    metric_rows = [
        Metric(
            timestamp=ctx.ts_slug,
            model_path=ctx.model_path,
            model_name=ctx.model_slug,
            suite="bfcl",
            metric_name=metric_name,
            metric_value=f"{metric_value:.6f}",
            metric_stderr="",
            limit=ctx.args.bfcl_limit,
            status="ok",
            error_note="",
        )
        for metric_name, metric_value in sorted(metrics.items())
    ]
    return primary_name, primary_value, metric_rows, f"{runtime_s:.3f}"

