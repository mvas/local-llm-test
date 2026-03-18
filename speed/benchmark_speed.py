#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULTS = {
    "ngl": 99,
    "ctx": 8192,
    "gen_tokens": 256,
    "temp": 0.0,
    "warmup": 1,
    "repeats": 3,
    "threads": None,
    "bench_n_prompt": 512, # Number of input tokens to llama-bench
    "bench_n_gen": 256, # Number of output tokens to llama-bench
    "bench_repetitions": 3, # Number of repetitions to run in llama-bench
    "server_host": "127.0.0.1",
    "server_port": 8081,
    "prompts_file": "speed/prompts.txt",
    "out_dir_base": "results/speed",
    "llama_bench_bin": os.environ.get("LLAMA_BENCH_BIN", "llama-bench"),
    "llama_server_bin": os.environ.get("LLAMA_SERVER_BIN", "llama-server"),
}

SUMMARY_FIELDS = [
    "timestamp",
    "model_path",
    "model_name",
    "ctx",
    "ngl",
    "gen_tokens",
    "prompt_toks_per_s",
    "decode_toks_per_s",
    "ttft_short_ms_median",
    "ttft_medium_ms_median",
    "ttft_long_ms_median",
    "e2e_short_ms_median",
    "e2e_medium_ms_median",
    "e2e_long_ms_median",
    "status",
    "error_note",
]

LATENCY_FIELDS = [
    "timestamp",
    "model_path",
    "prompt_size",
    "run_idx",
    "ttft_ms",
    "e2e_ms",
    "output_tokens",
    "status",
]


class BenchmarkError(RuntimeError):
    pass


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify_model(path: str) -> str:
    base = Path(path).name
    if base.endswith(".gguf"):
        base = base[:-5]
    cleaned = "".join(ch for ch in base.replace(" ", "_") if ch.isalnum() or ch in "_.-")
    return cleaned or "model"


def expand_model_path(text: str) -> str:
    return str(Path(os.path.expanduser(text.strip())).resolve())


def read_models_file(path: Path) -> List[str]:
    models: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        models.append(expand_model_path(line))
    return models


def parse_prompts_file(path: Path) -> Dict[str, str]:
    sections = {"short": [], "medium": [], "long": []}
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if lowered in ("[short]", "[medium]", "[long]"):
            current = lowered.strip("[]")
            continue
        if current is not None:
            sections[current].append(raw)

    prompts = {k: "\n".join(v).strip() for k, v in sections.items()}
    missing = [k for k, v in prompts.items() if not v]
    if missing:
        raise BenchmarkError(f"missing prompt section(s): {', '.join(f'[{m}]' for m in missing)}")
    return prompts


def ensure_commands_exist(commands: Iterable[str]) -> None:
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        raise BenchmarkError("missing required command(s): " + ", ".join(missing))


def write_csv(path: Path, headers: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


def append_csv_row(path: Path, headers: List[str], row: Dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def run_llama_bench(
    llama_bench_bin: str,
    model_path: str,
    model_raw_dir: Path,
    ngl: int,
    threads: Optional[int],
    bench_n_prompt: int,
    bench_n_gen: int,
    bench_repetitions: int,
) -> Tuple[float, float]:
    bench_jsonl = model_raw_dir / "bench.jsonl"
    bench_stderr = model_raw_dir / "bench.stderr.log"

    cmd = [
        llama_bench_bin,
        # GGUF model file to benchmark
        "-m",
        model_path,
        # Number of input tokens to benchmark
        "-p",
        str(bench_n_prompt),
        # Number of output tokens to benchmark
        "-n",
        str(bench_n_gen),
        # how many layers to offload to GPU (99 usually means “all possible”)
        "-ngl",
        str(ngl),
        # Number of repetitions to run
        "-r",
        str(bench_repetitions),
        # Output format
        "-o",
        "jsonl",
    ]
    if threads is not None:
        cmd.extend(["-t", str(threads)])

    with bench_jsonl.open("w", encoding="utf-8") as out, bench_stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        raise BenchmarkError(f"llama-bench failed (see {bench_stderr.name})")

    prompt_vals: List[float] = []
    decode_vals: List[float] = []
    for raw in bench_jsonl.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        n_prompt = int(obj.get("n_prompt", 0))
        n_gen = int(obj.get("n_gen", 0))
        avg_ts = obj.get("avg_ts")
        if avg_ts is None:
            continue
        if n_prompt > 0 and n_gen == 0:
            prompt_vals.append(float(avg_ts))
        elif n_prompt == 0 and n_gen > 0:
            decode_vals.append(float(avg_ts))

    prompt_toks = statistics.mean(prompt_vals) if prompt_vals else 0.0
    decode_toks = statistics.mean(decode_vals) if decode_vals else 0.0
    return prompt_toks, decode_toks


def wait_for_health(host: str, port: int, timeout_s: int = 180) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def run_latency_once(
    host: str,
    port: int,
    prompt: str,
    temp: float,
    max_tokens: int,
) -> Tuple[float, float, int]:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first: Optional[float] = None
    end: Optional[float] = None
    output_tokens = -1

    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                end = time.perf_counter()
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content and first is None:
                    first = time.perf_counter()

            usage = obj.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                output_tokens = int(usage["completion_tokens"])

    if end is None:
        end = time.perf_counter()

    ttft_ms = -1.0 if first is None else (first - start) * 1000.0
    e2e_ms = (end - start) * 1000.0
    return ttft_ms, e2e_ms, output_tokens


def run_latency_suite(
    host: str,
    port: int,
    prompts: Dict[str, str],
    warmup: int,
    repeats: int,
    temp: float,
    gen_tokens: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    medians: Dict[str, Dict[str, float]] = {}

    for prompt_size in ("short", "medium", "long"):
        prompt = prompts[prompt_size]

        for _ in range(warmup):
            run_latency_once(host, port, prompt, temp=temp, max_tokens=gen_tokens)

        ttfts: List[float] = []
        e2es: List[float] = []
        for run_idx in range(1, repeats + 1):
            ttft_ms, e2e_ms, output_tokens = run_latency_once(
                host, port, prompt, temp=temp, max_tokens=gen_tokens
            )
            if ttft_ms >= 0:
                ttfts.append(ttft_ms)
            e2es.append(e2e_ms)
            rows.append(
                {
                    "prompt_size": prompt_size,
                    "run_idx": run_idx,
                    "ttft_ms": ttft_ms,
                    "e2e_ms": e2e_ms,
                    "output_tokens": output_tokens,
                    "status": "ok" if ttft_ms >= 0 else "partial",
                }
            )

        medians[prompt_size] = {
            "ttft_ms_median": statistics.median(ttfts) if ttfts else -1.0,
            "e2e_ms_median": statistics.median(e2es),
        }

    return rows, medians


def write_meta(path: Path, args: argparse.Namespace, resolved_models: List[str]) -> None:
    lines = [
        f"run_timestamp_utc={now_iso()}",
        f"models_file={args.models_file}",
        f"prompts_file={args.prompts_file}",
        f"models_count={len(resolved_models)}",
        f"ngl={args.ngl}",
        f"ctx={args.ctx}",
        f"gen_tokens={args.gen_tokens}",
        f"temp={args.temp}",
        f"warmup={args.warmup}",
        f"repeats={args.repeats}",
        f"threads={args.threads if args.threads is not None else 'auto'}",
        f"bench_n_prompt={args.bench_n_prompt}",
        f"bench_n_gen={args.bench_n_gen}",
        f"bench_repetitions={args.bench_repetitions}",
        f"llama_bench_bin={args.llama_bench_bin}",
        f"llama_server_bin={args.llama_server_bin}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark local GGUF models for speed, TTFT, and end-to-end latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("models_file", help="Path to models list file (one GGUF path per line).")
    p.add_argument(
        "prompts_file",
        nargs="?",
        default=DEFAULTS["prompts_file"],
        help="Path to prompts file with [short], [medium], [long] sections.",
    )
    p.add_argument("--out-dir-base", default=DEFAULTS["out_dir_base"])
    p.add_argument("--ngl", type=int, default=DEFAULTS["ngl"])
    p.add_argument("--ctx", type=int, default=DEFAULTS["ctx"])
    p.add_argument("--gen-tokens", type=int, default=DEFAULTS["gen_tokens"])
    p.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    p.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    p.add_argument("--repeats", type=int, default=DEFAULTS["repeats"])
    p.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    p.add_argument("--bench-n-prompt", type=int, default=DEFAULTS["bench_n_prompt"])
    p.add_argument("--bench-n-gen", type=int, default=DEFAULTS["bench_n_gen"])
    p.add_argument("--bench-repetitions", type=int, default=DEFAULTS["bench_repetitions"])
    p.add_argument("--server-host", default=DEFAULTS["server_host"])
    p.add_argument("--server-port", type=int, default=DEFAULTS["server_port"])
    p.add_argument("--llama-bench-bin", default=DEFAULTS["llama_bench_bin"])
    p.add_argument("--llama-server-bin", default=DEFAULTS["llama_server_bin"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    models_file = Path(args.models_file)
    prompts_file = Path(args.prompts_file)
    if not models_file.is_file():
        raise BenchmarkError(f"models file not found: {models_file}")
    if not prompts_file.is_file():
        raise BenchmarkError(f"prompts file not found: {prompts_file}")

    ensure_commands_exist([args.llama_bench_bin, args.llama_server_bin])
    prompts = parse_prompts_file(prompts_file)
    models = read_models_file(models_file)
    if not models:
        raise BenchmarkError(f"no model entries found in {models_file}")

    ts_slug = timestamp_slug()
    run_dir = Path(args.out_dir_base) / ts_slug
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = run_dir / "summary.csv"
    latency_csv = run_dir / "latency_by_prompt.csv"
    meta_file = run_dir / "run-meta.txt"
    write_csv(summary_csv, SUMMARY_FIELDS)
    write_csv(latency_csv, LATENCY_FIELDS)
    write_meta(meta_file, args, models)

    print(f"Run directory: {run_dir}")
    print()

    model_count = 0
    ok_count = 0
    server_proc: Optional[subprocess.Popen[str]] = None

    def cleanup_server(*_: object) -> None:
        nonlocal server_proc
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5)

    signal.signal(signal.SIGINT, cleanup_server)
    signal.signal(signal.SIGTERM, cleanup_server)

    for model_path in models:
        model_count += 1
        model_slug = slugify_model(model_path)
        model_raw_dir = raw_dir / model_slug
        model_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{model_count}] Model: {model_path}")

        if not Path(model_path).is_file():
            err = "model file not found"
            print(f"  - ERROR: {err}")
            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                {
                    "timestamp": ts_slug,
                    "model_path": model_path,
                    "model_name": model_slug,
                    "ctx": args.ctx,
                    "ngl": args.ngl,
                    "gen_tokens": args.gen_tokens,
                    "prompt_toks_per_s": "",
                    "decode_toks_per_s": "",
                    "ttft_short_ms_median": "",
                    "ttft_medium_ms_median": "",
                    "ttft_long_ms_median": "",
                    "e2e_short_ms_median": "",
                    "e2e_medium_ms_median": "",
                    "e2e_long_ms_median": "",
                    "status": "failed",
                    "error_note": err,
                },
            )
            print()
            continue

        prompt_toks = 0.0
        decode_toks = 0.0
        try:
            prompt_toks, decode_toks = run_llama_bench(
                args.llama_bench_bin,
                model_path,
                model_raw_dir,
                ngl=args.ngl,
                threads=args.threads,
                bench_n_prompt=args.bench_n_prompt,
                bench_n_gen=args.bench_n_gen,
                bench_repetitions=args.bench_repetitions,
            )

            server_log = model_raw_dir / "server.log"
            server_cmd = [
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
            ]
            if args.threads is not None:
                server_cmd.extend(["-t", str(args.threads)])

            with server_log.open("w", encoding="utf-8") as logf:
                server_proc = subprocess.Popen(server_cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

            if not wait_for_health(args.server_host, args.server_port, timeout_s=180):
                raise BenchmarkError("llama-server did not become ready")

            rows, medians = run_latency_suite(
                args.server_host,
                args.server_port,
                prompts,
                warmup=args.warmup,
                repeats=args.repeats,
                temp=args.temp,
                gen_tokens=args.gen_tokens,
            )

            (model_raw_dir / "latency_raw.json").write_text(
                json.dumps({"model_path": model_path, "rows": rows, "medians": medians}, indent=2),
                encoding="utf-8",
            )

            for row in rows:
                append_csv_row(
                    latency_csv,
                    LATENCY_FIELDS,
                    {
                        "timestamp": ts_slug,
                        "model_path": model_path,
                        "prompt_size": row["prompt_size"],
                        "run_idx": row["run_idx"],
                        "ttft_ms": f'{row["ttft_ms"]:.3f}',
                        "e2e_ms": f'{row["e2e_ms"]:.3f}',
                        "output_tokens": row["output_tokens"],
                        "status": row["status"],
                    },
                )

            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                {
                    "timestamp": ts_slug,
                    "model_path": model_path,
                    "model_name": model_slug,
                    "ctx": args.ctx,
                    "ngl": args.ngl,
                    "gen_tokens": args.gen_tokens,
                    "prompt_toks_per_s": f"{prompt_toks:.6f}",
                    "decode_toks_per_s": f"{decode_toks:.6f}",
                    "ttft_short_ms_median": f'{medians["short"]["ttft_ms_median"]:.3f}',
                    "ttft_medium_ms_median": f'{medians["medium"]["ttft_ms_median"]:.3f}',
                    "ttft_long_ms_median": f'{medians["long"]["ttft_ms_median"]:.3f}',
                    "e2e_short_ms_median": f'{medians["short"]["e2e_ms_median"]:.3f}',
                    "e2e_medium_ms_median": f'{medians["medium"]["e2e_ms_median"]:.3f}',
                    "e2e_long_ms_median": f'{medians["long"]["e2e_ms_median"]:.3f}',
                    "status": "ok",
                    "error_note": "",
                },
            )

            ok_count += 1
            print(f"  - Throughput: prompt={prompt_toks:.6f} tok/s, decode={decode_toks:.6f} tok/s")

        except (BenchmarkError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
            append_csv_row(
                summary_csv,
                SUMMARY_FIELDS,
                {
                    "timestamp": ts_slug,
                    "model_path": model_path,
                    "model_name": model_slug,
                    "ctx": args.ctx,
                    "ngl": args.ngl,
                    "gen_tokens": args.gen_tokens,
                    "prompt_toks_per_s": f"{prompt_toks:.6f}" if prompt_toks else "",
                    "decode_toks_per_s": f"{decode_toks:.6f}" if decode_toks else "",
                    "ttft_short_ms_median": "",
                    "ttft_medium_ms_median": "",
                    "ttft_long_ms_median": "",
                    "e2e_short_ms_median": "",
                    "e2e_medium_ms_median": "",
                    "e2e_long_ms_median": "",
                    "status": "failed",
                    "error_note": str(exc),
                },
            )
            print(f"  - ERROR: {exc}")
        finally:
            cleanup_server()
            server_proc = None
            print()

    print("Done.")
    print(f"Models processed: {model_count}")
    print(f"Models successful: {ok_count}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Latency CSV: {latency_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
