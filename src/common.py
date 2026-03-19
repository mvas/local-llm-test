from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelContext:
    args: argparse.Namespace
    ts_slug: str
    model_path: str
    model_slug: str
    model_raw_dir: Path

@dataclass(frozen=True)
class Metric:
    timestamp: str
    model_path: str
    model_name: str
    suite: str
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

def start_server(
    llama_server_bin: str,
    server_log: Path,
    model_path: str,
    ngl: int,
    ctx: int,
    host: str,
    port: int,
    threads: Optional[int],
    temp: Optional[float],
    top_p: Optional[float],
    seed: Optional[int]) -> subprocess.Popen[str]:
    cmd = [
        llama_server_bin,
        "-m",
        model_path,
        "-ngl",
        str(ngl),
        "-c",
        str(ctx),
        "--reasoning",
        "off",
        "--host",
        host,
        "--port",
        str(port)
    ]
    if threads is not None:
        cmd.extend(["-t", str(threads)])
    if threads is not None:
        cmd.extend(["--temp", str(temp)])
    if threads is not None:
        cmd.extend(["--top-p", str(top_p)])
    if threads is not None:
        cmd.extend(["--seed", str(seed)])

    with server_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    if not _wait_for_health(host, port, timeout_s=180):
        raise BenchmarkError("llama-server did not become ready")
    return proc
    
def _wait_for_health(host: str, port: int, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    return False


def cleanup_server(server_proc: Optional[subprocess.Popen[str]]) -> None:
    if server_proc is None or server_proc.poll() is not None:
        return
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=5)