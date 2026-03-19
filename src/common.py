from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List


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

def wait_for_health(host: str, port: int, timeout_s: int = 180) -> bool:
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