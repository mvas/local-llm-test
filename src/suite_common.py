
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple



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