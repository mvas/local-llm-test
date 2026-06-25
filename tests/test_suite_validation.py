import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_performance import _selected_suite_names, _validate_selected_suites
from common import BenchmarkError
from suite_aider import validate_aider_setup
from suite_bfcl import validate_bfcl_setup


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "run_lm_evals": False,
        "run_humaneval": False,
        "run_mbpp": False,
        "run_bfcl": False,
        "run_aider": False,
        "bfcl_model_id_map_file": "models/bfcl-model-ids.txt",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class SelectedSuiteNamesTests(unittest.TestCase):
    def test_returns_selected_suites(self) -> None:
        args = _args(run_lm_evals=True, run_humaneval=True, run_bfcl=True)
        self.assertEqual(_selected_suite_names(args), ["lm-eval", "humaneval", "bfcl"])

    def test_empty_when_none_selected(self) -> None:
        self.assertEqual(_selected_suite_names(_args()), [])


class ValidateSelectedSuitesTests(unittest.TestCase):
    @patch("benchmark_performance.validate_aider_setup")
    @patch("benchmark_performance.ensure_commands_exist")
    def test_no_suite_selected_raises(
        self, mock_ensure: object, mock_aider: object
    ) -> None:
        with self.assertRaises(BenchmarkError) as ctx:
            _validate_selected_suites(_args(), ["/tmp/model.gguf"])
        self.assertIn("no benchmark suites selected", str(ctx.exception).lower())


class ValidateBfclSetupTests(unittest.TestCase):
    @patch("suite_bfcl.ensure_commands_exist")
    def test_missing_map_file(self, _: object) -> None:
        with self.assertRaises(BenchmarkError) as ctx:
            validate_bfcl_setup("missing-map.txt", ["/tmp/model.gguf"])
        self.assertIn("map file not found", str(ctx.exception).lower())

    @patch("suite_bfcl.ensure_commands_exist")
    def test_missing_model_mappings(self, _: object) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("/mapped/model.gguf=bfcl-id\n")
            map_file = handle.name
        try:
            with self.assertRaises(BenchmarkError) as ctx:
                validate_bfcl_setup(map_file, ["/mapped/model.gguf", "/missing/model.gguf"])
            msg = str(ctx.exception)
            self.assertIn("missing BFCL model id mapping", msg)
            self.assertIn("/missing/model.gguf", msg)
        finally:
            Path(map_file).unlink()


class ValidateAiderSetupTests(unittest.TestCase):
    @patch("suite_aider.ensure_commands_exist")
    def test_missing_checkout(self, _: object) -> None:
        missing = Path(tempfile.gettempdir()) / "nonexistent-aider-repo"
        with self.assertRaises(BenchmarkError) as ctx:
            validate_aider_setup(missing)
        self.assertIn("aider repo directory not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
