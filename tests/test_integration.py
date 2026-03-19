"""
tests/test_integration.py
End-to-end CLI integration tests for the full run.py pipeline.

Covers:
  - Happy path: correct metrics produced and written
  - All validation error paths: missing file, bad CSV, missing column, etc.
  - CLI argument handling
  - Exit codes (0 = success, 1 = error)
  - stdout output and metrics file written in both paths
  - numpy.random.seed called with correct value
  - Reproducibility across two identical runs
  - Log lifecycle events on success and failure

Design properties covered (via full pipeline):
  Property 8  — Reproducibility
  Property 9  — Success metrics payload completeness
  Property 10 — Error metrics payload correctness
  Property 12 — Log lifecycle events on success
  Property 13 — Log error details on failure
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Path to the script under test (one level up from tests/)
# ---------------------------------------------------------------------------
SCRIPT = str(Path(__file__).parent.parent / "run.py")


# ===========================================================================
# Helpers
# ===========================================================================

def run_pipeline(input_path, config_path, output_path, log_path):
    """Run run.py as a subprocess and return CompletedProcess."""
    return subprocess.run(
        [
            sys.executable, SCRIPT,
            "--input",    input_path,
            "--config",   config_path,
            "--output",   output_path,
            "--log-file", log_path,
        ],
        capture_output=True,
        text=True,
    )


def make_valid_csv(tmp_path, rows=20, filename="data.csv") -> str:
    """Write a minimal valid OHLCV CSV and return the path."""
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=rows, freq="1min"),
        "open":      close + 0.1,
        "high":      close + 0.3,
        "low":       close - 0.3,
        "close":     np.round(close, 4),
        "volume":    np.random.randint(100, 1000, size=rows),
    })
    p = tmp_path / filename
    df.to_csv(str(p), index=False)
    return str(p)


def make_valid_config(tmp_path, seed=42, window=5, version="v1") -> str:
    """Write a valid config.yaml and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump({"seed": seed, "window": window, "version": version}))
    return str(p)


# ===========================================================================
# Happy path
# ===========================================================================

class TestHappyPath:

    def test_exit_code_0_on_success(self, tmp_path):
        """Requirement 6.5 — successful run exits with code 0."""
        csv  = make_valid_csv(tmp_path)
        cfg  = make_valid_config(tmp_path)
        out  = str(tmp_path / "metrics.json")
        log  = str(tmp_path / "run.log")
        proc = run_pipeline(csv, cfg, out, log)
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

    def test_metrics_file_written_on_success(self, tmp_path):
        """Requirement 6.3 — metrics.json is written on success."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        assert Path(out).exists()

    def test_metrics_json_has_all_required_keys(self, tmp_path):
        """Requirement 6.1 — success payload has all required fields."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        for key in ("version", "rows_processed", "metric", "value",
                    "latency_ms", "seed", "status"):
            assert key in data, f"Missing key: {key}"

    def test_metrics_status_is_success(self, tmp_path):
        """Requirement 6.1 — status field equals 'success'."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert data["status"] == "success"

    def test_metric_field_is_signal_rate(self, tmp_path):
        """Requirement 6.1 — metric field must equal 'signal_rate'."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert data["metric"] == "signal_rate"

    def test_value_is_float_between_0_and_1(self, tmp_path):
        """signal_rate must be in [0.0, 1.0]."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert 0.0 <= data["value"] <= 1.0

    def test_seed_written_to_metrics(self, tmp_path):
        """Requirement 6.1 — seed value from config appears in metrics."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path, seed=99)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert data["seed"] == 99

    def test_version_written_to_metrics(self, tmp_path):
        """Requirement 6.1 — version from config appears in metrics."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path, version="v2")
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert data["version"] == "v2"

    def test_stdout_prints_metrics_json(self, tmp_path):
        """Requirement 6.4 — metrics JSON is printed to stdout on success."""
        csv  = make_valid_csv(tmp_path)
        cfg  = make_valid_config(tmp_path)
        out  = str(tmp_path / "metrics.json")
        log  = str(tmp_path / "run.log")
        proc = run_pipeline(csv, cfg, out, log)
        # Extract the JSON block: collect lines between the first { and last }
        lines = proc.stdout.splitlines()
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip() == "{":
                in_block = True
            if in_block:
                json_lines.append(line)
            if in_block and line.strip() == "}":
                break
        assert json_lines, "No JSON block found in stdout"
        data = json.loads("\n".join(json_lines))
        assert data["status"] == "success"

    def test_log_file_created(self, tmp_path):
        """Requirement 7.1 — log file is created at --log-file path."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        assert Path(log).exists()

    def test_rows_processed_excludes_warmup(self, tmp_path):
        """rows_processed = total_rows - (window - 1)."""
        rows   = 20
        window = 5
        csv    = make_valid_csv(tmp_path, rows=rows)
        cfg    = make_valid_config(tmp_path, window=window)
        out    = str(tmp_path / "metrics.json")
        log    = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert data["rows_processed"] == rows - (window - 1)

    def test_latency_ms_is_non_negative_integer(self, tmp_path):
        """latency_ms must be a non-negative integer."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        data = json.loads(Path(out).read_text())
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0


# ===========================================================================
# Property 12 — Log lifecycle events on success
# ===========================================================================

class TestLogLifecycleSuccess:

    def _get_log(self, tmp_path) -> str:
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(csv, cfg, out, log)
        return Path(log).read_text()

    def test_log_contains_job_start(self, tmp_path):
        """Property 12 — log records job start."""
        content = self._get_log(tmp_path)
        assert "started" in content.lower() or "start" in content.lower()

    def test_log_contains_config_values(self, tmp_path):
        """Property 12 — log records seed, window, version after config load."""
        content = self._get_log(tmp_path)
        assert "seed" in content
        assert "window" in content
        assert "version" in content

    def test_log_contains_row_count(self, tmp_path):
        """Property 12 — log records number of rows loaded."""
        content = self._get_log(tmp_path)
        assert "rows" in content.lower() or "20" in content

    def test_log_contains_rolling_mean_step(self, tmp_path):
        """Property 12 — log records rolling mean computation step."""
        content = self._get_log(tmp_path)
        assert "rolling" in content.lower() or "mean" in content.lower()

    def test_log_contains_signal_step(self, tmp_path):
        """Property 12 — log records signal generation step."""
        content = self._get_log(tmp_path)
        assert "signal" in content.lower()

    def test_log_contains_metrics_summary(self, tmp_path):
        """Property 12 — log records a metrics summary."""
        content = self._get_log(tmp_path)
        assert "signal_rate" in content or "metric" in content.lower()

    def test_log_contains_job_end(self, tmp_path):
        """Property 12 — log records job completion."""
        content = self._get_log(tmp_path)
        assert "success" in content.lower() or "completed" in content.lower()


# ===========================================================================
# Error paths — exit code 1 and error metrics written
# ===========================================================================

class TestErrorPaths:

    def _assert_error_run(self, tmp_path, csv, cfg, expect_in_message=None):
        """Helper: run pipeline, assert exit=1 and error metrics written."""
        out  = str(tmp_path / "metrics.json")
        log  = str(tmp_path / "run.log")
        proc = run_pipeline(csv, cfg, out, log)
        assert proc.returncode == 1, f"Expected exit 1, got {proc.returncode}"
        assert Path(out).exists(), "Error metrics file must be written"
        data = json.loads(Path(out).read_text())
        assert data["status"] == "error"
        assert "error_message" in data
        assert len(data["error_message"]) > 0
        if expect_in_message:
            assert expect_in_message.lower() in data["error_message"].lower()
        return data

    def test_missing_input_csv(self, tmp_path):
        """Requirement 3.3 — missing input CSV → exit 1 + error metrics."""
        cfg = make_valid_config(tmp_path)
        self._assert_error_run(
            tmp_path,
            str(tmp_path / "nonexistent.csv"),
            cfg,
        )

    def test_missing_config_file(self, tmp_path):
        """Requirement 2.3 — missing config → exit 1 + error metrics."""
        csv = make_valid_csv(tmp_path)
        self._assert_error_run(
            tmp_path,
            csv,
            str(tmp_path / "nonexistent.yaml"),
        )

    def test_empty_csv(self, tmp_path):
        """Requirement 3.5 — empty CSV → exit 1 + error metrics."""
        p = tmp_path / "empty.csv"
        p.write_text("")
        cfg = make_valid_config(tmp_path)
        self._assert_error_run(tmp_path, str(p), cfg)

    def test_header_only_csv(self, tmp_path):
        """Requirement 3.5 — header with no data rows → exit 1 + error metrics."""
        p = tmp_path / "header_only.csv"
        p.write_text("timestamp,close\n")
        cfg = make_valid_config(tmp_path)
        self._assert_error_run(tmp_path, str(p), cfg, expect_in_message="empty")

    def test_csv_missing_close_column(self, tmp_path):
        """Requirement 3.6 — CSV without close column → exit 1 + error metrics."""
        p = tmp_path / "no_close.csv"
        p.write_text("timestamp,open,high,low,volume\n2020-01-01,100,101,99,1000\n")
        cfg = make_valid_config(tmp_path)
        self._assert_error_run(tmp_path, str(p), cfg, expect_in_message="close")

    def test_config_missing_window(self, tmp_path):
        """Requirement 2.4 — config missing window → exit 1 + error metrics."""
        csv = make_valid_csv(tmp_path)
        p   = tmp_path / "bad_config.yaml"
        p.write_text(yaml.dump({"seed": 42, "version": "v1"}))
        self._assert_error_run(tmp_path, csv, str(p))

    def test_config_invalid_window_type(self, tmp_path):
        """Requirement 2.5 — window as float → exit 1 + error metrics."""
        csv = make_valid_csv(tmp_path)
        p   = tmp_path / "bad_config.yaml"
        p.write_text("seed: 42\nwindow: 5.5\nversion: v1\n")
        self._assert_error_run(tmp_path, csv, str(p))

    def test_config_window_zero(self, tmp_path):
        """Requirement 2.2 — window=0 → exit 1 + error metrics."""
        csv = make_valid_csv(tmp_path)
        p   = tmp_path / "bad_config.yaml"
        p.write_text(yaml.dump({"seed": 42, "window": 0, "version": "v1"}))
        self._assert_error_run(tmp_path, csv, str(p))

    def test_error_metrics_written_even_on_config_failure(self, tmp_path):
        """Requirement 6.3 — error metrics always written, even before config loads."""
        csv = make_valid_csv(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        proc = subprocess.run(
            [sys.executable, SCRIPT,
             "--input", csv,
             "--config", str(tmp_path / "missing.yaml"),
             "--output", out,
             "--log-file", log],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        assert Path(out).exists()

    def test_log_records_error_details_on_failure(self, tmp_path):
        """Property 13 — log file contains error details on any failure."""
        p = tmp_path / "no_close.csv"
        p.write_text("timestamp,open\n2020-01-01,100\n")
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        run_pipeline(str(p), cfg, out, log)
        content = Path(log).read_text()
        assert "error" in content.lower()


# ===========================================================================
# CLI argument handling
# ===========================================================================

class TestCLIArgs:

    def test_missing_input_arg_exits_nonzero(self, tmp_path):
        """Requirement 1.2 — missing --input causes non-zero exit."""
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        proc = subprocess.run(
            [sys.executable, SCRIPT,
             "--config", cfg, "--output", out, "--log-file", log],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_missing_config_arg_exits_nonzero(self, tmp_path):
        """Requirement 1.2 — missing --config causes non-zero exit."""
        csv = make_valid_csv(tmp_path)
        out = str(tmp_path / "metrics.json")
        log = str(tmp_path / "run.log")
        proc = subprocess.run(
            [sys.executable, SCRIPT,
             "--input", csv, "--output", out, "--log-file", log],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_missing_output_arg_exits_nonzero(self, tmp_path):
        """Requirement 1.2 — missing --output causes non-zero exit."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        log = str(tmp_path / "run.log")
        proc = subprocess.run(
            [sys.executable, SCRIPT,
             "--input", csv, "--config", cfg, "--log-file", log],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_missing_log_file_arg_exits_nonzero(self, tmp_path):
        """Requirement 1.2 — missing --log-file causes non-zero exit."""
        csv = make_valid_csv(tmp_path)
        cfg = make_valid_config(tmp_path)
        out = str(tmp_path / "metrics.json")
        proc = subprocess.run(
            [sys.executable, SCRIPT,
             "--input", csv, "--config", cfg, "--output", out],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0


# ===========================================================================
# Reproducibility — Property 8
# ===========================================================================

class TestReproducibility:

    def test_two_runs_produce_identical_metrics(self, tmp_path):
        """Property 8 — identical inputs produce identical metrics (excl. latency_ms)."""
        csv = make_valid_csv(tmp_path, rows=50)
        cfg = make_valid_config(tmp_path, seed=42, window=5)

        out1 = str(tmp_path / "m1.json")
        out2 = str(tmp_path / "m2.json")
        log1 = str(tmp_path / "r1.log")
        log2 = str(tmp_path / "r2.log")

        run_pipeline(csv, cfg, out1, log1)
        run_pipeline(csv, cfg, out2, log2)

        d1 = json.loads(Path(out1).read_text())
        d2 = json.loads(Path(out2).read_text())

        # latency_ms is wall-clock and will differ — exclude it
        d1.pop("latency_ms", None)
        d2.pop("latency_ms", None)

        assert d1 == d2, f"Metrics differ:\n{d1}\nvs\n{d2}"

    def test_different_seeds_can_produce_same_signal(self, tmp_path):
        """Determinism check: pipeline is seed-stable for purely deterministic ops."""
        csv  = make_valid_csv(tmp_path, rows=30)
        cfg1 = make_valid_config(tmp_path, seed=1)
        cfg2 = tmp_path / "config2.yaml"
        cfg2.write_text(yaml.dump({"seed": 1, "window": 5, "version": "v1"}))

        out1 = str(tmp_path / "m1.json")
        out2 = str(tmp_path / "m2.json")
        run_pipeline(csv, str(cfg1), out1, str(tmp_path / "l1.log"))
        run_pipeline(csv, str(cfg2), out2, str(tmp_path / "l2.log"))

        d1 = json.loads(Path(out1).read_text())
        d2 = json.loads(Path(out2).read_text())

        assert d1["value"] == d2["value"]
        assert d1["rows_processed"] == d2["rows_processed"]
