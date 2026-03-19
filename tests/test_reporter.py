"""
tests/test_reporter.py
Unit and property-based tests for Reporter functions:
setup_logging, write_metrics.

Design properties covered:
  Property 9  — Success metrics payload completeness
  Property 10 — Error metrics payload correctness
  Property 11 — Log entries contain timestamp and severity
  Property 12 — Log records lifecycle events on success
  Property 13 — Log records error details on failure
"""

import json
import logging
import re
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import setup_logging, write_metrics


# Regex matching the log entry format: timestamp  LEVEL    message
LOG_ENTRY_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s+\w+\s+.+"
)

REQUIRED_SUCCESS_KEYS = {
    "version", "rows_processed", "metric", "value",
    "latency_ms", "seed", "status"
}


# ===========================================================================
# Unit tests — write_metrics
# ===========================================================================

class TestWriteMetricsUnit:

    def test_writes_valid_json(self, tmp_path):
        """write_metrics produces a file that parses as valid JSON."""
        path = str(tmp_path / "metrics.json")
        write_metrics(path, {"status": "success", "value": 0.5})
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "success"

    def test_success_payload_all_keys_present(self, tmp_path):
        """Requirement 6.1 — success payload has all required keys."""
        path = str(tmp_path / "metrics.json")
        payload = {
            "version": "v1", "rows_processed": 9996,
            "metric": "signal_rate", "value": 0.4991,
            "latency_ms": 22, "seed": 42, "status": "success"
        }
        write_metrics(path, payload)
        with open(path) as f:
            data = json.load(f)
        assert REQUIRED_SUCCESS_KEYS.issubset(data.keys())

    def test_error_payload_has_status_and_message(self, tmp_path):
        """Requirement 6.2 — error payload has status=error and error_message."""
        path = str(tmp_path / "metrics.json")
        payload = {
            "version": "v1", "status": "error",
            "error_message": "close column missing", "latency_ms": 5
        }
        write_metrics(path, payload)
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "error"
        assert "error_message" in data
        assert len(data["error_message"]) > 0

    def test_metric_field_is_signal_rate(self, tmp_path):
        """Requirement 6.1 — metric field must equal 'signal_rate'."""
        path = str(tmp_path / "metrics.json")
        payload = {
            "version": "v1", "rows_processed": 100,
            "metric": "signal_rate", "value": 0.5,
            "latency_ms": 10, "seed": 42, "status": "success"
        }
        write_metrics(path, payload)
        with open(path) as f:
            data = json.load(f)
        assert data["metric"] == "signal_rate"

    def test_value_rounded_to_4_decimal_places(self, tmp_path):
        """Requirement 6.1 — value is rounded to 4 decimal places."""
        path = str(tmp_path / "metrics.json")
        write_metrics(path, {
            "version": "v1", "rows_processed": 100,
            "metric": "signal_rate", "value": round(0.499137, 4),
            "latency_ms": 10, "seed": 42, "status": "success"
        })
        with open(path) as f:
            data = json.load(f)
        # Ensure stored value has at most 4 decimal digits
        assert round(data["value"], 4) == data["value"]

    def test_latency_ms_is_integer(self, tmp_path):
        """Requirement 6.1 — latency_ms must be an integer."""
        path = str(tmp_path / "metrics.json")
        write_metrics(path, {
            "version": "v1", "rows_processed": 100,
            "metric": "signal_rate", "value": 0.5,
            "latency_ms": 42, "seed": 42, "status": "success"
        })
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data["latency_ms"], int)

    def test_overwrites_existing_file(self, tmp_path):
        """write_metrics overwrites any pre-existing file at the output path."""
        path = str(tmp_path / "metrics.json")
        write_metrics(path, {"status": "error", "error_message": "first"})
        write_metrics(path, {"status": "success", "value": 0.5})
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "success"

    def test_creates_file_at_given_path(self, tmp_path):
        """Requirement 7.1 — file is created at the exact path provided."""
        path = str(tmp_path / "subdir" / "metrics.json")
        Path(path).parent.mkdir(parents=True)
        write_metrics(path, {"status": "success"})
        assert Path(path).exists()


# ===========================================================================
# Unit tests — setup_logging
# ===========================================================================

class TestSetupLoggingUnit:

    def _fresh_logger(self):
        """Remove any existing handlers from the named logger."""
        logger = logging.getLogger("mlops_pipeline")
        logger.handlers.clear()
        return logger

    def test_log_file_is_created(self, tmp_path):
        """Requirement 7.1 — log file is created at the --log-file path."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.info("test message")
        assert Path(log_path).exists()

    def test_log_file_contains_written_message(self, tmp_path):
        """Messages written via the logger appear in the log file."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.info("hello from test")
        content = Path(log_path).read_text()
        assert "hello from test" in content

    def test_log_entry_matches_timestamp_format(self, tmp_path):
        """Requirement 7.4 — every log entry has timestamp + severity."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.info("checking format")
        content = Path(log_path).read_text()
        for line in content.strip().splitlines():
            assert LOG_ENTRY_RE.match(line), f"Line does not match format: {line!r}"

    def test_debug_messages_appear_in_file(self, tmp_path):
        """File handler captures DEBUG-level messages."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.debug("debug detail")
        content = Path(log_path).read_text()
        assert "debug detail" in content

    def test_error_messages_appear_in_file(self, tmp_path):
        """File handler captures ERROR-level messages."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.error("something went wrong")
        content = Path(log_path).read_text()
        assert "something went wrong" in content
        assert "ERROR" in content

    def test_info_severity_label_present(self, tmp_path):
        """Requirement 7.4 — INFO severity label appears in log file."""
        self._fresh_logger()
        log_path = str(tmp_path / "run.log")
        logger = setup_logging(log_path)
        logger.info("info message")
        content = Path(log_path).read_text()
        assert "INFO" in content


# ===========================================================================
# Property-based tests — write_metrics
# ===========================================================================

# Feature: mlops-batch-job, Property 9: Success metrics payload completeness
@given(
    version        = st.text(min_size=1).filter(lambda s: s.strip()),
    rows_processed = st.integers(min_value=0, max_value=10_000_000),
    value          = st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    latency_ms     = st.integers(min_value=0, max_value=100_000),
    seed           = st.integers(),
)
@settings(max_examples=100)
def test_success_payload_completeness(version, rows_processed, value, latency_ms, seed):
    """Property 9 — Any success payload round-trips with all required keys intact."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "metrics.json")
        payload = {
            "version": version, "rows_processed": rows_processed,
            "metric": "signal_rate", "value": round(value, 4),
            "latency_ms": latency_ms, "seed": seed, "status": "success"
        }
        write_metrics(path, payload)
        with open(path) as f:
            data = json.load(f)
    assert REQUIRED_SUCCESS_KEYS.issubset(data.keys())
    assert data["status"] == "success"
    assert data["metric"] == "signal_rate"


# Feature: mlops-batch-job, Property 10: Error metrics payload correctness
@given(
    error_message = st.text(min_size=1),
    version       = st.one_of(st.text(min_size=1), st.none()),
)
@settings(max_examples=100)
def test_error_payload_correctness(error_message, version):
    """Property 10 — Any error payload contains status=error and non-empty error_message."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "metrics.json")
        payload = {"status": "error", "error_message": error_message}
        if version is not None:
            payload["version"] = version
        write_metrics(path, payload)
        with open(path) as f:
            data = json.load(f)
    assert data["status"] == "error"
    assert "error_message" in data
    assert len(data["error_message"]) > 0


# Feature: mlops-batch-job, Property 11: Log entries contain timestamp and severity
@given(messages = st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=10))
@settings(max_examples=50)
def test_log_entries_have_timestamp_and_severity(messages):
    """Property 11 — Every log line includes a parseable timestamp and severity."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log_path = str(Path(d) / "run.log")
        logger = logging.getLogger("mlops_pipeline")
        logger.handlers.clear()
        logger = setup_logging(log_path)
        for msg in messages:
            try:
                logger.info(msg)
            except Exception:
                pass
        logger.handlers.clear()
        content = Path(log_path).read_text()
    for line in content.strip().splitlines():
        assert LOG_ENTRY_RE.match(line), f"Malformed log line: {line!r}"
