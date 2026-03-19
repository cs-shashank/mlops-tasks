"""
tests/test_validator.py
Unit and property-based tests for Validator functions: load_config, load_dataset.

Design properties covered:
  Property 1  — Config round-trip
  Property 2  — Config rejects invalid field types/values
  Property 3  — Config rejects missing fields
  Property 4  — Dataset rejects missing close column
  Property 5  — Dataset row count preserved after load
"""

import io
import textwrap
import pytest
import yaml
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Import functions under test
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import load_config, load_dataset


# ===========================================================================
# Helpers
# ===========================================================================

def write_config(tmp_path, data: dict) -> str:
    """Write a dict as YAML to a temp file and return the path string."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


def write_csv(tmp_path, content: str, filename="data.csv") -> str:
    """Write raw CSV text to a temp file and return the path string."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
    return str(p)


# ===========================================================================
# Unit tests — load_config
# ===========================================================================

class TestLoadConfigUnit:

    def test_valid_config_returns_correct_values(self, tmp_path):
        """Happy path: all required fields present with correct types."""
        path = write_config(tmp_path, {"seed": 42, "window": 5, "version": "v1"})
        cfg = load_config(path)
        assert cfg["seed"] == 42
        assert cfg["window"] == 5
        assert cfg["version"] == "v1"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Requirement 2.3 — non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_missing_seed_raises_value_error(self, tmp_path):
        """Requirement 2.4 — missing seed raises ValueError."""
        path = write_config(tmp_path, {"window": 5, "version": "v1"})
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(path)

    def test_missing_window_raises_value_error(self, tmp_path):
        """Requirement 2.4 — missing window raises ValueError."""
        path = write_config(tmp_path, {"seed": 42, "version": "v1"})
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(path)

    def test_missing_version_raises_value_error(self, tmp_path):
        """Requirement 2.4 — missing version raises ValueError."""
        path = write_config(tmp_path, {"seed": 42, "window": 5})
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(path)

    def test_seed_as_float_raises_value_error(self, tmp_path):
        """Requirement 2.2 / 2.5 — seed must be int, not float."""
        path = write_config(tmp_path, {"seed": 3.14, "window": 5, "version": "v1"})
        with pytest.raises(ValueError, match="seed"):
            load_config(path)

    def test_seed_as_string_raises_value_error(self, tmp_path):
        """Requirement 2.5 — seed as string is invalid."""
        path = write_config(tmp_path, {"seed": "42", "window": 5, "version": "v1"})
        with pytest.raises(ValueError, match="seed"):
            load_config(path)

    def test_window_zero_raises_value_error(self, tmp_path):
        """Requirement 2.2 — window must be > 0."""
        path = write_config(tmp_path, {"seed": 42, "window": 0, "version": "v1"})
        with pytest.raises(ValueError, match="window"):
            load_config(path)

    def test_window_negative_raises_value_error(self, tmp_path):
        """Requirement 2.2 — negative window is invalid."""
        path = write_config(tmp_path, {"seed": 42, "window": -3, "version": "v1"})
        with pytest.raises(ValueError, match="window"):
            load_config(path)

    def test_window_float_raises_value_error(self, tmp_path):
        """Requirement 2.5 — window as float is invalid."""
        path = write_config(tmp_path, {"seed": 42, "window": 5.0, "version": "v1"})
        with pytest.raises(ValueError, match="window"):
            load_config(path)

    def test_version_empty_string_raises_value_error(self, tmp_path):
        """Requirement 2.2 — empty version string is invalid."""
        path = write_config(tmp_path, {"seed": 42, "window": 5, "version": ""})
        with pytest.raises(ValueError, match="version"):
            load_config(path)

    def test_version_whitespace_only_raises_value_error(self, tmp_path):
        """Requirement 2.2 — whitespace-only version is invalid."""
        path = write_config(tmp_path, {"seed": 42, "window": 5, "version": "   "})
        with pytest.raises(ValueError, match="version"):
            load_config(path)

    def test_version_integer_raises_value_error(self, tmp_path):
        """Requirement 2.5 — version as integer is invalid."""
        path = write_config(tmp_path, {"seed": 42, "window": 5, "version": 1})
        with pytest.raises(ValueError, match="version"):
            load_config(path)

    def test_empty_yaml_raises_value_error(self, tmp_path):
        """Empty YAML file (parses to None) is not a valid config."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValueError):
            load_config(str(p))

    def test_extra_keys_are_allowed(self, tmp_path):
        """Extra keys in config beyond the required three are tolerated."""
        path = write_config(tmp_path, {
            "seed": 42, "window": 5, "version": "v1", "extra_key": "ignored"
        })
        cfg = load_config(path)
        assert cfg["seed"] == 42

    def test_negative_seed_is_valid(self, tmp_path):
        """seed can be any integer including negative."""
        path = write_config(tmp_path, {"seed": -1, "window": 1, "version": "v1"})
        cfg = load_config(path)
        assert cfg["seed"] == -1

    def test_window_one_is_valid(self, tmp_path):
        """window = 1 is the minimum valid value."""
        path = write_config(tmp_path, {"seed": 0, "window": 1, "version": "v1"})
        cfg = load_config(path)
        assert cfg["window"] == 1


# ===========================================================================
# Property-based tests — load_config
# (use tempfile internally — avoids Hypothesis function-scoped fixture warning)
# ===========================================================================

# Feature: mlops-batch-job, Property 1: Config round-trip
@given(
    seed    = st.integers(),
    window  = st.integers(min_value=1, max_value=1000),
    version = st.text(min_size=1).filter(lambda s: s.strip() != ""),
)
@settings(max_examples=100)
def test_config_round_trip(seed, window, version):
    """Property 1 — Any valid config survives a YAML serialise/deserialise cycle."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = write_config(Path(d), {"seed": seed, "window": window, "version": version})
        cfg = load_config(path)
    assert cfg["seed"]    == seed
    assert cfg["window"]  == window
    assert cfg["version"] == version


# Feature: mlops-batch-job, Property 2: Config rejects invalid field types/values
@given(
    seed   = st.one_of(st.floats(allow_nan=False), st.text(), st.none()),
    window = st.integers(min_value=1),
    version= st.text(min_size=1),
)
@settings(max_examples=100)
def test_config_rejects_invalid_seed(seed, window, version):
    """Property 2 — Non-integer seed always raises ValueError."""
    assume(not isinstance(seed, int))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = write_config(Path(d), {"seed": seed, "window": window, "version": version})
        with pytest.raises(ValueError):
            load_config(path)


# Feature: mlops-batch-job, Property 2: Config rejects invalid field types/values
@given(window = st.one_of(st.integers(max_value=0), st.floats(allow_nan=False), st.text()))
@settings(max_examples=100)
def test_config_rejects_invalid_window(window):
    """Property 2 — Non-positive or non-integer window always raises ValueError."""
    assume(not (isinstance(window, int) and window > 0))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = write_config(Path(d), {"seed": 42, "window": window, "version": "v1"})
        with pytest.raises(ValueError):
            load_config(path)


# Feature: mlops-batch-job, Property 3: Config rejects missing fields
@given(missing_key = st.sampled_from(["seed", "window", "version"]))
@settings(max_examples=30)
def test_config_rejects_missing_fields(missing_key):
    """Property 3 — Omitting any required field raises ValueError."""
    import tempfile
    base = {"seed": 42, "window": 5, "version": "v1"}
    del base[missing_key]
    with tempfile.TemporaryDirectory() as d:
        path = write_config(Path(d), base)
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(path)


# ===========================================================================
# Unit tests — load_dataset
# ===========================================================================

class TestLoadDatasetUnit:

    def test_valid_csv_returns_dataframe(self, tmp_path):
        """Happy path: well-formed CSV with close column loads correctly."""
        path = write_csv(tmp_path, "timestamp,close\n2020-01-01,100.0\n2020-01-02,101.0\n")
        df = load_dataset(path)
        assert isinstance(df, pd.DataFrame)
        assert "close" in df.columns
        assert len(df) == 2

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Requirement 3.3 — non-existent CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input CSV not found"):
            load_dataset(str(tmp_path / "missing.csv"))

    def test_empty_file_raises_value_error(self, tmp_path):
        """Requirement 3.5 — completely empty CSV raises ValueError."""
        path = write_csv(tmp_path, "")
        with pytest.raises((ValueError, Exception)):
            load_dataset(path)

    def test_header_only_raises_value_error(self, tmp_path):
        """Requirement 3.5 — header with no data rows raises ValueError."""
        path = write_csv(tmp_path, "timestamp,close\n")
        with pytest.raises(ValueError, match="empty"):
            load_dataset(path)

    def test_missing_close_column_raises_value_error(self, tmp_path):
        """Requirement 3.6 — CSV without close column raises ValueError."""
        path = write_csv(tmp_path, "timestamp,open,high,low,volume\n2020-01-01,100,101,99,1000\n")
        with pytest.raises(ValueError, match="close"):
            load_dataset(path)

    def test_non_csv_content_raises_value_error(self, tmp_path):
        """Requirement 3.4 — binary/non-CSV content raises ValueError."""
        p = tmp_path / "bad.csv"
        p.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        with pytest.raises((ValueError, Exception)):
            load_dataset(str(p))

    def test_extra_ohlcv_columns_are_preserved(self, tmp_path):
        """Provider-agnostic: extra columns beyond close are kept."""
        path = write_csv(tmp_path,
            "timestamp,open,high,low,close,volume_btc,volume_usd\n"
            "2020-01-01,100,101,99,100.5,1.5,15000\n"
        )
        df = load_dataset(path)
        assert "close" in df.columns
        assert len(df) == 1

    def test_close_column_case_sensitive(self, tmp_path):
        """close column match is case-sensitive — 'Close' is not accepted."""
        path = write_csv(tmp_path, "Close,volume\n100.0,1000\n")
        with pytest.raises(ValueError, match="close"):
            load_dataset(path)


# Feature: mlops-batch-job, Property 4: Dataset validation requires close column
@given(
    columns = st.lists(
        st.text(
            min_size=1,
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd", "Pc"),
                max_codepoint=127          # ASCII only — avoids Windows cp1252 issues
            )
        ),
        min_size=1, max_size=6, unique=True
    ).filter(lambda cols: "close" not in cols)
)
@settings(max_examples=100)
def test_dataset_rejects_missing_close(columns):
    """Property 4 — Any CSV without a close column raises ValueError."""
    import tempfile
    header = ",".join(columns)
    row    = ",".join(["1.0"] * len(columns))
    with tempfile.TemporaryDirectory() as d:
        path = write_csv(Path(d), f"{header}\n{row}\n")
        with pytest.raises(ValueError, match="close"):
            load_dataset(path)


# Feature: mlops-batch-job, Property 5: Dataset row count preserved after load
@given(
    rows = st.integers(min_value=1, max_value=500),
    extra_cols = st.integers(min_value=0, max_value=4),
)
@settings(max_examples=100, deadline=None)
def test_dataset_row_count_preserved(rows, extra_cols):
    """Property 5 — Row count in loaded DataFrame matches CSV data rows."""
    import tempfile
    extra_headers = ",".join(f"col{i}" for i in range(extra_cols))
    header = f"close{(',' + extra_headers) if extra_cols else ''}"
    data_rows = "\n".join("1.0" + (",1.0" * extra_cols) for _ in range(rows))
    with tempfile.TemporaryDirectory() as d:
        path = write_csv(Path(d), f"{header}\n{data_rows}\n")
        df = load_dataset(path)
    assert len(df) == rows
