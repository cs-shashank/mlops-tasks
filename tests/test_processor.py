"""
tests/test_processor.py
Unit and property-based tests for Processor functions:
compute_rolling_mean, compute_signal.

Design properties covered:
  Property 6  — Rolling mean correctness and NaN invariant
  Property 7  — Signal correctness on non-NaN rows
  Property 8  — Reproducibility — identical inputs produce identical outputs
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import compute_rolling_mean, compute_signal


# ===========================================================================
# Helpers
# ===========================================================================

def make_close(values: list) -> pd.Series:
    return pd.Series(values, dtype=float)


# ===========================================================================
# Unit tests — compute_rolling_mean
# ===========================================================================

class TestComputeRollingMeanUnit:

    def test_first_window_minus_one_rows_are_nan(self):
        """Requirement 4.2 — first window-1 rows must be NaN."""
        close = make_close([1, 2, 3, 4, 5, 6, 7])
        result = compute_rolling_mean(close, window=3)
        assert result.iloc[0] != result.iloc[0]   # NaN != NaN
        assert result.iloc[1] != result.iloc[1]
        assert not pd.isna(result.iloc[2])

    def test_window_1_produces_no_nans(self):
        """Window of 1 means every row has a full window — no NaN expected."""
        close = make_close([10.0, 20.0, 30.0])
        result = compute_rolling_mean(close, window=1)
        assert result.isna().sum() == 0
        pd.testing.assert_series_equal(result, close, check_names=False)

    def test_correct_mean_values(self):
        """Requirement 4.1 — verify exact arithmetic mean values."""
        close = make_close([2.0, 4.0, 6.0, 8.0, 10.0])
        result = compute_rolling_mean(close, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(4.0)   # (2+4+6)/3
        assert result.iloc[3] == pytest.approx(6.0)   # (4+6+8)/3
        assert result.iloc[4] == pytest.approx(8.0)   # (6+8+10)/3

    def test_window_equals_series_length(self):
        """Window equal to series length — only last row is non-NaN."""
        close = make_close([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_rolling_mean(close, window=5)
        assert result.isna().sum() == 4
        assert result.iloc[4] == pytest.approx(3.0)

    def test_window_larger_than_series(self):
        """Window larger than series — all values are NaN."""
        close = make_close([1.0, 2.0, 3.0])
        result = compute_rolling_mean(close, window=10)
        assert result.isna().all()

    def test_output_length_matches_input(self):
        """Output series must have the same length as input."""
        close = make_close(list(range(20)))
        result = compute_rolling_mean(close, window=5)
        assert len(result) == len(close)

    def test_single_row_window_1(self):
        """Edge case: 1-row series with window=1."""
        close = make_close([42.0])
        result = compute_rolling_mean(close, window=1)
        assert result.iloc[0] == pytest.approx(42.0)

    def test_uniform_values_mean_equals_value(self):
        """If all close values are identical, rolling mean equals that value."""
        close = make_close([5.0] * 10)
        result = compute_rolling_mean(close, window=4)
        valid = result.dropna()
        assert all(abs(v - 5.0) < 1e-9 for v in valid)

    def test_nan_count_equals_window_minus_one(self):
        """Exactly window-1 NaN values for any valid input."""
        close = make_close(list(range(1, 11)))
        for w in range(1, 8):
            result = compute_rolling_mean(close, window=w)
            assert result.isna().sum() == w - 1


# ===========================================================================
# Unit tests — compute_signal
# ===========================================================================

class TestComputeSignalUnit:

    def test_close_greater_than_mean_gives_1(self):
        """Requirement 5.1 — signal=1 when close > rolling_mean."""
        close        = make_close([10.0, 11.0, 12.0])
        rolling_mean = make_close([9.0,  9.0,  9.0])
        signal = compute_signal(close, rolling_mean)
        valid  = signal.dropna()
        assert (valid == 1).all()

    def test_close_less_than_mean_gives_0(self):
        """Requirement 5.1 — signal=0 when close <= rolling_mean."""
        close        = make_close([8.0, 7.0, 6.0])
        rolling_mean = make_close([9.0, 9.0, 9.0])
        signal = compute_signal(close, rolling_mean)
        valid  = signal.dropna()
        assert (valid == 0).all()

    def test_close_equal_to_mean_gives_0(self):
        """Requirement 5.1 — signal is strictly greater; equal gives 0."""
        close        = make_close([9.0, 9.0])
        rolling_mean = make_close([9.0, 9.0])
        signal = compute_signal(close, rolling_mean)
        valid  = signal.dropna()
        assert (valid == 0).all()

    def test_nan_rolling_mean_rows_excluded(self):
        """Requirement 4.3 / 5.2 — NaN rows in rolling_mean become NaN in signal."""
        close        = make_close([1.0, 2.0, 3.0, 10.0, 10.0])
        rolling_mean = make_close([np.nan, np.nan, np.nan, 5.0, 5.0])
        signal = compute_signal(close, rolling_mean)
        assert pd.isna(signal.iloc[0])
        assert pd.isna(signal.iloc[1])
        assert pd.isna(signal.iloc[2])
        assert signal.iloc[3] == 1
        assert signal.iloc[4] == 1

    def test_output_length_matches_input(self):
        """Output length equals input length."""
        close        = make_close([1.0, 2.0, 3.0, 4.0, 5.0])
        rolling_mean = make_close([2.0, 2.0, 2.0, 2.0, 2.0])
        signal = compute_signal(close, rolling_mean)
        assert len(signal) == len(close)

    def test_signal_values_are_binary(self):
        """Valid (non-NaN) signal values must be exactly 0 or 1."""
        close        = make_close([1.0, 5.0, 3.0, 7.0, 2.0])
        rolling_mean = make_close([3.0, 3.0, 3.0, 3.0, 3.0])
        signal = compute_signal(close, rolling_mean)
        valid  = signal.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_mixed_signal(self):
        """Verify alternating 1/0 on known input."""
        close        = make_close([10.0, 1.0, 10.0, 1.0])
        rolling_mean = make_close([5.0,  5.0,  5.0,  5.0])
        signal = compute_signal(close, rolling_mean)
        valid  = signal.dropna().tolist()
        assert valid == [1, 0, 1, 0]

    def test_all_nan_rolling_mean_all_excluded(self):
        """All NaN rolling_mean → all signal values excluded (NaN)."""
        close        = make_close([1.0, 2.0, 3.0])
        rolling_mean = make_close([np.nan, np.nan, np.nan])
        signal = compute_signal(close, rolling_mean)
        assert signal.isna().all()


# ===========================================================================
# Property-based tests — compute_rolling_mean
# ===========================================================================

# Feature: mlops-batch-job, Property 6: Rolling mean correctness and NaN invariant
@given(
    values = st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9),
        min_size=1, max_size=200
    ),
    window = st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_rolling_mean_nan_invariant(values, window):
    """Property 6a — first window-1 values are always NaN."""
    close  = make_close(values)
    result = compute_rolling_mean(close, window)
    expected_nans = min(window - 1, len(values))
    assert result.isna().sum() == expected_nans


@given(
    values = st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1, max_size=200
    ),
    window = st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_rolling_mean_correctness(values, window):
    """Property 6b — each valid value equals arithmetic mean of its window."""
    close  = make_close(values)
    result = compute_rolling_mean(close, window)
    for i in range(window - 1, len(values)):
        expected = sum(values[i - window + 1 : i + 1]) / window
        assert result.iloc[i] == pytest.approx(expected, rel=1e-5, abs=1e-8)


# ===========================================================================
# Property-based tests — compute_signal
# ===========================================================================

# Feature: mlops-batch-job, Property 7: Signal correctness on non-NaN rows
@given(
    n      = st.integers(min_value=1, max_value=200),
    close_vals = st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1, max_size=200
    ),
)
@settings(max_examples=100)
def test_signal_correctness_no_nan(close_vals, n):
    """Property 7 — signal=1 iff close > rolling_mean, for all valid rows."""
    assume(len(close_vals) >= 1)
    close        = make_close(close_vals)
    rolling_mean = make_close([0.0] * len(close_vals))  # no NaN rows
    signal = compute_signal(close, rolling_mean)
    for i, (c, m, s) in enumerate(zip(close_vals, [0.0]*len(close_vals), signal)):
        expected = 1 if c > m else 0
        assert s == expected, f"row {i}: close={c}, mean={m}, expected={expected}, got={s}"


# Feature: mlops-batch-job, Property 8: Reproducibility — identical inputs produce identical outputs
@given(
    values = st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=1000.0),
        min_size=5, max_size=100
    ),
    window = st.integers(min_value=1, max_value=10),
    seed   = st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_reproducibility(values, window, seed):
    """Property 8 — identical inputs always produce identical signal outputs."""
    np.random.seed(seed)
    close  = make_close(values)
    rm1    = compute_rolling_mean(close, window)
    sig1   = compute_signal(close, rm1).dropna().tolist()

    np.random.seed(seed)
    close  = make_close(values)
    rm2    = compute_rolling_mean(close, window)
    sig2   = compute_signal(close, rm2).dropna().tolist()

    assert sig1 == sig2
