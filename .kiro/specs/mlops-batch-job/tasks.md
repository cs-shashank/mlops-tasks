# Implementation Plan: MLOps Rolling-Mean Signal Pipeline

## Overview

Single-file Python batch job (`run.py`) with a linear Validator → Processor → Reporter pipeline. All logic lives in `run.py`; tests are split across four files under `tests/`. Implementation proceeds bottom-up: project scaffold → validator → processor → reporter → CLI wiring → Docker/docs.

## Tasks

- [x] 1. Scaffold project structure and dependencies
  - Create `requirements.txt` with `numpy`, `pandas`, `PyYAML`; add `hypothesis` for tests
  - Create `config.yaml` with `seed`, `window`, `version` fields
  - Create a 10,000-row `data.csv` with at minimum a `close` column (OHLCV layout)
  - Create empty `run.py` with top-level imports and a `main()` stub
  - Create `tests/` directory with empty `__init__.py` and the four test file stubs
  - _Requirements: 1.3, 9.1, 9.6_

- [x] 2. Implement Validator — `load_config()`
  - [x] 2.1 Write `load_config(config_path)` in `run.py`
    - Open and parse YAML; raise `FileNotFoundError` if file missing
    - Assert `seed` is `int`, `window` is `int > 0`, `version` is non-empty `str`; raise `ValueError` on any violation
    - Return validated dict `{seed, window, version}`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.2 Write property test for `load_config` — Property 1: Config round-trip
    - **Property 1: Config round-trip**
    - **Validates: Requirements 2.1**
    - In `tests/test_validator.py`, use Hypothesis `@given` with valid `seed` (int), `window` (int > 0), `version` (non-empty str); write to temp YAML; assert `load_config` returns identical values

  - [ ]* 2.3 Write property test for `load_config` — Property 2: Rejects invalid fields
    - **Property 2: Config validation rejects invalid fields**
    - **Validates: Requirements 2.2, 2.5**
    - Generate configs where at least one field violates type/value constraints; assert `load_config` raises `ValueError`

  - [ ]* 2.4 Write property test for `load_config` — Property 3: Rejects missing fields
    - **Property 3: Config validation rejects missing fields**
    - **Validates: Requirements 2.4**
    - Generate configs with one or more of `seed`/`window`/`version` omitted; assert `load_config` raises `ValueError`

  - [ ]* 2.5 Write unit tests for `load_config`
    - Config file not found raises `FileNotFoundError`
    - Missing individual required fields each raise `ValueError`
    - Invalid types (e.g. `window: "five"`) raise `ValueError`
    - _Requirements: 2.3, 2.4, 2.5_

- [x] 3. Implement Validator — `load_dataset()`
  - [x] 3.1 Write `load_dataset(input_path)` in `run.py`
    - Read CSV with `pandas.read_csv`; raise `FileNotFoundError` if file missing
    - Raise `ValueError` if file is not parseable as CSV, if DataFrame is empty, or if `close` column is absent
    - Return the DataFrame
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 3.2 Write property test for `load_dataset` — Property 4: Rejects missing close column
    - **Property 4: Dataset validation requires close column**
    - **Validates: Requirements 3.2, 3.6**
    - Generate CSVs with arbitrary column names that exclude `close`; assert `load_dataset` raises `ValueError`

  - [ ]* 3.3 Write property test for `load_dataset` — Property 5: Row count preserved
    - **Property 5: Dataset round-trip preserves row count**
    - **Validates: Requirements 3.1**
    - Generate valid CSVs with a `close` column and N >= 1 rows; assert returned DataFrame has exactly N rows

  - [ ]* 3.4 Write unit tests for `load_dataset`
    - File not found raises `FileNotFoundError`
    - Non-CSV content raises `ValueError`
    - Empty CSV (header only) raises `ValueError`
    - CSV without `close` column raises `ValueError`
    - _Requirements: 3.3, 3.4, 3.5, 3.6_

- [x] 4. Checkpoint — Validator complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Processor — `compute_rolling_mean()` and `compute_signal()`
  - [x] 5.1 Write `compute_rolling_mean(close, window)` in `run.py`
    - Use `close.rolling(window=window, min_periods=window).mean()`
    - Return the resulting Series (first `window - 1` values are NaN)
    - _Requirements: 4.1, 4.2_

  - [x] 5.2 Write `compute_signal(close, rolling_mean)` in `run.py`
    - Filter to rows where `rolling_mean` is not NaN
    - Assign `1` where `close > rolling_mean`, else `0`
    - Return a Series aligned to the valid-row index
    - _Requirements: 5.1, 5.2, 4.3_

  - [ ]* 5.3 Write property test for `compute_rolling_mean` — Property 6: Rolling mean correctness and NaN invariant
    - **Property 6: Rolling mean correctness and NaN invariant**
    - **Validates: Requirements 4.1, 4.2**
    - Generate numeric Series of length >= 1 and window 1–20; assert first `window - 1` values are NaN and remaining values equal the arithmetic mean of each window

  - [ ]* 5.4 Write property test for `compute_signal` — Property 7: Signal correctness on non-NaN rows
    - **Property 7: Signal correctness on non-NaN rows**
    - **Validates: Requirements 5.1, 5.2, 4.3**
    - Generate equal-length numeric Series with no NaN; assert each signal value is exactly `1` if `close[i] > rolling_mean[i]` else `0`

  - [ ]* 5.5 Write unit tests for Processor
    - `compute_rolling_mean` with `window=1` returns the original Series (no NaN)
    - `compute_rolling_mean` with `window > len(series)` returns all NaN
    - `compute_signal` returns only 0s and 1s
    - `compute_signal` output length equals number of non-NaN rolling mean rows
    - _Requirements: 4.2, 4.3, 5.2_

- [x] 6. Implement Reporter — `setup_logging()` and `write_metrics()`
  - [x] 6.1 Write `setup_logging(log_file)` in `run.py`
    - Attach a `FileHandler` at DEBUG level and a `StreamHandler` at INFO level to the root logger
    - Format: `%(asctime)s %(levelname)-5s %(message)s` (timestamp + severity on every line)
    - Return the configured logger
    - _Requirements: 7.1, 7.4_

  - [x] 6.2 Write `write_metrics(output_path, payload)` in `run.py`
    - Serialize `payload` dict to JSON with `json.dump`; write to `output_path`
    - Called in both success and error paths
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 6.3 Write property test for Reporter — Property 9: Success metrics payload completeness
    - **Property 9: Success metrics payload completeness**
    - **Validates: Requirements 6.1**
    - Generate valid pipeline inputs; run full pipeline; assert metrics JSON contains all required fields with correct types (`value` rounded to 4 dp, `latency_ms` integer, `status` = `"success"`)

  - [ ]* 6.4 Write property test for Reporter — Property 10: Error metrics payload correctness
    - **Property 10: Error metrics payload correctness**
    - **Validates: Requirements 6.2**
    - Trigger each error condition; assert written metrics JSON has `status` = `"error"` and non-empty `error_message`

  - [ ]* 6.5 Write property test for Reporter — Property 11: Log entries contain timestamp and severity
    - **Property 11: Log entries contain timestamp and severity**
    - **Validates: Requirements 7.4**
    - Run pipeline with a temp log file; read each line; assert every line matches the expected timestamp + severity pattern

  - [ ]* 6.6 Write property test for Reporter — Property 12: Log records lifecycle events on success
    - **Property 12: Log records lifecycle events on success**
    - **Validates: Requirements 7.2**
    - Run a successful pipeline; assert log contains entries for job start, config loaded, dataset row count, rolling mean step, signal step, metrics summary, and job completion

  - [ ]* 6.7 Write property test for Reporter — Property 13: Log records error details on failure
    - **Property 13: Log records error details on failure**
    - **Validates: Requirements 7.3**
    - Trigger a validation error; assert log file contains the error message before job exits

  - [ ]* 6.8 Write unit tests for Reporter
    - `write_metrics` creates the output file with valid JSON
    - `setup_logging` creates the log file at the specified path
    - Log file contains at least one line after a successful run
    - _Requirements: 6.3, 7.1_

- [x] 7. Checkpoint — Core functions complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Wire everything together in `main()` and `parse_args()`
  - [x] 8.1 Write `parse_args()` in `run.py`
    - Use `argparse`; declare `--input`, `--config`, `--output`, `--log-file` as required arguments with no defaults
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 8.2 Complete `main()` in `run.py`
    - Call `parse_args()` → `setup_logging()` → log job start → `load_config()` → `numpy.random.seed(seed)` → `load_dataset()` → `compute_rolling_mean()` → `compute_signal()` → compute `signal_rate`, `rows_processed`, `latency_ms` → `write_metrics()` success payload → print metrics JSON to stdout → log job end → `sys.exit(0)`
    - Wrap steps 3–9 in `try/except`; on any exception log error, call `write_metrics()` error payload, print to stderr, `sys.exit(1)`
    - _Requirements: 1.1, 1.2, 2.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.2, 7.3, 8.1, 8.2_

  - [ ]* 8.3 Write property test for reproducibility — Property 8: Identical inputs produce identical outputs
    - **Property 8: Reproducibility — identical inputs produce identical outputs**
    - **Validates: Requirements 8.1, 5.3**
    - Generate a dataset and config; invoke `main()` twice via subprocess or direct call; assert both metrics JSON outputs are byte-identical

  - [ ]* 8.4 Write integration tests in `tests/test_integration.py`
    - CLI parses all four required arguments correctly
    - Missing any CLI argument causes non-zero exit
    - Successful run exits with code 0 and prints metrics JSON to stdout
    - Error run exits with code 1
    - Log file is created at the `--log-file` path after a successful run
    - `numpy.random.seed` is called with the correct seed value
    - _Requirements: 1.1, 1.2, 6.4, 6.5, 7.1, 8.2_

- [x] 9. Checkpoint — Full pipeline wired and tested
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Create Dockerfile and finalize deliverables
  - [x] 10.1 Write `Dockerfile`
    - Base image `python:3.9-slim`, `WORKDIR /app`, `PYTHONUNBUFFERED=1`
    - `COPY requirements.txt` then `pip install --no-cache-dir -r requirements.txt`
    - `COPY run.py config.yaml data.csv`
    - `CMD ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]`
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6_

  - [x] 10.2 Generate sample `metrics.json` and `run.log`
    - Run the pipeline locally once to produce representative sample output files
    - _Requirements: 9.4, 10.3_

  - [x] 10.3 Write `README.md`
    - Document all four CLI arguments and their descriptions
    - Include local run command and Docker build + run commands
    - Include example `metrics.json` output
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with a minimum of 100 iterations per property
- All 13 correctness properties from the design document are covered by property-based tests
- Checkpoints at tasks 4, 7, 9, and 11 ensure incremental validation
