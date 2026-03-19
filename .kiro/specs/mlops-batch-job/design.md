# Design Document — MLOps Rolling-Mean Signal Pipeline

**Version:** v1  
**Status:** Final  
**Scope:** T0 Technical Assessment — MetaStackerBandit ML/MLOps Engineering Internship

---

## 1. Overview

This document describes the architecture, component breakdown, data flow, and execution sequence for the MLOps batch job defined in the Requirements Document. The job loads OHLCV price data, computes a rolling mean on the `close` column, generates a binary trading signal, and writes structured metrics and logs.

---

## 2. Goals & Non-Goals

### Goals
- Single-file Python pipeline (`run.py`) that is easy to read and audit
- Fully deterministic — same inputs always produce same outputs
- All paths passed via CLI — no hardcoded values anywhere
- Metrics written in both success and error paths
- Runs identically locally and inside Docker

### Non-Goals
- Real-time / streaming processing
- Multi-asset or multi-signal support
- Database or external API integration
- Model training or ML inference

---

## 3. Architecture

The pipeline is a linear batch job with four sequential stages. There is no parallelism, no external state, and no network I/O.

```
┌─────────────────────────────────────────────────────────┐
│                        run.py                           │
│                                                         │
│  CLI Args                                               │
│  --input  --config  --output  --log-file                │
│      │         │        │          │                    │
│      ▼         ▼        │          ▼                    │
│  ┌──────────────────┐   │   ┌─────────────┐            │
│  │    Validator     │   │   │   Reporter  │            │
│  │ • Config YAML    │   │   │ • Logging   │            │
│  │ • Dataset CSV    │   │   │ • Metrics   │            │
│  └────────┬─────────┘   │   └──────┬──────┘            │
│           │             │          │                    │
│           ▼             │          │                    │
│  ┌──────────────────┐   │          │                    │
│  │    Processor     │   │          │                    │
│  │ • Rolling Mean   │   │          │                    │
│  │ • Signal Gen     │   │          │                    │
│  └────────┬─────────┘   │          │                    │
│           │             │          │                    │
│           └─────────────┴──────────┘                   │
│                         │                               │
│                    metrics.json                         │
│                    run.log                              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 CLI Layer (`parse_args`)
- Uses Python `argparse`
- Exposes four required flags: `--input`, `--config`, `--output`, `--log-file`
- No defaults — all paths must be supplied explicitly
- Runs before logging is initialized; argument errors go to stderr

### 4.2 Validator
Responsible for all input validation. Raises descriptive exceptions that the Reporter catches and writes to metrics.

**Config validation (`load_config`)**
- Checks file exists and is readable
- Parses YAML into a dict
- Asserts `seed` (int), `window` (positive int), `version` (non-empty string) are all present and correctly typed

**Dataset validation (`load_dataset`)**
- Checks file exists and is readable
- Parses CSV with `pandas.read_csv`
- Asserts the DataFrame is non-empty
- Asserts `close` column is present
- Provider-agnostic: all other columns are ignored

### 4.3 Processor
Stateless functions that operate on pandas Series. No side effects.

**`compute_rolling_mean(close, window)`**
- Uses `pandas.Series.rolling(window=window, min_periods=window).mean()`
- First `window - 1` rows → NaN (warm-up period, excluded downstream)

**`compute_signal(close, rolling_mean)`**
- Operates only on rows where `rolling_mean` is not NaN
- `signal = 1` if `close > rolling_mean`, else `0`
- Returns a Series aligned to the input index

### 4.4 Reporter
Two responsibilities: logging and metrics output.

**Logging (`setup_logging`)**
- Python `logging` module with two handlers:
  - `FileHandler` → `run.log` at DEBUG level (full detail)
  - `StreamHandler` → stdout at INFO level (human summary)
- Timestamp + severity on every line
- Single logger instance shared across all stages

**Metrics (`write_metrics`)**
- Serializes a dict to JSON with `json.dump`
- Called in both success and error paths — always written before exit
- Success payload: `version`, `rows_processed`, `metric`, `value`, `latency_ms`, `seed`, `status`
- Error payload: `version` (if available), `status`, `error_message`, `latency_ms`

---

## 5. Data Flow

```
data.csv                config.yaml
    │                        │
    ▼                        ▼
load_dataset()          load_config()
    │                        │
    │ DataFrame              │ {seed, window, version}
    │                        │
    └──────────┬─────────────┘
               │
               ▼
    numpy.random.seed(seed)          ← reproducibility anchor
               │
               ▼
    df["close"]  ──►  compute_rolling_mean(close, window)
               │                │
               │         rolling_mean (Series, NaN for first window-1 rows)
               │                │
               └────────┬───────┘
                        │
                        ▼
             compute_signal(close, rolling_mean)
                        │
                  signal (Series, valid rows only)
                        │
                        ▼
              signal_rate = mean(signal)
              rows_processed = len(signal)
              latency_ms = elapsed time
                        │
                        ▼
                  metrics.json
                  run.log
                  stdout (metrics JSON)
```

---

## 6. Execution Sequence

```
run.py starts
     │
     ├─ 1. parse_args()
     │       └─ Resolve: --input, --config, --output, --log-file
     │
     ├─ 2. setup_logging(log_file)
     │       └─ Open run.log, attach stdout handler
     │
     ├─ 3. Record job start timestamp → log
     │
     ├─ 4. load_config(config_path)          [Validator]
     │       ├─ Parse YAML
     │       ├─ Validate fields + types
     │       ├─ Log: version, seed, window
     │       └─ numpy.random.seed(seed)
     │
     ├─ 5. load_dataset(input_path)          [Validator]
     │       ├─ Read CSV → DataFrame
     │       ├─ Assert non-empty + close column present
     │       └─ Log: row count, column names
     │
     ├─ 6. compute_rolling_mean(close, window)   [Processor]
     │       ├─ Rolling arithmetic mean
     │       └─ Log: warm-up rows, valid rows
     │
     ├─ 7. compute_signal(close, rolling_mean)   [Processor]
     │       ├─ Binary comparison on valid rows
     │       └─ Log: rows_processed, signal_rate, 1s vs 0s
     │
     ├─ 8. Compute latency_ms
     │
     ├─ 9. write_metrics(output_path, success_payload)   [Reporter]
     │       └─ Log: metrics written
     │
     ├─ 10. Print metrics JSON → stdout
     │
     ├─ 11. Log: job end, status=success
     │
     └─ exit(0)

On any exception in steps 4–9:
     ├─ Log: exception details
     ├─ write_metrics(output_path, error_payload)
     ├─ Print error JSON → stderr
     └─ exit(1)
```

---

## 7. Error Handling Strategy

All validation and processing errors follow a single pattern:
1. Raise a typed Python exception with a descriptive message
2. The top-level `try/except` in `main()` catches it
3. Reporter writes an error `metrics.json` with `status: "error"` and `error_message`
4. Logger records the full traceback at ERROR level
5. Process exits with code `1`

This ensures `metrics.json` is always written, even on failure — a hard requirement for downstream monitoring.

| Error Scenario | Exception Type | Metrics `version` included? |
|---|---|---|
| Missing config file | `FileNotFoundError` | No |
| Bad YAML / missing keys | `ValueError` | No |
| Missing input CSV | `FileNotFoundError` | Yes (if config loaded) |
| Unparseable CSV | `ValueError` (wrapped) | Yes |
| Empty dataset | `ValueError` | Yes |
| Missing `close` column | `ValueError` | Yes |
| Any unexpected error | `Exception` | Yes (if config loaded) |

---

## 8. Reproducibility Design

Determinism is guaranteed by three properties:

1. **Seed** — `numpy.random.seed(seed)` is called immediately after config validation, before any data processing
2. **Deterministic operations** — `rolling().mean()` and `>` comparison have no stochastic component; they produce identical results for identical inputs regardless of seed
3. **Config-driven** — all parameters (`window`, `seed`, `version`) come from `config.yaml`, not environment variables or runtime state

The seed primarily future-proofs the pipeline for cases where stochastic steps (e.g., sampling, augmentation) are added later. For the current implementation, the rolling mean and signal are fully deterministic without it.

---

## 9. Docker Design

```
Base image: python:3.9-slim
    │
    ├─ COPY requirements.txt → pip install
    ├─ COPY run.py
    ├─ COPY config.yaml
    └─ COPY data.csv

CMD: python run.py \
       --input    data.csv \
       --config   config.yaml \
       --output   metrics.json \
       --log-file run.log
```

Design decisions:
- Data files are baked into the image at build time — no volume mounts required for the default run
- All paths in CMD are relative to `WORKDIR /app` — portable, no absolute paths
- `PYTHONUNBUFFERED=1` ensures stdout is flushed immediately so metrics JSON appears in `docker run` output
- Exit code propagates from Python `sys.exit()` directly to the container exit code

---

## 10. File Structure

```
mlops-task/
├── run.py            # Full pipeline (Validator + Processor + Reporter + CLI)
├── config.yaml       # Runtime parameters: seed, window, version
├── data.csv          # OHLCV input dataset (10,000 rows)
├── requirements.txt  # numpy, pandas, PyYAML
├── Dockerfile        # python:3.9-slim, bundles data files
├── README.md         # Local + Docker run instructions
├── metrics.json      # Sample output from a successful run
├── run.log           # Sample log from a successful run
└── DESIGN.md         # This document
```

---

## 11. Assumptions & Constraints

| Item | Decision |
|---|---|
| Only `close` column is required | All other OHLCV columns are ignored; column names are provider-agnostic |
| Warm-up rows excluded | First `window - 1` rows produce NaN rolling mean and are excluded from `rows_processed` and `signal_rate` |
| `signal_rate` rounding | Rounded to 4 decimal places in metrics output |
| `latency_ms` | Wall-clock time from job start to metrics write, measured with `time.perf_counter()` |
| Single-file implementation | All logic in `run.py` — no submodules — to keep the codebase auditable at a glance |
| Python version | 3.9+ (matches Docker base image) |

---

## 12. Data Models

### Config (parsed from YAML)

| Field | Type | Constraints |
|---|---|---|
| `seed` | `int` | any integer |
| `window` | `int` | > 0 |
| `version` | `str` | non-empty |

### Dataset (parsed from CSV)

| Column | Type | Notes |
|---|---|---|
| `close` | numeric | required; other columns may exist and vary by provider |

### Metrics JSON — Success

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4990,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```

### Metrics JSON — Error

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Dataset missing required 'close' column"
}
```

`version` is omitted from the error payload when config could not be loaded.

### Log Entry Format

```
2024-01-15 10:23:01,234 INFO  Job started
2024-01-15 10:23:01,250 INFO  Config loaded: seed=42, window=5, version=v1
2024-01-15 10:23:01,260 INFO  Dataset loaded: 10000 rows
2024-01-15 10:23:01,270 INFO  Rolling mean computed (window=5)
2024-01-15 10:23:01,275 INFO  Signal generated
2024-01-15 10:23:01,280 INFO  Metrics: signal_rate=0.4990, rows_processed=9996
2024-01-15 10:23:01,285 INFO  Job completed successfully
```

---

## 13. Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system.*

### Property 1: Config round-trip
*For any* valid config dict with integer `seed`, positive integer `window`, and non-empty string `version`, serializing it to YAML and loading it via `load_config` should produce a dict with identical field values.

**Validates: Requirements 2.1**

### Property 2: Config validation rejects invalid fields
*For any* config where at least one of `seed` is not an integer, `window` is not a positive integer, or `version` is an empty string, `load_config` should raise a `ValueError`.

**Validates: Requirements 2.2, 2.5**

### Property 3: Config validation rejects missing fields
*For any* config YAML that omits one or more of `seed`, `window`, `version`, `load_config` should raise a `ValueError`.

**Validates: Requirements 2.4**

### Property 4: Dataset validation requires close column
*For any* CSV that does not contain a column named `close`, `load_dataset` should raise a `ValueError`.

**Validates: Requirements 3.2, 3.6**

### Property 5: Dataset round-trip preserves row count
*For any* valid CSV with a `close` column and at least one data row, loading it via `load_dataset` should return a DataFrame whose row count equals the number of data rows in the CSV.

**Validates: Requirements 3.1**

### Property 6: Rolling mean correctness and NaN invariant
*For any* numeric `close` Series and any positive integer `window`, `compute_rolling_mean` should return a Series where: (a) the first `window - 1` values are NaN, and (b) all subsequent values equal the arithmetic mean of the corresponding `window`-sized window of `close` values.

**Validates: Requirements 4.1, 4.2**

### Property 7: Signal correctness on non-NaN rows
*For any* pair of equal-length numeric Series `close` and `rolling_mean` where `rolling_mean` has no NaN values, `compute_signal` should return a Series of the same length where each value is 1 if `close[i] > rolling_mean[i]` and 0 otherwise.

**Validates: Requirements 5.1, 5.2, 4.3**

### Property 8: Reproducibility — identical inputs produce identical outputs
*For any* dataset, config, and CLI arguments, running the full pipeline twice with the same seed should produce byte-identical metrics JSON output.

**Validates: Requirements 8.1, 5.3**

### Property 9: Success metrics payload completeness
*For any* successful run, the metrics JSON should contain all required fields: `version`, `rows_processed`, `metric` (= `"signal_rate"`), `value` (float, 4 decimal places), `latency_ms` (integer), `seed`, and `status` (= `"success"`).

**Validates: Requirements 6.1**

### Property 10: Error metrics payload correctness
*For any* error condition, the metrics JSON should contain `status` = `"error"` and a non-empty `error_message` string.

**Validates: Requirements 6.2**

### Property 11: Log entries contain timestamp and severity
*For any* log entry written by the Reporter, the entry should include a parseable timestamp and a severity level (INFO, ERROR, etc.).

**Validates: Requirements 7.4**

### Property 12: Log records lifecycle events on success
*For any* successful run, the log file should contain entries for: job start, config loaded (with seed/window/version), dataset row count, rolling mean step, signal generation step, metrics summary, and job completion.

**Validates: Requirements 7.2**

### Property 13: Log records error details on failure
*For any* run that encounters a validation or processing error, the log file should contain an entry with the error message before the job exits.

**Validates: Requirements 7.3**

---

## 14. Testing Strategy

### Property-Based Testing

Library: **Hypothesis** (Python). Each property test runs a minimum of 100 iterations.

| Design Property | Test Description |
|---|---|
| Property 1 | Config YAML round-trip |
| Property 2 | Config rejects invalid field types/values |
| Property 3 | Config rejects missing fields |
| Property 4 | Dataset rejects missing `close` column |
| Property 5 | Dataset row count preserved after load |
| Property 6 | Rolling mean correctness and NaN invariant |
| Property 7 | Signal correctness on non-NaN rows |
| Property 8 | Reproducibility across two runs |
| Property 9 | Success metrics payload completeness |
| Property 10 | Error metrics payload correctness |
| Property 11 | Log entry format (timestamp + severity) |
| Property 12 | Log lifecycle events on success |
| Property 13 | Log error details on failure |

### Unit Tests

- CLI parses all four required arguments correctly
- Missing CLI argument causes non-zero exit
- Config file not found raises error and writes error metrics
- Dataset file not found raises error
- Non-CSV file raises error
- Empty CSV raises error
- `numpy.random.seed` is called with the correct seed value
- Successful run prints metrics JSON to stdout
- Successful run exits with code 0; error run exits with code 1
- Log file is created at the `--log-file` path

### Test File Layout

```
tests/
├── test_validator.py    # unit + property tests for Validator
├── test_processor.py    # unit + property tests for Processor
├── test_reporter.py     # unit + property tests for Reporter
└── test_integration.py  # end-to-end CLI tests
```
