# MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal MLOps-style batch job that loads OHLCV data, computes a rolling-mean signal, and writes structured metrics + logs. Runs locally or inside Docker with a single command.

## Requirements

- Python 3.9+
- Dependencies: `numpy`, `pandas`, `PyYAML`

```bash
pip install -r requirements.txt
```

## Local Run

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

### CLI Arguments

| Argument | Description |
|---|---|
| `--input` | Path to OHLCV CSV file (must contain a `close` column) |
| `--config` | Path to YAML config file |
| `--output` | Path for output metrics JSON |
| `--log-file` | Path for log file |

### Config Format (`config.yaml`)

```yaml
seed: 42
window: 5
version: "v1"
```

## Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`, runs the pipeline, and prints the final metrics JSON to stdout. Exit code is `0` on success, non-zero on failure.

## Example Output

### metrics.json

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 70,
  "seed": 42,
  "status": "success"
}
```

`rows_processed` = total rows minus warm-up rows (`window - 1`). With 10,000 rows and `window=5`, that's 9,996 valid rows.

### Error output

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found.",
  "latency_ms": 5
}
```

## Running Tests

```bash
python -m pytest tests/ -q
```

## Project Structure

```
run.py          # Single-file pipeline (validator + processor + reporter + CLI)
config.yaml     # Job configuration
data.csv        # 10,000-row OHLCV dataset
requirements.txt
Dockerfile
metrics.json    # Sample output from a successful run
run.log         # Sample log from a successful run
tests/
  test_validator.py
  test_processor.py
  test_reporter.py
  test_integration.py
  conftest.py
```
