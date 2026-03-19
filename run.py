"""
MLOps Batch Job — Rolling-Mean Signal Pipeline
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class _SafeFormatter(logging.Formatter):
    """Formatter that makes log messages printable ASCII and single-line."""

    # Characters that Python's splitlines() treats as line breaks (including
    # ASCII control chars that some terminals/readers interpret as separators)
    _CTRL_ESCAPE = {i: f"\\x{i:02x}" for i in range(32) if i not in (9,)}
    # Also escape DEL (127) and everything above ASCII printable range
    _CTRL_ESCAPE[127] = "\\x7f"

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        # Interpolate % args first, then sanitize
        try:
            msg = record.msg % record.args if record.args else str(record.msg)
        except Exception:
            msg = str(record.msg)
        # Sanitize: escape CR/LF explicitly, then all other control chars and
        # non-ASCII so the log file is pure printable ASCII
        msg = msg.replace("\r", "\\r").replace("\n", "\\n")
        sanitized = []
        for ch in msg:
            cp = ord(ch)
            if cp in self._CTRL_ESCAPE:
                sanitized.append(self._CTRL_ESCAPE[cp])
            elif cp > 126:
                # Non-ASCII: use \uXXXX notation
                sanitized.append(f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}")
            else:
                sanitized.append(ch)
        msg = "".join(sanitized)
        record.msg = msg if msg.strip() else "(empty)"
        record.args = None
        return super().format(record)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = _SafeFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler — full detail (UTF-8 so all characters are preserved)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above (replace unencodable chars on Windows)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if hasattr(ch.stream, "reconfigure"):
        try:
            ch.stream.reconfigure(errors="replace")
        except Exception:
            pass

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value pairs)")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")

    # Type validation
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got {cfg['window']!r}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got {cfg['version']!r}")

    return cfg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no rows)")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if df["close"].isnull().all():
        raise ValueError("Column 'close' contains only NaN values")

    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def compute_rolling_mean(close: pd.Series, window: int) -> pd.Series:
    """
    Standard rolling mean.  The first (window-1) rows will be NaN
    and are excluded from signal computation downstream.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    signal = 1  if close > rolling_mean
    signal = 0  otherwise
    Rows where rolling_mean is NaN (warm-up period) are excluded.
    """
    valid = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[valid] = (close[valid] > rolling_mean[valid]).astype(int)
    return signal


# ---------------------------------------------------------------------------
# Metrics / output helpers
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    job_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("MLOps batch job started")
    logger.info(f"  input   : {args.input}")
    logger.info(f"  config  : {args.config}")
    logger.info(f"  output  : {args.output}")
    logger.info(f"  log-file: {args.log_file}")
    logger.info("=" * 60)

    version = "unknown"  # populated after config load

    try:
        # ---- 1. Config -------------------------------------------------------
        logger.info("Loading config …")
        cfg = load_config(args.config)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]
        logger.info(f"Config loaded — version={version!r}  seed={seed}  window={window}")

        # Set global RNG seed for reproducibility
        np.random.seed(seed)
        logger.debug(f"numpy.random.seed({seed}) set")

        # ---- 2. Dataset ------------------------------------------------------
        logger.info(f"Loading dataset from '{args.input}' …")
        df = load_dataset(args.input)
        total_rows = len(df)
        logger.info(f"Dataset loaded — {total_rows:,} rows, columns: {list(df.columns)}")

        close = df["close"].reset_index(drop=True)
        null_count = int(close.isnull().sum())
        if null_count:
            logger.warning(f"'close' column has {null_count} NaN rows — they will be excluded")

        # ---- 3. Rolling mean -------------------------------------------------
        logger.info(f"Computing rolling mean (window={window}) …")
        rolling_mean = compute_rolling_mean(close, window)
        warmup_rows  = int(rolling_mean.isnull().sum())
        valid_rows   = total_rows - warmup_rows
        logger.info(
            f"Rolling mean computed — "
            f"warm-up rows (NaN): {warmup_rows}, valid rows: {valid_rows:,}"
        )
        logger.debug(f"Rolling mean sample (rows 0-6): {rolling_mean.iloc[:7].tolist()}")

        # ---- 4. Signal -------------------------------------------------------
        logger.info("Generating binary signal (1 = close > rolling_mean) …")
        signal = compute_signal(close, rolling_mean)

        valid_signal  = signal.dropna()
        rows_processed = len(valid_signal)
        signal_rate    = float(valid_signal.mean())

        ones  = int((valid_signal == 1).sum())
        zeros = int((valid_signal == 0).sum())
        logger.info(
            f"Signal generated — rows_processed={rows_processed:,}, "
            f"signal_rate={signal_rate:.4f}  (1s={ones:,}, 0s={zeros:,})"
        )

        # ---- 5. Metrics + timing ---------------------------------------------
        latency_ms = round((time.perf_counter() - job_start) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        logger.info("Writing metrics …")
        write_metrics(args.output, metrics)
        logger.info(f"Metrics written to '{args.output}'")
        logger.info(f"Final metrics: {json.dumps(metrics)}")

        logger.info("=" * 60)
        logger.info(f"Job completed successfully  (latency={latency_ms} ms)")
        logger.info("=" * 60)

        # Print final metrics to stdout for Docker capture
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        latency_ms = round((time.perf_counter() - job_start) * 1000)
        logger.error(f"Job failed: {exc}", exc_info=True)

        error_payload = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    latency_ms,
        }
        try:
            write_metrics(args.output, error_payload)
            logger.info(f"Error metrics written to '{args.output}'")
        except Exception as write_exc:
            logger.error(f"Could not write error metrics: {write_exc}")

        logger.info("=" * 60)
        logger.info("Job ended with FAILURE")
        logger.info("=" * 60)

        print(json.dumps(error_payload, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
