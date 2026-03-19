# Requirements Document

## Introduction

A minimal MLOps-style batch job in Python that demonstrates reproducibility (deterministic runs via config and seed), observability (structured logs and machine-readable metrics), and deployment readiness (Dockerized, one-command run). The job loads OHLCV data, computes a rolling mean on the close price, generates a binary signal, and writes structured metrics and logs.

## Glossary

- **Job**: The batch processing program (`run.py`) that orchestrates the full pipeline.
- **Config**: A YAML file containing runtime parameters: `seed`, `window`, and `version`.
- **Dataset**: A CSV file containing OHLCV columns. The exact column names may vary by data provider; only the `close` column is required and used for processing.
- **Rolling_Mean**: The rolling arithmetic mean of the `close` column computed over a sliding window of size `window`.
- **Signal**: A binary integer (0 or 1) derived per row by comparing `close` to `Rolling_Mean`.
- **Signal_Rate**: The arithmetic mean of all Signal values computed over rows where Rolling_Mean is defined.
- **Metrics**: A JSON object written to the output file containing job results and status.
- **Log**: A structured text file recording job lifecycle events and any errors.
- **Validator**: The component responsible for validating Config and Dataset inputs.
- **Processor**: The component responsible for computing Rolling_Mean and Signal values.
- **Reporter**: The component responsible for writing Metrics and Log entries.

---

## Requirements

### Requirement 1: CLI Interface

**User Story:** As a data engineer, I want to invoke the Job via a CLI with explicit path arguments, so that no paths are hardcoded and the Job is portable across environments.

#### Acceptance Criteria

1. THE Job SHALL accept the following CLI arguments: `--input` (path to Dataset CSV), `--config` (path to Config YAML), `--output` (path for Metrics JSON), `--log-file` (path for Log file).
2. IF any required CLI argument is missing, THEN THE Job SHALL exit with a non-zero exit code and write an error Metrics file to the `--output` path if determinable.
3. THE Job SHALL use only the paths provided via CLI arguments and SHALL NOT use any hardcoded file paths.

---

### Requirement 2: Config Loading and Validation

**User Story:** As a data engineer, I want the Job to load and validate a YAML config file, so that all runs are governed by explicit, versioned parameters.

#### Acceptance Criteria

1. WHEN a valid Config file is provided, THE Validator SHALL parse the YAML and extract `seed`, `window`, and `version` fields.
2. THE Validator SHALL confirm that `seed` is an integer, `window` is a positive integer greater than zero, and `version` is a non-empty string.
3. IF the Config file does not exist or cannot be read, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
4. IF the Config file is missing any of the required fields (`seed`, `window`, `version`), THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
5. IF any Config field has an invalid type or value, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
6. WHEN Config is successfully loaded and validated, THE Job SHALL call `numpy.random.seed(seed)` with the integer value of `seed` to ensure deterministic behavior.

---

### Requirement 3: Dataset Loading and Validation

**User Story:** As a data engineer, I want the Job to load and validate the input CSV, so that processing only proceeds on well-formed data.

#### Acceptance Criteria

1. WHEN a valid Dataset file is provided, THE Validator SHALL parse the CSV into a tabular structure with typed columns.
2. THE Validator SHALL confirm that the Dataset contains a `close` column.
3. IF the Dataset file does not exist or cannot be read, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
4. IF the Dataset file is not a valid CSV format, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
5. IF the Dataset file is empty or contains zero data rows, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.
6. IF the Dataset does not contain a `close` column, THEN THE Validator SHALL raise a descriptive error and THE Reporter SHALL write an error Metrics file.

---

### Requirement 4: Rolling Mean Computation

**User Story:** As a data engineer, I want the Processor to compute a rolling mean on the close price, so that a smoothed baseline is available for signal generation.

#### Acceptance Criteria

1. WHEN the Dataset is loaded and validated, THE Processor SHALL compute Rolling_Mean as the rolling arithmetic mean of the `close` column using a window size equal to `window` from Config.
2. THE Processor SHALL produce NaN values for the first `window - 1` rows where a full window is not available.
3. THE Processor SHALL exclude rows where Rolling_Mean is NaN from all subsequent Signal and Signal_Rate computations.

---

### Requirement 5: Signal Generation

**User Story:** As a data engineer, I want the Processor to generate a binary signal per row, so that a tradeable indicator is produced from the price data.

#### Acceptance Criteria

1. WHEN Rolling_Mean is defined for a row, THE Processor SHALL assign Signal = 1 if `close` is strictly greater than Rolling_Mean, else Signal = 0.
2. THE Processor SHALL compute Signal only for rows where Rolling_Mean is not NaN.
3. THE Processor SHALL produce deterministic Signal values for identical inputs and Config parameters.

---

### Requirement 6: Metrics Output

**User Story:** As a data engineer, I want the Reporter to write a machine-readable metrics JSON file, so that job results can be consumed by downstream monitoring systems.

#### Acceptance Criteria

1. WHEN the Job completes successfully, THE Reporter SHALL write a Metrics JSON file to the `--output` path containing: `version`, `rows_processed`, `metric` (value: `"signal_rate"`), `value` (Signal_Rate as a float rounded to 4 decimal places), `latency_ms` (total runtime in milliseconds as an integer), `seed`, and `status` (value: `"success"`).
2. IF the Job encounters any error, THEN THE Reporter SHALL write a Metrics JSON file to the `--output` path containing: `version` (if available, else omitted), `status` (value: `"error"`), and `error_message` (a descriptive string).
3. THE Reporter SHALL write the Metrics file in both success and error cases before the Job exits.
4. WHEN the Job completes successfully, THE Job SHALL print the Metrics JSON to stdout.
5. THE Job SHALL exit with code 0 on success and a non-zero exit code on any error.

---

### Requirement 7: Logging

**User Story:** As a data engineer, I want the Reporter to write a structured log file, so that job execution can be audited and debugged.

#### Acceptance Criteria

1. THE Reporter SHALL write Log entries to the file path specified by `--log-file`.
2. THE Reporter SHALL record the following events in the Log: job start timestamp, Config loaded and validated (including `seed`, `window`, `version` values), number of rows loaded from Dataset, rolling mean computation step, signal generation step, Metrics summary, job end with final status.
3. IF any validation or processing error occurs, THEN THE Reporter SHALL record the exception details and error message in the Log before the Job exits.
4. EACH Log entry SHALL include a timestamp and severity level.

---

### Requirement 8: Reproducibility

**User Story:** As a data engineer, I want identical inputs and config to always produce identical outputs, so that results can be audited and reproduced.

#### Acceptance Criteria

1. THE Job SHALL produce identical Metrics output for identical Dataset, Config, and CLI arguments across multiple runs.
2. THE Job SHALL set the random seed via `numpy.random.seed(seed)` before any processing begins, using the `seed` value from Config.
3. THE Processor SHALL use only deterministic operations (rolling mean, comparison) that do not depend on random state beyond seed initialization.

---

### Requirement 9: Docker Deployment

**User Story:** As a DevOps engineer, I want the Job to run inside a Docker container with a single command, so that the environment is fully reproducible and portable.

#### Acceptance Criteria

1. THE Job SHALL be packaged in a Dockerfile using `python:3.9-slim` as the base image.
2. THE Dockerfile SHALL copy `data.csv`, `config.yaml`, `run.py`, and `requirements.txt` into the container image.
3. WHEN the container is run with `docker run --rm mlops-task`, THE Job SHALL execute with default paths for input, config, output, and log file without requiring additional arguments.
4. WHEN the container run completes successfully, THE Job SHALL produce `metrics.json` and `run.log` inside the container and print the Metrics JSON to stdout.
5. THE container SHALL exit with code 0 on success and a non-zero exit code on failure.
6. THE Dockerfile SHALL install all Python dependencies from `requirements.txt`.

---

### Requirement 10: Documentation

**User Story:** As a developer, I want a README that documents how to run the Job locally and via Docker, so that the project is immediately usable without prior knowledge.

#### Acceptance Criteria

1. THE README SHALL include instructions for running the Job locally using the CLI.
2. THE README SHALL include the Docker build command (`docker build -t mlops-task .`) and Docker run command (`docker run --rm mlops-task`).
3. THE README SHALL include an example `metrics.json` output.
4. THE README SHALL document all required CLI arguments and their descriptions.
