# Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real BitGuard-BNN training valid, reproducible, recoverable, scalable, and regression-tested.

**Architecture:** Preserve the CLI and configuration-driven architecture. Add small validation/calibration/checkpoint helpers at existing module boundaries, make streaming state incremental, and keep all new behavior dependency-free.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, PyTorch, unittest.

---

### Task 1: Experiment protocol and fixed-FPR calibration

**Files:**
- Modify: `src/bitguard_bnn/config.py`
- Modify: `src/bitguard_bnn/metrics.py`
- Modify: `src/bitguard_bnn/trainer.py`
- Modify: `configs/nbaiot.yaml`
- Modify: `configs/botiot.yaml`
- Create: `configs/nbaiot_attack.yaml`
- Create: `configs/botiot_attack.yaml`
- Create: `tests/test_config_and_metrics.py`

- [x] Add tests that rejected split options fail and fixed-FPR thresholds are derived from validation only.
- [x] Run `PYTHONPATH=src python -m unittest tests.test_config_and_metrics -v` and confirm the new tests fail for the intended missing behavior.
- [x] Add `calibrate_fixed_fpr_thresholds(...)` and pass its result from validation to `classification_metrics(...)`.
- [x] Add split-option validation and separate protocol configs.
- [x] Re-run the focused tests and confirm they pass.

### Task 2: Deterministic, resumable, efficient neural fitting

**Files:**
- Modify: `src/bitguard_bnn/config.py`
- Modify: `src/bitguard_bnn/trainer.py`
- Create: `tests/test_training_runtime.py`

- [x] Add tests for cuBLAS configuration, environment version capture, atomic epoch checkpoint content, and resume epoch continuity.
- [x] Run `PYTHONPATH=src python -m unittest tests.test_training_runtime -v` and confirm failure for the missing behavior.
- [x] Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Torch use and record core package versions.
- [x] Replace deprecated scaling with `torch.amp.GradScaler`, enable persistent workers when workers are configured, and add atomic save/resume state.
- [x] Re-run the focused tests and confirm they pass without warnings.

### Task 3: Memory and streaming scalability

**Files:**
- Modify: `src/bitguard_bnn/data.py`
- Modify: `src/bitguard_bnn/trainer.py`
- Modify: `src/bitguard_bnn/streaming.py`
- Modify: `tests/test_data_and_preprocess.py`
- Modify: `tests/test_state_streaming_cascade.py`

- [x] Add regression tests for single-concat chunk loading, configured row-limit failure, window expiry, and aggregate-feature parity.
- [x] Run the two focused test modules and confirm new tests fail.
- [x] Replace repeated CSV concatenation with one final concat, add an optional early row limit, release the loaded source after splitting, and clear obsolete arrays between phases.
- [x] Implement incremental device-window aggregates with explicit append/expire operations.
- [x] Re-run focused tests and compare a 1,000-event benchmark against the recorded pre-change 2,048-window throughput of 322 rows/s.

### Task 4: Cost-aware export and experiment aggregation

**Files:**
- Modify: `src/bitguard_bnn/models.py`
- Modify: `src/bitguard_bnn/export.py`
- Modify: `src/bitguard_bnn/trainer.py`
- Modify: `scripts/run_experiment_matrix.py`
- Create: `tests/test_export_and_matrix.py`

- [x] Add tests that gate-selected encoded columns are pruned with logit parity and classical combinations ignore neural loss variants.
- [x] Run the focused export/matrix tests and confirm failure.
- [x] Add gate selection summaries and prune first-layer export inputs while retaining explicit open-set feature requirements.
- [x] Add combination generation and multi-seed summary helpers; write raw and summary CSV files.
- [x] Re-run focused tests and confirm parity and aggregation pass.

### Task 5: Full neural integration and documentation

**Files:**
- Create: `tests/test_training_integration.py`
- Modify: `README.md`
- Modify: `VALIDATION.md`

- [x] Add a CPU one-epoch synthetic BNN run that verifies metrics, checkpoint, resume metadata, and packed export parity.
- [x] Run the integration test and resolve any cross-module failures.
- [x] Document development caps versus final full-data runs, protocol-specific configs, AMP validation, resume usage, and output summary files.
- [x] Run `PYTHONPATH=src python -m unittest discover -s tests -v`.
- [x] Run `python -m compileall -q src scripts tests` and a fresh CUDA mini-smoke when CUDA is available.

### Task 6: Final review

**Files:**
- Review all modified production, config, test, and documentation files.

- [x] Compare every design requirement against the diff and test evidence.
- [x] Attempt static analysis with `python -m ruff check --no-cache src tests scripts`; Ruff was unavailable in this environment.
- [x] Confirm `git diff --check` is clean and report any unavailable verification explicitly.
