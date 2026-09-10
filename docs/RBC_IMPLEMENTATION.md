# BitGuard RBC implementation

The reference is `BitGuard_RBC_연구계획서_20260908.docx` (2026-09-08).
Existing working-tree edits are preserved. Research outcomes require real experiments.

Acceptance checklist:

- Four disjoint train/selection/calibration/test partitions, held-out devices or time.
- Fixed comparison budget; train-only candidate fitting and internal collision selection.
- Detection-first shared BNN; optional teacher and group protection; frozen detector during type training.
- Packed CPU scoring with output quantization before independent threshold calibration.
- Strict `score > threshold`, fixed type label set, misses retained, group supports.
- Natural-distribution bounded streaming mixing and normalized weighted focal loss.
- Repeatable CLI run, artifacts, fingerprints, regression and end-to-end tests.

Optional Hamming coding, exact early exit, and temporal aggregation are separate experiments.
No automatic blocking is performed. Quantization is not assumed to preserve floating-point
output decisions; the deployed integer score must be calibrated anew.

The RBC runner uses the existing normalized CSV source adapters. Its explicit
`dataset.max_loaded_rows` is a hard host-memory admission limit, not a sample target.
It does not silently downsample full datasets. The existing Parquet trainer remains
available for the normalized legacy baseline; full out-of-core RBC training must be
validated separately before claiming a full-data result.
The natural-distribution Parquet mixer now uses algorithm v3. A v2 cursor/checkpoint
cannot be resumed under v3; keep the original checkout/runtime for historical B0
reproduction. v3 resumes an exact suffix by regenerating/skipping the epoch prefix,
which can increase restart I/O. Mixing uses one shared row budget rather than one
full buffer per class.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m bitguard_bnn train-rbc --config configs/rbc/nbaiot.yaml
.\.venv\Scripts\python.exe -m bitguard_bnn predict-rbc --run runs/nbaiot_rbc --input features.csv --output predictions.csv
```

Use `configs/rbc/botiot.yaml` for time-ordered BoT-IoT. Set real source paths and
an explicit host-memory row limit; these examples deliberately stop above one million
source rows. Choose a fresh `rbc.output_dir` for every run/seed. No existing run is overwritten.
Candidate fitting uses at most `rbc.encoder_fit_rows` training rows and records that
budget; detector training consumes every admitted training row once per epoch.
When comparisons change, an equally configured quantile-encoder student is trained
and compared on selection-only TPR at the target FPR. An inferior candidate is rejected;
the extra training run and both selection scores are recorded. This is a selection
guard, not proof of test-set superiority or quantized decision equivalence.
Feature cost caps currently use unit costs unless supplied to the encoder API; they
are proxies, not measured feature-state bytes or energy.

Artifacts include source provenance, partition hashes/supports, encoder replacement
history, model tensors, training diagnostics, packed integer edge data, independent
calibration, held-out predictions/metrics, and artifact hashes. The encoder JSON carries
raw thresholds and missing-value replacements. Test rows from a held-out raw subtype
are explicitly marked unknown even if their coarse behavior class is known.

For detection-only/quantile ablation, set `max_swaps: 0`, `distillation_alpha: 0`,
`worst_group_blend: 0`, and `type_epochs: 0`; the untrained type head must not be
interpreted as a trained type classifier. Enable each factor separately for P1/P2/P3.
The legacy `train` command remains the existing baseline path; its original metrics
are not automatically interchangeable with the four-partition RBC evaluator.

Not established by the implementation tests: a matched B1/P3 performance improvement,
full-data out-of-core RBC operation, actual peak RAM/feature state, complete executable
size, hardware energy, ARM/MCU behavior, confidence intervals across devices, or
the non-inferiority claim. The report explicitly withholds research success.
NP calibration is opt-in (`np_order_statistic`) and requires an independently justified
IID assumption; the default empirical method makes no population FPR guarantee.

Validation on 2026-09-09 (Python 3.11.14, torch 2.13.0+cpu, pandas 3.0.3):

- RBC tests: 32 passed. Final lazy-import changes: six export/inference/pipeline tests passed.
- Dataset mixer: 14 passed; out-of-core training: 20 passed; run/recovery: 72 passed with one skip.
- Existing gate/export/deployment tests: 25 passed with 29 subtests.
- Ruff on changed code/tests, compileall, and diff whitespace checks passed.
- An isolated `rbc_predict` import loads neither torch nor sklearn.
- Static type checking was not run because mypy is not installed.
- Full-suite execution was stopped in the lengthy shard crash-recovery tests;
  no full-suite pass is claimed. It also showed failures before that point.
- Separately, `test_bootstrap_inspect.py` has five failures, 37 passes, one skip:
  all-NA dtype witnesses, missing boolean timestamps, and three numeric-conversion
  expectations. These files were already modified before this task and were not
  changed by this implementation. Some assertions directly assume pandas behavior
  that differs in the installed 3.0.3 environment.
- The configured N-BaIoT and BoT-IoT data directories are absent. Integration uses
  generated test data and does not establish research performance or deployment budgets.
