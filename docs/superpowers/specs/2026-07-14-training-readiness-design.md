# Training Readiness Design

## Goal

Make BitGuard-BNN safe to use for real, repeatable training runs without changing its public CLI or adding dependencies.

## Scope

The change covers the defects found in the repository audit:

1. reject split settings that are silently ignored and provide separate attack-held-out configs;
2. calibrate fixed-FPR operating points on validation data, never test labels;
3. make CUDA determinism explicit and record the complete dependency environment;
4. reduce peak host memory and remove repeated CSV concatenation;
5. add durable epoch checkpoints, resume support, progress records, modern AMP, and persistent workers;
6. replace per-packet full-window feature recomputation with incremental window statistics;
7. expose and export the cost-aware gate's physically prunable classifier inputs;
8. remove duplicate classical experiment combinations and write multi-seed summaries;
9. add integration and regression tests for the complete neural training/export path.

## Design

### Experiment validity

`validate_config` will reject non-empty `held_out_attacks` unless `split.strategy` is `attack`, and reject non-empty `held_out_devices` unless the strategy is `device`. Existing device/time configs will remove ignored keys; dedicated attack configs will carry the held-out lists. Fixed-FPR thresholds will be produced from validation labels and probabilities and passed into test metric calculation as immutable calibration data.

### Reproducible and recoverable training

Seeding will set the cuBLAS workspace before importing/using CUDA, enforce deterministic algorithms, and disable cuDNN benchmarking. The environment manifest will include core dependency versions and deterministic settings. Neural fitting will use `torch.amp.GradScaler`, optional AMP, worker persistence, atomic epoch checkpoints, and resume state containing model, optimizer, scheduler, scaler, history, best state, and early-stopping counters.

### Bounded resource use

CSV chunks will be collected once rather than repeatedly concatenated. The source dataset reference will be released after splitting, and training will release obsolete arrays/dataframes between phases. Sampled configurations remain development-only; manifests continue to withhold natural-distribution and temporal claims. Full-data runs will fail early when an optional configured row limit is exceeded instead of reaching an uncontrolled OOM.

### Streaming features

Each device window will maintain counters, sums, destination/port cardinalities, per-second bins, and interval moments. Appending and expiring an event will update those aggregates. Feature production will therefore be proportional to the fixed feature count rather than rebuilding arrays and counters from every event in the window.

### Cost-aware export

Learned hard gate selections will be included in model summaries. The edge exporter will prune inactive first-layer classifier columns and expose separate classifier-active and open-set feature requirements. Open-set acquisition requirements remain explicit when open-set detection is enabled.

### Experiment aggregation

Classical models will run once per encoder/seed because neural loss selections do not affect their fit. The matrix script will write raw runs plus a grouped multi-seed summary containing count, mean, sample standard deviation, and a documented normal-approximation 95% interval.

## Error handling

Invalid protocol combinations fail during config loading. Resume checkpoints are validated against model structure. Atomic checkpoint replacement prevents partial files from being treated as resumable. Streaming input order and capacity checks remain strict.

## Verification

Every behavior change starts with a failing `unittest`. Completion requires the complete unit suite, a CPU neural train/export integration test, compile checks, and a fresh CUDA mini-smoke when CUDA is available.

## Non-goals

This change does not add a Raspberry Pi runtime, download licensed datasets, claim target-device energy results, or add third-party data frameworks.
