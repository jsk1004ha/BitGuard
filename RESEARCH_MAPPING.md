# Research-plan mapping

This file maps the formal BitGuard-BNN plan to executable code and states what
each result can and cannot support.

| Research question | Implementation | Primary output | Validity condition |
|---|---|---|---|
| RQ1: feature extraction bottleneck | `stream-features`, `MicroSecurityFeatureProcessor.latency_summary`, PyTorch batch-1 benchmark | p50/p95 feature update and model latency | Measure again on the target Raspberry Pi; Python/PyTorch is not an XNOR runtime |
| RQ2: 115/64/32/16/8 features | `scripts/run_feature_ablation.py`, train-only selector and cost proxy | per-run `metrics.json`, `feature_manifest.json` | Native N-BaIoT features evaluate model budgets, not raw acquisition energy |
| RQ3: cascade savings | validation-constrained Boolean/Tiny/Main cascade | `cascade_calibration.json`, `boolean_fast_path.json`, cascade section of `metrics.json` | Operation savings are dense-equivalent estimates until a target runtime is benchmarked |
| RQ4: temporal low-rate behavior | five 4-bit counters and capture-aware replay | `temporal_predictions.csv` or `.parquet`, `operational_metrics.json` | Requires a continuous time/sequence/device test with no pre-split row cap |
| RQ5: defense recommendation | monotonic Level 0-5 action simulator | benign disruption, action precision, episode misses, time to mitigation | Recommendations only; traffic reduction values are sensitivity assumptions |

## Dataset tracks

### Track A: N-BaIoT native 115-feature model study

Use for FP32/BNN performance, feature budgets, device-held-out,
attack-held-out/open-set, and cascade classification. It has no verified
wall-clock timestamp and no labelled beacon/exfil class. Do not report
device-hour, seconds-to-detection, beacon recall, or exfil recall from this
track.

### Track B: common streaming metadata schema

Generate the same 24 features from ordered packet/flow metadata with
`stream-features`. Use `configs/common_stream_full.yaml` for the complete
cascade + state + action path. Only this track can support feature-update cost,
wall-clock delay, and a schema-controlled cross-dataset experiment.

## Full-profile evidence semantics

### Exact row coverage and optimization

The full profiles do not sample accepted rows away from splitting, optimization,
or evaluation to fit memory. Source normalization assigns a globally unique row
ID, and the split planner publishes disk-backed membership plus an exact count
for train, validation, and test. Shard verification checks that every accepted
row appears in exactly one partition, partitions are disjoint, row IDs remain
unique, row counts agree, and file/schema hashes match. A malformed or
unassigned row fails preparation instead of being silently dropped. The
streaming trainer's deterministic epoch cursor consumes every train row exactly
once per epoch, including after a compatible mid-epoch resume.

This exactness applies to row counts, split membership, ANOVA sufficient
statistics over the frozen imputed values, model optimization, and full-test
evaluation. It does not make every preprocessing statistic exact.

### Train-only approximate quantiles

Imputation medians, robust-scaler quantiles, quantile encoder thresholds, and
the benign center/distance threshold are fitted only from the train partition.
They use a deterministic BLAKE2b bottom-k sample of complete rows keyed by the
configured seed and global row ID. The full configs retain at most 200,000 rows;
the result is stable across chunk and merge order for the same source, seed, and
capacity.

`feature_manifest.json` records algorithm/version, seed, capacity, retained and
total counts, exact versus approximate fields, and a 95% Dvoretzky-Kiefer-
Wolfowitz CDF supremum bound per finite sampled column. That bound is
probabilistic and distributional. No deterministic quantile value-error bound
is claimed. ANOVA is deterministic after imputation but inherits the median
approximation when missing values are filled.

### Exact numeric metrics, sampled plots

`metrics.json` is computed over every test row. Confusion-derived accuracy,
balanced accuracy, Macro-F1, per-class measures, MCC, high-risk false-negative
rate, unknown-like recall, Brier score, 10-bin ECE, and fixed-FPR observations
are accumulated without a plot sample. Per-class AUROC and AUPRC use bounded
external sorted runs and are exact for the persisted scores, including ties.

Only visualization input is sampled. `plot_sample.parquet` retains up to the
configured limit (50,000 in the full defaults) by the smallest deterministic
SHA-256 priorities of `seed` and `row_uid`. `plot_manifest.json` records both
`numeric_metrics_scope: full_test` and
`plot_rows_scope: deterministic_sample`. Never derive or describe numeric model
performance from the plot sample.

### Leakage and fixed-FPR contract

Split membership is frozen before preprocessing. Imputation, scaling, feature
selection, encoder thresholds, benign reference statistics, open-set
calibration inputs, and training loss never use test rows. Open-set, cascade,
Boolean fast-path, and fixed-FPR thresholds are calibrated from the complete
validation partition. Test labels only measure the already frozen operating
points; they do not tune them. The prediction Parquet metadata binds the exact
test inference contract used to produce `metrics.json`.

### Provenance chain

Keep the complete fingerprint chain when publishing a result:

| Boundary | Evidence |
|---|---|
| Source | immutable source manifest, official locator/license metadata, file sizes and SHA-256, normalized-source fingerprint |
| Environment | bootstrap and run environment reports, Python/package/Torch versions, selected CPU/CUDA profile, deterministic-runtime settings |
| Split | strategy/config, membership Parquet and manifest hashes, semantic fingerprint, exact partition counts |
| Preprocess | `preprocessor.joblib`, `feature_manifest.json`, train-only sketch provenance, selected/materialized feature contract |
| Shards | `shard_manifest.json`, per-file hashes/schema/row counts, source/split/preprocessing fingerprints |
| Checkpoints | prepared-descriptor and split signatures, model/training signature, epoch/optimizer cursor, checkpoint SHA-256 |
| Inference | calibrated config, Main/Tiny checkpoint hashes, preprocessing/open-set/cascade/Boolean/fixed-FPR/routing settings, prediction temporal flags, `inference_contract.json` fingerprint |

This chain supports reproduction when inputs match and an explicit rejection
when a source, split, preprocessor, checkpoint, or inference setting changes.

## Experiment commands

```bash
# Sanity check
bitguard make-demo --output data/demo.csv --rows 12000 --seed 2309
bitguard train --config configs/demo.yaml

# N-BaIoT core run
bitguard train --config configs/nbaiot.yaml

# Feature-budget ablation
python scripts/run_feature_ablation.py \
  --config configs/nbaiot.yaml --budgets 115 64 32 16 8

# Ordered metadata to common schema, then full system
bitguard stream-features \
  --input data/packet_metadata.csv \
  --output data/common_stream_features.csv
bitguard train --config configs/common_stream_full.yaml
```

## Required final-study practice

- Use at least three development seeds and five final seeds.
- Keep evaluation class priors natural; row caps are development-only.
- Use device or capture blocks for confidence intervals, never independent-row
  bootstrap on adjacent traffic windows.
- Report attack-episode misses alongside delay percentiles.
- Treat action traffic-reduction estimates as sensitivity analysis.
- Claim edge memory/latency/energy only after measuring the packed runtime on
  the actual target.

## Deliberate limits of version 0.1

- PyTorch BNN layers validate training accuracy but still execute floating
  tensor kernels.
- The Main BNN has a verified packed export description; the bundled Tiny
  checkpoint still needs a target-specific packed runtime.
- INT8 QAT, constrained feature-space attacks, and Raspberry Pi energy
  instrumentation are follow-up modules, not silently simulated results.

