# Research-plan mapping

This file maps the formal BitGuard-BNN plan to executable code and states what
each result can and cannot support.

| Research question | Implementation | Primary output | Validity condition |
|---|---|---|---|
| RQ1: feature extraction bottleneck | `stream-features`, `MicroSecurityFeatureProcessor.latency_summary`, PyTorch batch-1 benchmark | p50/p95 feature update and model latency | Measure again on the target Raspberry Pi; Python/PyTorch is not an XNOR runtime |
| RQ2: 115/64/32/16/8 features | `scripts/run_feature_ablation.py`, train-only selector and cost proxy | per-run `metrics.json`, `feature_manifest.json` | Native N-BaIoT features evaluate model budgets, not raw acquisition energy |
| RQ3: cascade savings | validation-constrained Boolean/Tiny/Main cascade | `cascade_calibration.json`, `boolean_fast_path.json`, cascade section of `metrics.json` | Operation savings are dense-equivalent estimates until a target runtime is benchmarked |
| RQ4: temporal low-rate behavior | five 4-bit counters and capture-aware replay | `temporal_predictions.csv`, `operational_metrics.json` | Requires a continuous time/sequence/device test with no pre-split row cap |
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
- INT8 QAT, constrained feature-space attacks, multi-seed confidence-interval
  aggregation, and Raspberry Pi energy instrumentation are follow-up modules,
  not silently simulated results.

