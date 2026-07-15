# BitGuard-BNN

`BitGuard-BNN` is a reproducible training and offline-evaluation project for the
research plan **"Ultra-low-cost Stateful Binarized Neural Security Processor
for IoT Botnet Detection and Defense"**.

The implementation deliberately separates two things that are often mixed in
IDS experiments:

1. adapters for precomputed CSV features such as N-BaIoT and BoT-IoT;
2. a deployable streaming metadata feature processor.

This matters because the 115 N-BaIoT columns and the BoT-IoT columns are not a
shared feature schema. A scientifically valid cross-dataset experiment must
first map both datasets to the same packet/flow metadata features; the code
never silently pads incompatible columns with zero.

## Included

- chunked N-BaIoT, BoT-IoT, generic CSV, and synthetic-data adapters;
- random, device-held-out, attack-held-out, time, and cross-dataset splits;
- train-only imputation, scaling, feature selection, and leakage checks;
- sign and 2-4 bit thermometer/hybrid encoders;
- FP32 MLP, vanilla BNN, and cost-aware gated BNN;
- class-weighted cross entropy and focal loss;
- Tiny-BNN/Main-BNN security-risk-aware cascade with threshold tuning;
- confidence + benign-distance open-set detector for `unknown_like`;
- per-device five-counter 4-bit temporal security state machine;
- conservative Level 0-5 action recommendation simulation;
- detection, open-set, operational, latency, model-size, operation-count, and
  early-exit metrics;
- packed binary weight and folded BatchNorm threshold export;
- JSON/CSV manifests, checkpoints, predictions, and plots;
- an end-to-end synthetic demo and unit tests.

The action simulator only emits recommendations. It does not send packets,
change firewall rules, or automatically block a device.

## 1. Installation

Python 3.10 or newer is recommended. CUDA is optional.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA, install the PyTorch build matching the local CUDA driver first, then
run `python -m pip install -e .`. The project itself does not force a CUDA wheel.

## Full-dataset source bootstrap

On a fresh checkout, the platform wrapper creates the locked `.venv`, verifies
compute and disk capacity, acquires the selected sources, safely extracts them,
and writes resumable state plus a machine-readable report. Python 3.10-3.12 is
required.

Windows PowerShell:

```powershell
.\bootstrap.ps1 --full --prepare-only --botiot-source D:\Downloads\BoT-IoT --accept-botiot-academic-license
```

Linux shell:

```bash
./bootstrap.sh --full --prepare-only --botiot-source ~/Downloads/BoT-IoT --accept-botiot-academic-license
```

N-BaIoT is downloaded automatically only from the [official UCI dataset
record](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot).
BoT-IoT is never downloaded automatically: review the [official UNSW project
page](https://research.unsw.edu.au/projects/bot-iot-dataset), download the
model-ready CSV distribution yourself, and supply its local directory, ZIP, or
RAR. `--accept-botiot-academic-license` records your acknowledgement that you
reviewed the terms; it does not grant or replace a license. Credential-bearing
URLs are neither accepted nor persisted.

PCAP capture acquisition and PCAP-to-flow conversion are excluded because this
trainer consumes model-ready CSV features. Full CSV preparation and training
can take many hours or days and require substantial disk. Before preparation,
the bootstrap sums source snapshots, split-membership SQLite, three-pass audit
SQLite, external merge runs, staged shards, and final shards per filesystem
device. The acquisition preflight also reserves a complete partial N-BaIoT
download and a complete verified archive/snapshot, extraction space, metadata
overhead, and a safety reserve. When remote size is unavailable, it uses a
documented conservative 4 GiB archive estimate.

`--prepare-only` now succeeds with `status: "prepared"` only after every
accepted CSV row appears exactly once in a verified train, validation, or test
Parquet shard. Resume is automatic, but the `validate` stage always rechecks
external shard bytes, schemas, split membership, preprocessing artifacts, and
coverage even when the control descriptor is reusable. Use `--restart-stage
NAME` only after inspecting a reported failure. The final report is
`data/.bitguard/bootstrap-report.json`; its `report_path` is authoritative when
a lock/path failure requires a deterministic sibling fallback. The `reports`
mapping locates preflight, environment, acquisition, extraction, preparation,
and per-dataset schema reports when those artifacts exist.

The ordinary `configs/nbaiot.yaml` and `configs/botiot.yaml` remain capped
development profiles. `configs/full/nbaiot.yaml` and
`configs/full/botiot.yaml` are separate uncapped Parquet profiles: all rows
contribute to exact counts, split coverage, ANOVA sufficient statistics, and
later optimization/evaluation. Median, robust-scaler, encoder, and benign
distance quantiles use a deterministic bounded priority sketch and are marked
approximate with capacity/error provenance in `feature_manifest.json`.

Prepare either dataset independently:

```powershell
.\bootstrap.ps1 --dataset nbaiot --prepare-only
.\bootstrap.ps1 --dataset botiot --prepare-only --botiot-source D:\Downloads\BoT-IoT --accept-botiot-academic-license
```

```bash
./bootstrap.sh --dataset nbaiot --prepare-only
./bootstrap.sh --dataset botiot --prepare-only --botiot-source ~/Downloads/BoT-IoT --accept-botiot-academic-license
```

The filesystem safety boundary covers untrusted network/archive content and
cooperative BitGuard writers inside a trusted workspace. Malicious
same-account mutation of parent directory entries or hardlinks is outside that
contract. Retained `.bitguard-retired-*` and `.bitguard-extract-*` artifacts are
reported with apparent and inode-deduplicated byte counts and are never deleted
automatically; inspect their link counts and active processes before manual
cleanup.

## 2. End-to-end smoke run

This path requires no external dataset and validates preprocessing, training,
open-set prediction, temporal replay, and artifact generation.

```bash
bitguard make-demo --output data/demo.csv --rows 12000 --seed 2309
bitguard train --config configs/demo.yaml
```

The command prints a run directory such as `runs/demo/20260710-150000`. It
contains:

- `resolved_config.yaml`, `calibrated_config.yaml`, `split_manifest.json`,
  `feature_manifest.json`;
- `preprocessor.joblib`, `best_model.pt`, `model_summary.json`;
- `metrics.json`, `predictions.csv`, `confusion_matrix.csv`;
- `temporal_predictions.csv`, `operational_metrics.json`;
- PR/ROC/confusion-matrix plots when Matplotlib is installed.

## 3. N-BaIoT layout

The loader recognizes the original directory-oriented release:

```text
data/N_BaIoT/
  Danmini_Doorbell/
    benign_traffic.csv
    gafgyt_attacks/combo.csv
    gafgyt_attacks/junk.csv
    gafgyt_attacks/scan.csv
    gafgyt_attacks/tcp.csv
    gafgyt_attacks/udp.csv
    mirai_attacks/ack.csv
    mirai_attacks/scan.csv
    mirai_attacks/syn.csv
    mirai_attacks/udp.csv
    mirai_attacks/udpplain.csv
  Ecobee_Thermostat/
  ...
```

Edit `configs/nbaiot.yaml`, especially `dataset.path`, then run:

```bash
bitguard train --config configs/nbaiot.yaml
```

The default behavior mapping is explicit: `scan` maps to `scan_like`; every
other Gafgyt/Mirai attack subtype maps conservatively to `flood_like`.
N-BaIoT does not provide labelled beacon or exfiltration behavior, so the code
does not manufacture those classes. These are behavioral proxies, not
malware-family ground truth. Keep `raw_attack` in every report.
Dataset-specific overrides can be versioned under `dataset.label_map`; every
override is written into the provenance manifest.

## 4. BoT-IoT

Set the glob in `configs/botiot.yaml`. The adapter normalizes common label names
(`category`, `subcategory`, `attack`, `label`) and maps scan, DoS/DDoS, theft,
keylogging, and exfiltration categories to the canonical behavior labels.

```bash
bitguard train --config configs/botiot.yaml
```

## 5. Split protocols

Choose `split.strategy`:

- `random`: stratified baseline only;
- `device`: all rows from configured devices are test-only;
- `attack`: configured raw attacks are absent from train/validation and become
  `unknown_like` in test; normal and known-attack comparison rows are also
  placed in test;
- `time`: stable chronological train/validation/test boundaries;
- `sequence`: source-row order for datasets such as N-BaIoT that lack real
  timestamps (do not report this as wall-clock time);
- `block`: keeps contiguous source-file blocks together, reducing adjacent
  window leakage compared with a row-random split;
- `cross`: train/validation use `dataset.path`, while test uses
  `dataset.cross_path` and must expose the same `dataset.shared_features`.
  It also requires identical `feature_schema_id` and
  `cross_feature_schema_id`, so equal column names cannot conceal different
  units or aggregation windows.

Scaling, imputation, selection, encoder thresholds, benign centroid, and
anomaly threshold are fit only on training data. A split manifest records group
overlap and class counts.

`max_rows_per_file` and `max_rows_per_class` are bounded-memory development
sampling controls. Because they run before the split, they can change the
validation/test class prior. Set both to `null` for final natural-distribution
results, or prepare an external sharded dataset whose evaluation partitions are
kept intact. Any run using these caps records the configuration in its manifest.

## 6. Unknown-like handling

An attack withheld from training is not a supervised sixth class. Treating it
as one would leak test knowledge. The default model learns the five known
classes:

```text
benign, scan_like, flood_like, beacon_like, exfil_like
```

At inference, a sample becomes `unknown_like` only when both conditions hold:

1. the known-class posterior is insufficiently confident;
2. its robust distance from the training benign distribution is high.

The thresholds are stored in the preprocessor artifact and can be tuned on a
separate calibration set. Attack-held-out results therefore measure real
open-set behavior rather than a mislabeled closed-set classifier.

## 7. Feature budgets and cost-aware training

The planned `preprocess.feature_budget` values are 115, 64, 32, 16, 8, and
`null`; any positive budget is accepted for development runs. Ranking is fit
on train only. `cost_aware` ranking maximizes a univariate detection score
per configured feature cost. The gated BNN additionally minimizes the expected
cost of its learned feature gates:

```text
L = L_detection
  + lambda_feature * expected_feature_cost
  + beta_fn * differentiable_false_negative_cost
  + gamma_fp * differentiable_false_positive_cost
  + distillation_loss (optional)
```

Neural early stopping uses a configurable validation composite of Macro-F1,
Macro-AUPRC, and attack recall (`training.selection_weights`), never accuracy
alone.

A fixed model-size term has no gradient, so it is reported as parameter/packed
bytes and compared through width/model ablations rather than added as a
numerically inert constant to the loss.

Costs can be supplied in a two-column CSV (`feature,cost`) or inferred from
feature-name groups. Inferred costs are research defaults, not hardware energy
measurements.

For the expert-set ablation, set `preprocess.selection: expert` and list the
exact ordered columns under `preprocess.expert_features`. A missing name raises
an error instead of silently falling back to a statistical selector.

Run the planned feature-budget ablation:

```bash
python scripts/run_feature_ablation.py \
  --config configs/nbaiot.yaml --budgets 115 64 32 16 8
```

For model/encoder/loss/multi-seed comparisons and one aggregate CSV:

```bash
python scripts/run_experiment_matrix.py \
  --config configs/nbaiot.yaml \
  --models logistic_regression fp32_mlp vanilla_bnn cost_aware_bnn \
  --encoders sign thermometer hybrid \
  --losses weighted_ce focal \
  --seeds 2309 2310 2311
```

## 8. Cascade

Enable `cascade.enabled: true`. Training creates:

- a Tiny BNN binary benign/attack gate on the first `tiny_feature_budget`
  selected features;
- the Main BNN behavior classifier;
- a validation-tuned benign exit threshold.

The threshold optimizer chooses the largest early-exit saving that satisfies
`cascade.min_attack_recall`. Temporal suspicion and optional device criticality
subtract from the exit score, so a recently suspicious device is routed to the
main model even when the tiny gate is confident.

## 9. Temporal replay and action levels

Rows are replayed in timestamp order per capture and device. Five saturating counters
(`scan`, `flood`, `beacon`, `unknown`, `benign`) stay in `[0, 15]`. The offline
action head emits:

| Level | Recommendation |
|---:|---|
| 0 | allow |
| 1 | log only |
| 2 | monitor / alert |
| 3 | rate-limit recommendation |
| 4 | temporary-isolation recommendation |
| 5 | administrator-confirmed quarantine recommendation |

Operational output includes false positives per device-hour, detection delay,
benign disruption, action precision, and alert reduction. For meaningful
device-hour and delay results, the source must include real device IDs and
timestamps and a continuous test episode. Otherwise the code withholds those
metrics and records the reason in the manifest.

## 10. Streaming metadata feature processor

`bitguard_bnn.streaming.MicroSecurityFeatureProcessor` consumes bounded packet
metadata events and emits the common 24-feature schema. It uses no payload.
The processor includes protocol/flag ratios, rates, bounded destination
state, repetition, inter-arrival stability, periodicity, direction, and flow
shape proxies. Use this schema on both capture sources before claiming a true
N-BaIoT-to-BoT-IoT cross-dataset result.

An ordered metadata CSV can be converted directly:

```bash
bitguard stream-features \
  --input data/packet_metadata.csv \
  --output data/common_stream_features.csv \
  --window-seconds 60
```

Required columns are `device_id`, `timestamp`, `length_bytes`, `protocol`,
`destination_ip`, and `destination_port`. Optional fields include `tcp_flags`,
`outbound`, `flow_duration_seconds`, and `connection_failed`.

For a continuous shared-schema experiment with the Boolean fast path, Tiny/Main
cascade, temporal state, and action simulation all enabled, use:

```bash
bitguard train --config configs/common_stream_full.yaml
```

Do not enable row caps for device-hour or wall-clock delay results; the trainer
withholds those metrics whenever it cannot verify a continuous evaluation
partition.

## 11. Edge export

After training:

```bash
bitguard export --run runs/nbaiot/<timestamp> --output export/bitguard_edge
```

The export contains packed positive-weight bitsets, XNOR/popcount dimensions,
and BatchNorm-folded thresholds/directions. The reference exporter describes
the arithmetic per layer: hybrid/continuous first-layer inputs are explicitly
marked as low-bit/FP accumulation rather than XNOR. Main-model folded and final
logit parity are checked. When a cascade exists, its Tiny checkpoint and
calibration are bundled, but the Tiny target runtime still must be packed and
verified separately. Actual Raspberry Pi latency and energy must be measured on
the target build; GPU timing is not evidence of edge cost.

## 12. Reproducibility and suggested research order

1. `random` N-BaIoT FP32 MLP and vanilla BNN;
2. device-held-out and attack-held-out protocols;
3. 115/64/32/16/8 feature ablation;
4. encoder and loss ablation;
5. cascade and threshold constraints;
6. temporal replay on timestamped packet/flow data;
7. shared-schema cross-dataset test;
8. Raspberry Pi feature-update/inference/RAM/energy measurements.

Every experiment should retain the resolved configuration, seed, source file
hashes, split manifest, selected features, thresholds, and raw predictions.

## 13. Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

The unit tests do not download either research dataset.
