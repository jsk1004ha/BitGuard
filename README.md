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
- validation-calibrated fixed-FPR metrics with no test-label threshold tuning;
- atomic epoch checkpoints, resumable neural training, and partial progress CSVs;
- JSON/CSV manifests, checkpoints, predictions, and plots;
- an end-to-end synthetic demo and unit tests.

The action simulator only emits recommendations. It does not send packets,
change firewall rules, or automatically block a device.

## 1. Installation

Python 3.10, 3.11, or 3.12 is required for the supported workflow. The
bootstrap rejects Python 3.13. CUDA is optional.

For lightweight development without dataset bootstrap:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a manually managed CUDA development environment, install the PyTorch build
matching the local driver first, then run `python -m pip install -e .`. Project
metadata does not force a CUDA wheel. This manual path does not provide the
locked dependency, compute verification, disk preflight, acquisition, or resume
contracts below. Use the platform wrapper for full-dataset work; it installs and
verifies the selected locked CPU/CUDA profile itself.

## Full-dataset setup, preparation, and training

The platform wrapper creates a locked `.venv`, installs the selected official
PyTorch profile and project dependencies, checks compute and disk, acquires and
extracts sources, prepares verified Parquet shards, trains N-BaIoT and then
BoT-IoT, evaluates the complete test partitions, and exports the edge artifacts.

### Prerequisites

- Use Python 3.10, 3.11, or 3.12. The wrapper rejects other versions.
- Review the academic-use terms on the
  [official UNSW BoT-IoT project page](https://research.unsw.edu.au/projects/bot-iot-dataset).
  Use `--accept-botiot-academic-license` only when those terms apply. The
  bootstrap can then fetch the pinned full-CSV archive automatically.
- ZIP files use Python's built-in extractor. A RAR source, including a RAR
  nested inside a ZIP, requires a `7z`, `7zz`, or `7za` executable. Without
  `--install-system-tools`, BitGuard prints the exact `winget`, `apt-get`, or
  `dnf` command and makes no operating-system change.
- Put `--data-root` and `--runs-root` on filesystems with enough free space.
  The bootstrap performs a conservative per-filesystem disk preflight before
  dataset acquisition or preparation.

From a fresh checkout, these are complete one-command CPU runs for both
datasets. They prepare and train; they do not stop at preparation.

Windows PowerShell:

```powershell
.\bootstrap.ps1 --full --compute cpu --accept-botiot-academic-license --data-root "$HOME\BitGuardData" --runs-root "$HOME\BitGuardRuns"
```

Linux shell:

```bash
./bootstrap.sh --full --compute cpu --accept-botiot-academic-license --data-root "$HOME/bitguard-data" --runs-root "$HOME/bitguard-runs"
```

For an RTX 5090, select the locked CUDA 12.8 profile:

```powershell
.\bootstrap.ps1 --full --compute cu128 `
  --accept-botiot-academic-license `
  --data-root "$HOME\BitGuardData" `
  --runs-root "$HOME\BitGuardRuns"
```

CPU training is bounded-memory but not quick: the uncapped profiles default to
30 epochs and can run for many hours or days. The CPU profile shown above is
also the target profile when CUDA is unsuitable; when reusing a failed data
root, add the stage-specific recovery option described below.

`--compute auto` selects CPU only when `nvidia-smi` is absent. A detected NVIDIA
driver maps to `cu118`, `cu124`, or `cu128`; a broken probe, an older CUDA
capability, a wheel mismatch, or a failed GPU allocation stops the run instead
of silently downgrading it. Explicit CUDA selections require the matching
locked PyTorch wheel and a working CUDA device. The `cu128` profile pins
`torch==2.11.0` for RTX 50-series support. If CUDA validation fails before
training starts, recover with the original command plus
`--compute cpu --restart-stage environment`. If CUDA training has already
created a checkpoint, use `--compute cpu --restart-stage train`: CUDA optimizer
state is not retried on CPU, so this deliberately starts fresh training and
forfeits optimizer/cache resume for that failed training stage.
Changing only `--compute cpu` while retaining that failed CUDA checkpoint is
rejected before a new training run is created; the explicit train restart is
required to acknowledge the loss of resumable optimizer state.

### Dataset scope and license boundary

N-BaIoT is downloaded automatically only from the [official UCI dataset
record](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot).
For BoT-IoT, UNSW remains the dataset and license authority. Automatic
transport uses version 1 of the public Kaggle mirror
[`vigneshvenkateswaran/bot-iot`](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot),
not Kaggle as a license authority. The registry pins the 1,257,092,644-byte
archive and SHA-256
`7869754e4b6192b45d4497be94cc34d621e1db81b6f76189e72ec4077e85bd75`;
the verified archive expands to about 15.0 GB. Download publication fails
closed if either pin changes. The downloader needs no Kaggle SDK, Kaggle
account, Microsoft account, or stored cookie, and it never persists the
temporary signed storage redirect.

`--accept-botiot-academic-license` records that you reviewed the UNSW
academic-use terms; it does not grant or replace a license. If the pinned
public mirror is unavailable, provide an already downloaded local directory,
ZIP, or RAR. The local path takes precedence and prevents BoT-IoT network
access:

```powershell
.\bootstrap.ps1 --full --compute cu128 `
  --botiot-source "$HOME\Datasets\BoT-IoT" `
  --accept-botiot-academic-license `
  --data-root "$HOME\BitGuardData" `
  --runs-root "$HOME\BitGuardRuns"
```

The full profiles consume complete model-ready CSV flow records. They do not
download the 69.3 GB BoT-IoT PCAP capture and do not implement PCAP-to-flow
conversion. `configs/full/nbaiot.yaml` and `configs/full/botiot.yaml` disable
row caps and use verified Parquet shards. Every accepted source row contributes
to exact split coverage; every training row is consumed once per epoch; and
numeric evaluation covers the complete test split. Quantile-derived
preprocessing values remain deterministic bounded-sample approximations, with
their capacity and statistical error contract recorded in
`feature_manifest.json`.

### Resume, restart, and prepare-only

A normal rerun reads `bootstrap-state.json`, fingerprints committed outputs,
and reuses only verified stages and completed dataset runs. The `validate`
stage still rechecks external shard bytes, schemas, split membership,
preprocessing artifacts, and coverage. Inspect the failed report and its
`recovery_command` before using `--restart-stage NAME`: that option deliberately
invalidates the named stage and every later stage; it is not the normal resume
switch. After an ordinary interruption or train-stage error, correct the cause
and rerun the exact same command; the bootstrap automatically reopens the
verified optimizer checkpoint and validation cache. Only a corrupt or
incompatible checkpoint, or an intentional CUDA-to-CPU switch after CUDA
training began, requires `--restart-stage train` and loses that automatic
training resume. Model-level optimizer resume is separate and uses the
compatible `training.resume_from` checkpoint described in section 12.

Distillation checkpoints have an additional identity constraint: the bootstrap
does not currently persist an independently verifiable teacher artifact. A
failed distillation run therefore cannot be resumed safely from its optimizer
checkpoint; rerun with `--restart-stage train` to rebuild the teacher and start
that training stage from scratch. The rejection happens before a new run
directory is allocated, so an unverified teacher fingerprint is never trusted
from the checkpoint itself.

To acquire and prepare both datasets without training, add `--prepare-only`:

```powershell
.\bootstrap.ps1 --full --compute cpu --prepare-only --accept-botiot-academic-license --data-root "$HOME\BitGuardData" --runs-root "$HOME\BitGuardRuns"
```

```bash
./bootstrap.sh --full --compute cpu --prepare-only --accept-botiot-academic-license --data-root "$HOME/bitguard-data" --runs-root "$HOME/bitguard-runs"
```

Successful preparation reports `status: "prepared"` and `next_stage: "train"`
only after exact accepted-row coverage is verified.

### Reports and artifacts

The requested data root contains the portable bootstrap control plane:

```text
<data-root>/.bitguard/
  bootstrap-report.json       # final status, recovery command, and locators
  bootstrap-state.json        # resumable stage fingerprints
  preflight.json
  environment.json
  acquisition.json
  extraction.json
  preparation.json
  training.json               # per-dataset status and verified run locator
  summary.json
  manifests/                  # immutable source manifests
  schema/                     # per-dataset schema reports
  prepared/                   # prepared-dataset control descriptors
<data-root>/prepared/{nbaiot,botiot}/
  resolved_config.yaml
  split/
    split-membership-*.parquet
    split-membership-*.manifest.json
  preprocessor.joblib
  feature_manifest.json
  shard_manifest.json
  dataset=*/split=*/label=*/part-*.parquet
```

The report's `report_path` is authoritative if a lock or path failure forces a
deterministic sibling fallback. Its `reports`, `prepared_datasets`,
`dataset_statuses`, and `trained_runs` mappings locate the verified outputs.
Each successful run under `<runs-root>/nbaiot_full/<timestamp>` or
`<runs-root>/botiot_full/<timestamp>` includes `run_summary.json`, resolved and
calibrated configs, environment and prepared-dataset manifests, train state and
best checkpoints, calibration files, `inference_contract.json`, exact
`metrics.json`, compressed `predictions.parquet`, deterministic plot sample and
manifest, phase resource measurements, and `edge/bitguard_edge_*` export files.

The filesystem safety boundary covers untrusted network/archive content and
cooperative BitGuard writers inside a trusted workspace. Malicious same-account
mutation of parent directory entries or hardlinks is outside that contract.
Retained `.bitguard-retired-*` and `.bitguard-extract-*` artifacts are reported
with apparent and inode-deduplicated byte counts and are never deleted
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
- `last_training_state.pt`, `training_history.partial.csv`,
  `training_history.csv` for neural runs;
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

That config is the device-held-out protocol. Use
`configs/nbaiot_attack.yaml` for the separate Mirai-scan-held-out protocol.

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

The default is a chronological split. Use `configs/botiot_attack.yaml` for the
keylogging/exfiltration-held-out open-set protocol. A non-empty
`held_out_attacks` or `held_out_devices` entry with the wrong split strategy is
rejected instead of being silently ignored.

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
`max_loaded_rows` is a separate safety ceiling: exceeding it raises an early
error instead of letting a multi-file load grow until the host runs out of
memory. It is not a sampler. Increase it only after checking available RAM.

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

The command writes the raw run CSV plus `<output-stem>_summary.csv` with run
count, mean, sample standard deviation, and a normal-approximation 95% interval.
Classical models run once per encoder and seed; neural loss choices no longer
duplicate identical classical fits.

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

For a cost-aware BNN, inactive gated columns are removed from the exported
first layer. The manifest separately lists classifier-active inputs and the
full raw-feature set still required by the open-set distance detector. Do not
claim acquisition savings from the classifier list while open-set detection is
enabled unless the open-set path is also redesigned and revalidated.

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
The environment manifest records NumPy, pandas, scikit-learn, joblib, PyYAML,
PyTorch, CUDA, and deterministic-runtime settings.

The supplied neural configs enable AMP when CUDA is selected; CPU runs ignore
that flag. Training uses persistent data-loader workers when `num_workers > 0`.
To resume an interrupted main-model fit, point a new config at the prior atomic
state file and keep the model/preprocessing structure unchanged:

```yaml
training:
  epochs: 30
  checkpoint_every_epochs: 1
  resume_from: runs/nbaiot/<timestamp>/last_training_state.pt
```

`epochs` is the total target epoch, not an additional epoch count. Resuming
continues the optimizer, scheduler, AMP scaler, shuffle generator, best model,
history, RNG state, and early-stopping counters. The checkpoint also records a
training signature and rejects changes to the model, loss, optimizer settings,
data split, selected features, label order, or preprocessing contract.

## 13. Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The unit tests do not download either research dataset.
