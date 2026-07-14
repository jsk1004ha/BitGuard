# Full-Dataset Bootstrap and Out-of-Core Training Design

**Status:** Approved in conversation; pending review of this written specification.

**Goal:** From a fresh Windows or Linux checkout, prepare an isolated Python environment, acquire and verify the complete model-ready N-BaIoT and BoT-IoT CSV datasets, convert them to bounded-memory shards, and train both full-data experiments with resumable execution.

## Decisions

- Support Windows and Linux.
- Use every record in the official model-ready CSV distributions. The BoT-IoT PCAP and Argus sources are not downloaded because the trainer consumes flow CSVs; PCAP-to-flow extraction is a separate project.
- Download N-BaIoT automatically from the official UCI distribution.
- Never store UNSW or Microsoft credentials. The user downloads BoT-IoT once and supplies a local archive or directory.
- Use a native Python pipeline rather than Docker so CUDA access, large local disks, and existing checkpoint workflows remain straightforward.
- Add PyArrow for partitioned Parquet storage and streaming record batches.
- Use all partition rows for optimization and evaluation. Development row caps are disabled in the generated full-data configurations.

## User Experience

After cloning the repository, the user runs one operating-system-specific wrapper:

```powershell
.\bootstrap.ps1 --full --botiot-source D:\Downloads\BoT-IoT --accept-botiot-academic-license
```

```bash
./bootstrap.sh --full --botiot-source ~/Downloads/BoT-IoT --accept-botiot-academic-license
```

The wrappers contain no dataset or training logic. They locate Python and invoke one Python bootstrap module with the same arguments. Python 3.10 through 3.12 is a prerequisite; if it is unavailable, the wrapper stops with an operating-system-specific installation command instead of making an administrator-level system change silently.

Optional flags:

- `--compute auto|cpu|cuda`: select the execution backend. `auto` uses a verified NVIDIA CUDA installation when available and otherwise uses CPU.
- `--data-root PATH`: override the default `data/` root.
- `--runs-root PATH`: override the default `runs/` root.
- `--prepare-only`: download, verify, and shard without training.
- `--dataset nbaiot|botiot|all`: restrict preparation and training.
- `--restart-stage NAME`: explicitly invalidate one stage and its dependants. Without this flag, a repeated command resumes or reuses verified work by default.
- `--install-system-tools`: explicitly allow a supported package manager to install an archive extractor when nested RAR files require it. Without this flag, a missing extractor produces a remediation command and no system mutation.

The terminal prints each stage, byte and row progress, the selected compute backend, checkpoint paths, and the final run directories. A machine-readable bootstrap report is always written.

## Components

### 1. Platform wrappers

`bootstrap.ps1` and `bootstrap.sh` validate the Python prerequisite, create or reuse `.venv`, and invoke the Python bootstrap module. They propagate exit codes and do not duplicate environment, download, extraction, or training logic.

### 2. Bootstrap orchestrator

A new `bitguard bootstrap` command owns the stage graph:

`preflight -> environment -> acquire -> extract -> inspect -> shard -> validate -> train -> summarize`

State is persisted after every stage in `data/.bitguard/bootstrap-state.json`. Each completed stage records its input signature and outputs. A stage is reused only when both still match; otherwise the affected stage and its dependants are invalidated.

The environment stage installs from versioned lock profiles rather than resolving an unbounded latest environment. The base profile pins Python packages including PyArrow. CPU and supported NVIDIA profiles pin the matching PyTorch wheel and official package index. `auto` chooses only among repository-declared profiles, then verifies a tensor operation and `torch.cuda.is_available()` when CUDA is selected. The resolved distributions and index/profile identifier are recorded in the bootstrap manifest.

### 3. Dataset registry

A versioned registry describes each official source, license notice, expected archive/file patterns, schemas, label mapping, and extraction rules. N-BaIoT uses the official UCI static distribution and records the UCI DOI and CC BY 4.0 attribution. BoT-IoT records the official UNSW project page, the supplied local source, and the required academic-use acknowledgement.

The registry never contains institutional credentials, transient SharePoint tokens, or an unofficial mirror.

### 4. Safe acquisition and extraction

The N-BaIoT downloader writes to a temporary partial file, supports HTTP range resumption when the server does, validates content length where available, and computes SHA-256 while streaming. Downloaded containers are checked before replacement of the final path.

The BoT-IoT source may be a local archive or extracted directory. Both datasets are checked for expected files and columns. ZIP and RAR extraction rejects absolute paths, parent traversal, links escaping the data root, duplicate destination names, and extraction whose declared size exceeds available disk. Nested N-BaIoT RAR files use an available 7-Zip-compatible extractor; installation is attempted only with `--install-system-tools`.

Raw sources are immutable after verification. Their relative paths, sizes, modification times, and SHA-256 values form the source manifest and later resume signature.

### 5. Resource preflight

Before download or extraction, the orchestrator checks:

- supported OS and Python;
- writable data and run roots;
- free disk for partial download, extracted CSVs, Parquet shards, temporary evaluation arrays, and checkpoints, plus a reserve;
- available RAM against the configured record-batch, shuffle-buffer, and worker counts;
- CUDA visibility from both the driver and installed PyTorch when CUDA is selected;
- required archive extraction capability.

Space is calculated from the supplied archive and inspected files when possible rather than relying only on a fixed estimate. Failure occurs before a large mutation and reports required versus available resources. A detected NVIDIA GPU whose CUDA-enabled PyTorch cannot be verified is an error, not a silent CPU fallback; the user may explicitly rerun with `--compute cpu`.

## Out-of-Core Data Pipeline

### 1. Inspection pass

CSV files are read in bounded chunks. The pass validates schema consistency, normalizes labels and metadata using the existing dataset adapters, counts rows/classes/devices, finds unusable columns, and produces stable row identifiers. It does not retain a complete DataFrame.

N-BaIoT device membership is derived from the source path. BoT-IoT uses the configured address and timestamp fields. Every source row is either assigned to exactly one partition or rejected with a recorded reason; the default policy is to fail on rejected rows rather than silently discard them.

### 2. Deterministic split planning

Splits are fixed before any preprocessing statistic is fitted.

- Device and attack protocols use stable group membership.
- BoT-IoT chronological splitting finds exact rank cutoffs using sorted temporary runs and an external merge, so it does not approximate the existing 70/15/15 time protocol.
- Row UIDs, source hashes, split settings, counts, and overlap checks are written to the split manifest.

No validation or test value contributes to train preprocessing.

### 3. Train-only streaming preprocessing

The preprocessor gains a bounded-memory fit path over train batches:

1. collect finite counts, class counts, sums, and squared sums;
2. maintain deterministic bounded quantile sketches for median imputation, robust scaling, binary encoder thresholds, and benign-distance calibration inputs;
3. compute ANOVA F-scores from mergeable class sufficient statistics;
4. apply cost-aware or configured ranking and freeze the selected feature order;
5. make a second train-only pass when selected-feature statistics depend on the first pass.

Every train row contributes to mergeable exact statistics. Quantile-derived values are approximate by design; sketch algorithm, capacity, seed, estimated error settings, and observed counts are saved in the feature manifest. Given the same source manifest and configuration, the results must be deterministic.

### 4. Partitioned Parquet shards

Normalized rows are transformed and written to immutable Parquet shards partitioned by dataset, split, and label. Shards target a bounded compressed size and contain row UID, required metadata, encoded features, and label. Temporary shard names are atomically renamed only after footer validation.

A shard manifest records row count, class count, schema fingerprint, byte size, SHA-256, min/max ordering metadata, and source coverage. The validation stage proves that source coverage is complete, partitions are disjoint, and totals match the inspection pass before training can start.

PyArrow record-batch iteration is used so shard size does not determine RAM usage.

## Out-of-Core Training and Evaluation

### 1. Training iterator

A Parquet iterable dataset reads only the requested feature columns. Each epoch deterministically permutes shard order and uses a bounded in-memory shuffle buffer within shards. Worker assignment is disjoint, so every train row is consumed once per epoch even with multiple workers.

The existing neural objective, Main/Tiny cascade, AMP, deterministic settings, early stopping, and best-model selection remain unchanged. Classical baselines that cannot learn incrementally fail with a clear full-data compatibility error unless a supported out-of-core implementation exists; they never fall back to a capped sample silently.

### 2. Checkpoint cursor

Full-data checkpoints add epoch, shard permutation, worker-independent shard cursor, batch offset, optimizer step, preprocessing/shard signatures, and accumulated epoch metrics. Checkpoints are written atomically every configured number of optimizer steps as well as at epoch boundaries.

After interruption, the run resumes from the last committed cursor. At worst, one uncommitted batch is repeated. A checkpoint is rejected when source hashes, shard manifest, preprocessing, labels, model, loss, optimizer, or compute precision contract differs.

### 3. Validation and test

Validation inference streams into bounded temporary memory-mapped arrays used for open-set, cascade, and fixed-FPR calibration. The validation cascade uses the same Boolean/Tiny/Main/temporal routing as test.

Test predictions are written incrementally to compressed Parquet. Confusion counts, Brier score, calibration bins, routing counts, and other mergeable metrics are accumulated online. Metrics requiring global ordering use external sorted runs or bounded memory-mapped vectors; they are not computed on a test subsample. Plotting may use a deterministic display sample, but the plot manifest states that fact and all reported numeric metrics use the complete test partition.

## Failure Handling and Idempotency

- Ctrl+C and process termination leave only `.partial` files and the last atomic state/checkpoint.
- A bootstrap lock prevents concurrent writers to the same data root.
- Corrupt downloads, unsafe archives, schema drift, insufficient disk, incomplete source coverage, split leakage, non-finite preprocessing state, or shard hash mismatch stop the pipeline before training.
- Re-running the same command reuses verified downloads and shards and resumes training.
- A changed source archive, configuration, code/data format version, or preprocessing contract invalidates only the necessary downstream stages.
- Logs include actionable recovery commands without exposing credentials or environment secrets.

## Generated Artifacts

The bootstrap root contains:

- source and license manifests;
- extraction and schema reports;
- exact split and shard manifests;
- streaming preprocessing artifacts and quantile-sketch metadata;
- generated uncapped full-data configs;
- stage state and logs;
- links to per-dataset training run directories;
- a final JSON summary containing success/failure, elapsed time, rows processed, environment, and checkpoints.

Original development configs remain capped and unchanged in meaning. Full-data configs live under a separate, explicit profile so a normal development run cannot accidentally start a multi-day job.

## Testing

### Unit tests

- range-resume and content validation with a local HTTP fixture;
- ZIP/RAR path traversal and oversized extraction rejection;
- dataset registry and license acknowledgement;
- stage-state invalidation and lock behavior;
- exact split planning and external chronological merge;
- streaming preprocessing versus in-memory preprocessing within declared quantile tolerance;
- shard coverage, disjointness, and hash verification;
- deterministic shard/worker iteration with every row seen once;
- online metrics versus existing in-memory metrics;
- checkpoint cursor and incompatible-resume rejection.

### Integration tests

- Windows-compatible and Linux-compatible wrapper argument forwarding;
- complete small-fixture bootstrap from acquisition through both dataset runs;
- CPU out-of-core one-epoch training and export parity;
- CUDA AMP one-epoch smoke when CUDA is available;
- forced termination during download, sharding, and training followed by successful resume;
- low-disk and malformed-BoT-IoT preflight failures with no partial final artifacts.

### Acceptance criteria

1. One wrapper command prepares and trains both datasets after a valid local BoT-IoT source is supplied.
2. No stage requires loading a complete CSV, split, or encoded dataset into RAM.
3. All accepted official CSV rows appear exactly once in train, validation, or test, and all train rows are consumed once per epoch.
4. Validation/test leakage checks and fixed-FPR calibration semantics remain intact.
5. Repeated commands are safe, and interrupted commands resume without restarting completed large stages.
6. Source, environment, split, preprocessing, shard, and checkpoint provenance is sufficient to reproduce or reject a run.

## Explicit Non-Goals

- Automatic SharePoint/UNSW login or storage of user credentials.
- Downloading the 69.3 GB BoT-IoT PCAP capture or implementing PCAP-to-flow feature extraction.
- Installing Python or modifying the operating system without explicit user authorization.
- Claiming Raspberry Pi latency, RAM, or energy results from workstation training.
- Silently sampling full-data experiments to make them fit.

## Known Trade-offs

- Full BoT-IoT training remains a long-running job even with bounded memory; automation and resume make it operable but do not remove the compute cost.
- Robust medians and encoder quantiles use deterministic approximate sketches. Their error contract is explicit, while every record is still used for model optimization and full-partition evaluation.
- PyArrow becomes a required dependency for the full-data profile, increasing installation size in exchange for bounded-memory Parquet I/O.
- NVIDIA CUDA and CPU are the initial compute targets. ROCm and other accelerators require a later, separately verified install profile.
