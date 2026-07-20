# Out-of-Core Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, calibrate, evaluate, checkpoint, and export BitGuard models over every prepared Parquet row without loading a complete split into RAM.

**Architecture:** A deterministic Parquet iterable yields transformed record batches with disjoint worker ownership and bounded shuffling. A streaming neural fit path preserves the existing objective/model semantics while adding step/cursor checkpoints. Validation uses memory-mapped calibration stores; test predictions and mergeable metrics stream to disk, while globally ordered metrics use external sorted runs.

**Tech Stack:** Python, PyTorch IterableDataset/DataLoader, PyArrow Parquet, NumPy memmap, heapq external merge, existing BitGuard model/cascade/export modules, unittest.

---

### Task 1: Deterministic Parquet batch dataset

**Files:**
- Create: `src/bitguard_bnn/out_of_core/dataset.py`
- Create: `tests/test_out_of_core_dataset.py`

- [ ] **Step 1: Write failing coverage and determinism tests**

Create multiple small shards. Iterate with zero and two workers, assert every UID exactly once, stable order for the same `(seed,epoch)`, a different order across epochs, and no batch over the configured size.

```python
dataset.set_epoch(3)
first = collect_uids(DataLoader(dataset, num_workers=2))
dataset.set_epoch(3)
second = collect_uids(DataLoader(dataset, num_workers=2))
self.assertEqual(first, second)
self.assertEqual(sorted(first), sorted(expected_uids))
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_dataset -v`

- [ ] **Step 3: Implement shard ordering and worker ownership**

Define:

```python
@dataclass(frozen=True)
class DataCursor:
    epoch: int
    shard_position: int
    batch_position: int
    optimizer_step: int

class ParquetTrainingDataset(torch.utils.data.IterableDataset):
    def set_epoch(self, epoch: int, cursor: DataCursor | None = None) -> None:
        self.epoch = int(epoch)
        self.cursor = cursor

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        for position, shard in enumerate(self.permuted_shards(self.epoch)):
            if position % worker_count != worker_id:
                continue
            if self.cursor is not None and position < self.cursor.shard_position:
                continue
            yield from self.iter_shard_batches(shard, position)
```

Derive shard permutation with `np.random.Generator(PCG64(seed + epoch))`. Assign permuted shard positions by `position % worker_count == worker_id`. A resume cursor filters globally earlier positions before worker assignment.

- [ ] **Step 4: Implement bounded within-shard shuffling**

Use `ParquetFile.iter_batches`. Fill a fixed-size row buffer, permute it from a seed derived from `(seed,epoch,shard fingerprint,buffer index)`, and yield feature/label/UID batches carrying their global `(shard_position,batch_position)` ID. Convert selected raw columns through the frozen preprocessor in the worker. The final partial buffer is also permuted. The main process consumes IDs in global order (buffering out-of-order prefetched worker results) and checkpoints the next unconsumed ID; prefetched but unapplied batches may be regenerated safely after resume.

- [ ] **Step 5: Run dataset tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_dataset -v`

- [ ] **Step 6: Commit Task 1**

Record every-row-once and worker-disjointness invariants.

### Task 2: Streaming class weights and validation cache

**Files:**
- Create: `src/bitguard_bnn/out_of_core/cache.py`
- Create: `tests/test_out_of_core_cache.py`

- [ ] **Step 1: Write failing memmap and recovery tests**

Assert a cache preallocates exact shapes from manifest counts, commits batches atomically through an index journal, reopens after interruption, rejects a mismatched manifest fingerprint, and exposes read-only NumPy memmaps.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_cache -v`

- [ ] **Step 3: Implement calibration storage**

`CalibrationCache` creates memmaps for row UID hashes, true labels, known probabilities, selected scaled values required for open-set distance, Tiny probabilities, Boolean flags, and routing metadata. A JSON journal records committed row ranges and fingerprint. Flush memmaps before atomically advancing the journal.

- [ ] **Step 4: Derive class weights from shard manifests**

Add `class_weights_from_counts(counts, active_labels)` and verify it matches existing `class_weights` on fixtures. Fail when any active training class has zero count.

- [ ] **Step 5: Run cache tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_cache -v`

- [ ] **Step 6: Commit Task 2**

Record why validation arrays are disk-backed rather than sampled.

### Task 3: Step-resumable streaming neural fit

**Files:**
- Modify: `src/bitguard_bnn/trainer.py`
- Create: `src/bitguard_bnn/out_of_core/trainer.py`
- Modify: `src/bitguard_bnn/config.py`
- Create: `tests/test_out_of_core_training.py`

- [ ] **Step 1: Write failing in-memory parity and resume tests**

With one shard, disabled shuffle, dropout zero, and identical model initialization, compare one epoch of streaming versus array training history/state. Interrupt after step 2, resume, and compare exact final history/model to uninterrupted streaming training with dropout enabled.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_training -v`

- [ ] **Step 3: Extract shared batch/validation operations**

Move one optimizer update into a helper used by both paths:

```python
def neural_train_step(
    model: Any,
    features: Any,
    target: Any,
    objective: Any,
    optimizer: Any,
    scaler: Any,
    config: dict[str, Any],
    teacher_model: Any | None,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=features.device.type, enabled=scaler.is_enabled()):
        logits = model(features)
        with torch.no_grad():
            teacher_logits = None if teacher_model is None else teacher_model(features)
        output = objective(model, logits, target, teacher_logits)
    scaler.scale(output.total).backward()
    scaler.unscale_(optimizer)
    clip = float(config["training"].get("gradient_clip", 0.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    clamp_binary_master_weights(model)
    return {
        "loss": float(output.total.detach()),
        "detection": float(output.detection.detach()),
        "feature_cost": float(output.feature_cost.detach()),
        "fn": float(output.false_negative.detach()),
        "fp": float(output.false_positive.detach()),
    }
```

Do not change objective terms, gradient clipping, binary weight clamping, AMP, or metric names.

- [ ] **Step 4: Implement `fit_neural_streaming`**

The function consumes `ParquetTrainingDataset`, class counts, and a validation inference callback. Accumulate epoch totals by actual rows. Run validation after every epoch from the complete validation cache/inference pass. Preserve selection score, early stopping, scheduler, teacher distillation, Main/Tiny behavior, and best-state semantics.

- [ ] **Step 5: Extend checkpoint state and cursor validation**

Add streaming `format_version: 4` with a primitive cursor mapping, safe tensor/primitive RNG state, optimizer step, partial epoch totals/seen rows, shard/preprocessor/source fingerprints, shuffle-buffer settings, and dataset algorithm version. Version 4 supersedes the incompatible local v3 prototype that serialized `DataCursor` and raw NumPy state. Save every `checkpoint_every_steps` and epoch. Restore Python/NumPy/Torch/DataLoader RNG and reject any signature difference.

Add config validation:

```yaml
training:
  checkpoint_every_steps: 1000
  shuffle_buffer_rows: 262144
```

- [ ] **Step 6: Make persistent worker cleanup exception-safe**

Wrap the entire streaming epoch loop in `try/finally` and call public iterator shutdown where available, retaining the current compatibility helper only in one function. Add a test whose objective raises and assert all mocked workers shut down.

- [ ] **Step 7: Run training tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_training tests.test_training_runtime -v`

- [ ] **Step 8: Commit Task 3**

Record exact resume invariants and the shared train-step boundary.

### Task 4: Streaming validation calibration and cascade routing

**Files:**
- Create: `src/bitguard_bnn/out_of_core/calibrate.py`
- Modify: `src/bitguard_bnn/cascade.py`
- Create: `tests/test_out_of_core_calibration.py`

- [ ] **Step 1: Write failing calibration parity tests**

Write fixture predictions into a calibration cache and compare open-set threshold, Tiny exit threshold, Boolean thresholds, routed validation probabilities, and fixed-FPR thresholds with current in-memory functions. Include tied float32 scores and temporal metadata across batch boundaries.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_calibration -v`

- [ ] **Step 3: Stream Main and Tiny validation inference into cache**

Populate each cache range from Parquet batches. Build Boolean calibration summaries from bounded per-feature sufficient state. Main known probabilities and selected scaled values are memmapped, not held as Python lists.

- [ ] **Step 4: Route validation in global chronological order**

Create external ordering runs of `(timestamp,device_id,row_uid,cache_position)` when temporal routing is enabled. Feed cache positions through the existing state machine in order and write routed probabilities to a second memmap. Non-temporal protocols use source sequence order.

- [ ] **Step 5: Calibrate fixed-FPR after routed validation**

Use the routed benign scores and the existing discrete threshold semantics. Compute exact top-k thresholds from a memory-mapped vector using `np.partition`; retain float64 comparison behavior. Never inspect test labels or probabilities.

- [ ] **Step 6: Run calibration tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_calibration tests.test_config_and_metrics -v`

- [ ] **Step 7: Commit Task 4**

Record that calibration follows the exact deployed cascade score pipeline.

### Task 5: Streaming test predictions and mergeable metrics

**Files:**
- Create: `src/bitguard_bnn/out_of_core/metrics.py`
- Create: `src/bitguard_bnn/out_of_core/evaluate.py`
- Create: `src/bitguard_bnn/out_of_core/replay.py`
- Modify: `src/bitguard_bnn/cli.py`
- Modify: `src/bitguard_bnn/state.py`
- Create: `tests/test_out_of_core_metrics.py`

- [ ] **Step 1: Write failing metric parity tests**

Split a fixture at many different batch boundaries and compare accuracy, balanced accuracy, macro/per-class precision/recall/F1, MCC, high-risk FNR, unknown recall, Brier, ECE, fixed-FPR observations, AUROC, and AUPRC to `classification_metrics` within `1e-12` where algorithms are exact.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_metrics -v`

- [ ] **Step 3: Implement mergeable accumulators**

Use integer confusion matrices and sample counts. Accumulate Brier sums in float64 and fixed ECE bin count/confidence/correct sums. Derive all confusion-based metrics from totals at finalize time.

```python
class StreamingClassificationMetrics:
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> None:
        true_index = self.encode(y_true)
        pred_index = self.encode(y_pred)
        np.add.at(self.confusion, (true_index, pred_index), 1)
        one_hot = np.eye(len(self.labels), dtype=np.float64)[true_index]
        self.brier_sum += float(np.square(probabilities - one_hot).sum())
        self.rows += len(y_true)
        self.update_calibration_bins(true_index, probabilities)
        self.score_runs.update(y_true, probabilities)

    def finalize(self, operating_thresholds: Mapping[float, float]) -> dict[str, Any]:
        metrics = metrics_from_confusion(self.confusion, self.labels)
        metrics["multiclass_brier_score"] = self.brier_sum / max(self.rows, 1)
        metrics.update(self.finalize_calibration_and_ordered_scores(operating_thresholds))
        return metrics
```

- [ ] **Step 4: Implement exact ordered score metrics**

Write `(score,target,row_uid)` bounded sorted runs per class. Merge runs and compute tie-aware ROC trapezoids and average precision using grouped equal scores. Tests must cover ties and absent classes. Delete runs only after metrics JSON is committed.

- [ ] **Step 5: Stream test evaluation to compressed Parquet**

Write prediction row groups atomically with metadata, labels, exit stage, and canonical probabilities. Update metrics from the same batch. Temporal cascade evaluation uses external chronological positions and bounded state, then restores output order by UID/position for the prediction artifact.

- [ ] **Step 6: Add deterministic display-only plot sampling**

Retain rows by stable priority up to `evaluation.plot_sample_rows`. Mark plot manifest fields `numeric_metrics_scope: full_test` and `plot_rows_scope: deterministic_sample`.

- [ ] **Step 7: Preserve temporal replay for Parquet predictions**

Teach `bitguard replay` to dispatch by prediction artifact format. CSV runs retain the existing `replay_predictions` path. Full-data runs stream prediction Parquet through `replay_parquet_predictions`, preserve chronological state across record batches, write `temporal_predictions.parquet`, and accumulate operational metrics without a complete DataFrame.

- [ ] **Step 8: Run metric and replay tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_metrics tests.test_config_and_metrics -v`

- [ ] **Step 9: Commit Task 5**

Record that ordered metrics use external sorting and numeric reports remain full-test.

### Task 6: Full run orchestration, export, and classical guardrails

**Files:**
- Create: `src/bitguard_bnn/out_of_core/run.py`
- Modify: `src/bitguard_bnn/trainer.py`
- Modify: `src/bitguard_bnn/export.py`
- Modify: `src/bitguard_bnn/bootstrap/orchestrator.py`
- Create: `tests/test_out_of_core_run.py`

- [ ] **Step 1: Write a failing end-to-end run test**

Prepare both dataset-shaped fixtures through Plan 2, train one Main/Tiny epoch, calibrate, evaluate, resume metadata, and export. Assert all-row train consumption, complete validation/test counts, fixed-FPR source/pipeline, prediction Parquet, model summaries, and end-to-end export logit parity.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_run -v`

- [ ] **Step 3: Dispatch by storage mode**

`run_training` loads config and calls `run_out_of_core_training` only when `dataset.storage == "parquet"`; existing CSV behavior stays unchanged. Out-of-core run loads and verifies shard/preprocessing manifests before creating a run directory.

- [ ] **Step 4: Sequence memory-bounded phases**

Train teacher only when configured, then Main, release training-only objects, train Tiny, build validation calibration cache, calibrate/route validation, release validation state, stream test evaluation, and export artifacts. Record phase peak RSS and disk temporary usage separately.

- [ ] **Step 5: Add classical-model guardrails**

For non-incremental classical model types in a Parquet full profile, raise:

```text
<model> does not support exact out-of-core fitting; choose a neural model or an explicitly supported incremental baseline
```

Do not invoke `fit` on a capped subset. Add this validation before any training allocation.

- [ ] **Step 6: Preserve export compatibility**

Export consumes the same best Main checkpoint and preprocessor artifact as in-memory runs. Parquet storage metadata stays outside the edge manifest. Run existing gate-pruning and BatchNorm parity checks.

- [ ] **Step 7: Complete bootstrap train/summarize stages**

For `--dataset all`, run N-BaIoT then BoT-IoT, writing per-dataset status so a failed second run does not invalidate the completed first run. `--prepare-only` skips both. Final report links manifests, runs, checkpoints, metrics, and exports.

- [ ] **Step 8: Run integration tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_run tests.test_training_integration tests.test_gate_export -v`

- [ ] **Step 9: Commit Task 6**

Record storage dispatch, phase lifetimes, and the no-silent-classical-sampling constraint.

### Task 7: Cross-platform smoke, recovery drills, and documentation

**Files:**
- Create: `tests/test_full_bootstrap_recovery.py`
- Modify: `README.md`
- Modify: `VALIDATION.md`
- Modify: `RESEARCH_MAPPING.md`

- [ ] **Step 1: Write forced-interruption integration tests**

Inject failures during a Parquet shard write, optimizer step, validation cache write, and prediction row group. Rerun bootstrap and assert it reuses only committed work, produces the same final fingerprints/metrics as uninterrupted execution, and leaves no final artifact sourced from `.partial` data.

- [ ] **Step 2: Add low-resource failure tests**

Mock insufficient disk and CUDA verification failure. Assert no train run directory exists and the final bootstrap report contains required/available bytes or the failed CUDA profile plus an explicit CPU rerun command.

- [ ] **Step 3: Run CPU full-pipeline smoke**

Run the complete small fixture through both dataset profiles with `--compute cpu`. Verify row coverage, one epoch, calibration, metrics, checkpoint, export, and idempotent second execution.

- [ ] **Step 4: Run CUDA AMP smoke when available**

Use the same prepared fixture with `--compute cuda`, AMP enabled, and deterministic algorithms. Verify a step checkpoint can be resumed and the environment report identifies the selected official wheel profile.

- [ ] **Step 5: Document the operational procedure**

README must include exact Windows/Linux commands, Python/7-Zip prerequisites, manual BoT-IoT acquisition, disk preflight, CPU warning, resume/restart-stage behavior, artifact layout, and a clear statement that complete CSV flow records—not PCAP—define the full profile.

VALIDATION must distinguish fixture smoke evidence from unperformed 16.7 GB production execution. RESEARCH_MAPPING must describe exact all-row coverage, approximate train-only quantile sketches, full-test numeric metrics, and plot sampling.

- [ ] **Step 6: Run final verification**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python -m ruff check --no-cache src tests scripts
git diff --check HEAD
```

Also load every YAML config and run fresh CPU/CUDA fixture bootstrap commands. If Ruff or CUDA is unavailable, record that explicitly; do not claim those checks passed.

- [ ] **Step 7: Review against every design acceptance criterion**

Map each of the six acceptance criteria in `docs/superpowers/specs/2026-07-14-full-dataset-bootstrap-design.md` to a passing test or smoke artifact. Resolve any gap before completion.

- [ ] **Step 8: Commit Task 7**

Use a Lore commit whose `Tested:` trailers list exact unit, CPU, CUDA, compile, lint, and diff evidence and whose `Not-tested:` identifies the real full 16.7 GB run unless it was actually executed.
