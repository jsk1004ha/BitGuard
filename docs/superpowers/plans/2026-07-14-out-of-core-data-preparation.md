# Out-of-Core Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every verified N-BaIoT and BoT-IoT CSV row into deterministic, leakage-free, bounded-memory Parquet partitions with train-only streaming preprocessing artifacts.

**Architecture:** Existing dataset normalization is exposed as a chunk iterator shared by in-memory and full-data paths. An inspection/index pass plans exact device/attack/time splits; streaming sufficient statistics and deterministic priority sketches fit preprocessing on train only; a final pass writes immutable selected-feature Parquet shards plus a coverage manifest.

**Tech Stack:** Python, pandas chunk readers, NumPy, PyArrow Parquet, heapq external merge, hashlib, existing BitGuard label/preprocessing contracts, unittest.

---

### Task 1: Share normalized CSV chunk adapters

**Files:**
- Modify: `src/bitguard_bnn/data.py`
- Create: `src/bitguard_bnn/out_of_core/__init__.py`
- Create: `src/bitguard_bnn/out_of_core/source.py`
- Modify: `tests/test_data_and_preprocess.py`
- Create: `tests/test_out_of_core_source.py`

- [ ] **Step 1: Write failing chunk-parity tests**

```python
def test_normalized_chunk_iterator_matches_in_memory_loader(self):
    expected = load_dataset(config).frame.sort_values("row_uid").reset_index(drop=True)
    actual = pd.concat(iter_normalized_chunks(config), ignore_index=True)
    actual = actual.sort_values("row_uid").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected)

def test_iterator_never_concatenates_source_chunks(self):
    with patch("bitguard_bnn.data.pd.concat", side_effect=AssertionError("concat")):
        rows = sum(len(chunk) for chunk in iter_normalized_chunks(config))
    self.assertEqual(rows, expected_rows)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_source -v`

- [ ] **Step 3: Extract a common chunk interface**

Define:

```python
@dataclass(frozen=True)
class NormalizedChunk:
    frame: pd.DataFrame
    source_relative_path: str
    source_row_start: int

def iter_normalized_chunks(
    config: dict[str, Any],
    *,
    path_override: Path | None = None,
    apply_sampling_caps: bool = True,
) -> Iterator[NormalizedChunk]:
    cfg = config["dataset"]
    paths = _resolve_source_paths(config, path_override)
    cap_state = _DevelopmentCapState(config) if apply_sampling_caps else None
    for source_path in paths:
        row_start = 0
        for raw_chunk in _read_source_chunks(source_path, int(cfg["chunk_size"])):
            frame = _normalize_source_chunk(config, source_path, raw_chunk, row_start)
            if cap_state is not None:
                frame = cap_state.apply(frame)
            yield NormalizedChunk(frame, _relative_source_name(source_path), row_start)
            row_start += len(raw_chunk)
```

Use the existing label maps, drop columns, metadata normalization, and stable row UID rules. `apply_sampling_caps=False` ignores only `max_rows_per_file`, `max_rows_per_class`, and `max_loaded_rows`; all schema and safety validation remains active.

- [ ] **Step 4: Rebuild the in-memory loader on the iterator**

`load_dataset` may concatenate yielded chunks only at its final compatibility boundary. Existing capped behavior and errors must remain byte-for-byte compatible where tests assert messages.

- [ ] **Step 5: Run old and new data tests**

Run: `PYTHONPATH=src python -m unittest tests.test_data_and_preprocess tests.test_out_of_core_source -v`

- [ ] **Step 6: Commit Task 1**

Record that normalization has one implementation and two storage consumers.

### Task 2: Inspection manifest and exact partition plans

**Files:**
- Create: `src/bitguard_bnn/out_of_core/manifest.py`
- Create: `src/bitguard_bnn/out_of_core/split.py`
- Create: `tests/test_out_of_core_split.py`

- [ ] **Step 1: Write failing device, attack, and time split tests**

Generate chunks whose timestamps are deliberately unsorted across files. Assert exact 70/15/15 rank membership matches an in-memory stable sort by `(timestamp, row_uid)`. Assert device and held-out attack members match existing split semantics and all UID sets are disjoint.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_split -v`

- [ ] **Step 3: Define inspection and split manifests**

```python
@dataclass(frozen=True)
class SourceRowRecord:
    row_uid: str
    source_file: str
    source_row: int
    behavior_label: str
    raw_attack: str
    device_id: str
    timestamp: float | None

@dataclass(frozen=True)
class SplitPlan:
    strategy: str
    train_count: int
    validation_count: int
    test_count: int
    membership_path: Path
    fingerprint: str
```

Write row index runs as Parquet with only split metadata. Store counts, source coverage, rejected reasons, schemas, and source-manifest fingerprint.

- [ ] **Step 4: Implement exact external chronological cutoffs**

Sort bounded runs by `(timestamp, row_uid)`, then merge with `heapq.merge`. At rank `n_train` and `n_train + n_validation`, persist the full ordering keys. Assign equal timestamps by UID so boundaries are deterministic. Reject missing/non-finite timestamps for a time protocol.

- [ ] **Step 5: Implement group membership plans**

Device and attack plans assign from normalized metadata. Random protocol uses a stable keyed hash of `(seed,row_uid)` and exact rank runs, not process-random hashes. Emit overlap and unknown-in-train checks identical to current manifests.

- [ ] **Step 6: Run split tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_split -v`

- [ ] **Step 7: Commit Task 2**

Record exact rank semantics and the UID tie-breaker.

### Task 3: Deterministic bounded quantile sketches

**Files:**
- Create: `src/bitguard_bnn/out_of_core/quantiles.py`
- Create: `tests/test_out_of_core_quantiles.py`

- [ ] **Step 1: Write determinism, merge, and error tests**

```python
def test_priority_sketch_is_order_and_merge_independent():
    whole = PriorityRowSketch(capacity=4096, seed=17, width=3)
    left = PriorityRowSketch(capacity=4096, seed=17, width=3)
    right = PriorityRowSketch(capacity=4096, seed=17, width=3)
    for uid, values in records:
        whole.update(uid, values)
    for uid, values in records[::2]:
        left.update(uid, values)
    for uid, values in records[1::2]:
        right.update(uid, values)
    left.merge(right)
    self.assertEqual(whole.snapshot(), left.snapshot())

def test_quantiles_stay_within_declared_fixture_tolerance():
    self.assertLessEqual(abs(sketch.quantile(0, 0.5) - np.median(values[:, 0])), 0.03)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_quantiles -v`

- [ ] **Step 3: Implement stable row-priority sampling**

For each `(seed,row_uid)`, derive one 128-bit BLAKE2b priority and retain the lowest `capacity` complete feature rows in a max heap, using row UID as the deterministic tie-breaker. Computing one priority per row avoids hashing every feature separately. Merge by applying the same top-k rule. Quantiles operate per column over finite retained values while total finite/missing counts are tracked over all rows. Serialize width, capacity, seed, retained priorities/UIDs/row values, counts, and algorithm version.

- [ ] **Step 4: Expose an error contract**

Report the Dvoretzky-Kiefer-Wolfowitz confidence bound for retained sample size as metadata, without claiming a deterministic value-error bound. Tests use distribution-specific tolerances only for parity fixtures.

- [ ] **Step 5: Run quantile tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_quantiles -v`

- [ ] **Step 6: Commit Task 3**

Record why priority sampling is used instead of retaining full feature columns.

### Task 4: Train-only streaming preprocessing

**Files:**
- Create: `src/bitguard_bnn/out_of_core/preprocess.py`
- Modify: `src/bitguard_bnn/preprocess.py`
- Create: `tests/test_out_of_core_preprocess.py`

- [ ] **Step 1: Write failing parity and leakage tests**

Use a fixture without missing values for exact F-score parity and another with missing/outliers for declared quantile tolerance. Change only validation/test values and assert every fitted train artifact remains identical.

```python
self.assertEqual(streaming.selected_features, in_memory.selected_features)
np.testing.assert_allclose(streaming.selection_scores, in_memory.selection_scores, rtol=1e-5)
self.assertEqual(before_train_artifact, after_test_mutation_artifact)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_preprocess -v`

- [ ] **Step 3: Implement mergeable class sufficient statistics**

Track per-feature/per-class finite count, sum, and squared sum. After median values are fixed from sketches, make a second train pass to include imputed values and compute ANOVA between/within sums of squares. Match `sklearn.feature_selection.f_classif` for finite fixtures.

- [ ] **Step 4: Implement the streaming fit phases**

Define:

```python
class StreamingFeaturePreprocessor:
    def inspect_batch(self, row_uid: np.ndarray, values: np.ndarray, labels: np.ndarray) -> None:
        self.class_counter.update(labels.tolist())
        self.finite_counts += np.isfinite(values).sum(axis=0)
        self.imputation_sketch.update_many(row_uid, values)

    def finalize_imputation(self) -> None:
        self.medians = np.asarray([
            self.imputation_sketch.quantile(index, 0.5)
            for index in range(self.imputation_sketch.width)
        ])

    def accumulate_anova_batch(self, values: np.ndarray, labels: np.ndarray) -> None:
        imputed = np.where(np.isfinite(values), values, self.medians)
        self.anova.update(imputed, labels)

    def finalize_selection(self) -> None:
        self.selection_scores = self.anova.finalize()
        self.selected_indices = self.rank_selected(self.selection_scores)

    def calibrate_selected_batch(self, row_uid: np.ndarray, values: np.ndarray, labels: np.ndarray) -> None:
        selected = np.where(
            np.isfinite(values[:, self.selected_indices]),
            values[:, self.selected_indices],
            self.medians[self.selected_indices],
        )
        self.selected_calibration.update_many(row_uid, selected, labels)

    def finalize(self) -> FeaturePreprocessor:
        result = FeaturePreprocessor(self.config)
        self.populate_frozen_preprocessor(result)
        result.fitted = True
        return result
```

Phase 1 determines usable features, medians, labels, and class counts. Phase 2 imputes every train row with the frozen medians, accumulates exact class sufficient statistics, and freezes F-scores, costs, and selection. Phase 3 determines robust quartiles, encoder thresholds, benign center, and benign-distance threshold from selected train features. Validation-only open-set confidence remains a later evaluation operation.

- [ ] **Step 5: Persist sketch provenance in `feature_manifest`**

Add `fit_mode`, algorithm version, capacity, seed, confidence bound, rows considered, retained counts, and exact-versus-approximate field names. The standard in-memory manifest keeps `fit_mode: in_memory_exact`.

- [ ] **Step 6: Run preprocessing tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_preprocess tests.test_data_and_preprocess -v`

- [ ] **Step 7: Commit Task 4**

Record the train-only multi-pass contract and explicit quantile approximation.

### Task 5: Immutable Parquet shard writer and coverage verifier

**Files:**
- Create: `src/bitguard_bnn/out_of_core/shard.py`
- Create: `tests/test_out_of_core_shard.py`

- [ ] **Step 1: Write failing shard coverage tests**

Prepare a fixture with three source files and all partitions. Assert each accepted UID appears once, row/class totals match inspection, shards stay below the configured row target except final shards, and a changed shard byte fails hash verification.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_shard -v`

- [ ] **Step 3: Implement bounded shard writing**

Read normalized chunks, join split membership by row UID using disk-backed sorted merge, select only frozen features plus metadata/label, and write `dataset=<name>/split=<split>/label=<label>/part-*.parquet`. Use `.partial`, validate the Parquet footer and row count, fsync, then rename.

- [ ] **Step 4: Define shard manifests**

Each entry contains relative path, SHA-256, row count, label counts, schema fingerprint, UID min/max, source-file coverage, and min/max ordering keys. The top-level fingerprint covers sorted entries plus preprocessing and split fingerprints.

- [ ] **Step 5: Implement complete coverage verification**

Externally sort UID runs from shard metadata and source inspection, compare them one-by-one, and reject duplicates, missing rows, extras, split overlap, schema drift, or mismatched counts before deleting temporary indexes.

- [ ] **Step 6: Run shard tests**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_shard -v`

- [ ] **Step 7: Commit Task 5**

Record that the shard manifest, not directory presence, defines preparation completion.

### Task 6: Full profiles and preparation orchestrator

**Files:**
- Create: `configs/full/nbaiot.yaml`
- Create: `configs/full/botiot.yaml`
- Create: `src/bitguard_bnn/out_of_core/prepare.py`
- Modify: `src/bitguard_bnn/bootstrap/orchestrator.py`
- Modify: `src/bitguard_bnn/config.py`
- Create: `tests/test_out_of_core_prepare.py`
- Modify: `README.md`
- Modify: `VALIDATION.md`

- [ ] **Step 1: Write a failing complete preparation test**

Run preparation on N-BaIoT- and BoT-IoT-shaped fixtures. Assert uncapped resolved configs, verified Parquet manifests, train-only preprocessing artifacts, exact source totals, and idempotent second execution.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src python -m unittest tests.test_out_of_core_prepare -v`

- [ ] **Step 3: Add explicit Parquet dataset config**

Add these fields with validation:

```yaml
dataset:
  storage: parquet
  shard_manifest: data/prepared/nbaiot/shard_manifest.json
  max_rows_per_file: null
  max_rows_per_class: null
  max_loaded_rows: null
  record_batch_rows: 65536
  shard_target_rows: 1000000
  quantile_sketch_capacity: 200000
```

Full configs retain current scientific split/model settings but use uncapped Parquet storage. Existing configs remain development profiles.

- [ ] **Step 4: Implement `prepare_full_dataset`**

The function executes inspection, split planning, preprocessing passes, sharding, and verification using bootstrap state signatures. Stage output is a `PreparedDataset` containing manifest paths/fingerprints and resolved config path.

- [ ] **Step 5: Extend bootstrap through `prepare-only`**

Replace the temporary `sources_verified` endpoint from Plan 1. `--prepare-only` now succeeds only after shard coverage validation. Normal execution returns prepared dataset descriptors for Plan 3 training.

- [ ] **Step 6: Document full versus development profiles**

Include storage estimates, approximate quantile semantics, exact all-row coverage, PCAP exclusion, and commands for independent `--dataset` preparation.

- [ ] **Step 7: Run plan-level verification**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_out_of_core_source tests.test_out_of_core_split tests.test_out_of_core_quantiles tests.test_out_of_core_preprocess tests.test_out_of_core_shard tests.test_out_of_core_prepare -v
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

Expected: new and existing suites pass; all source fixture rows are accounted for.

- [ ] **Step 8: Commit Task 6**

Use a Lore commit explaining why uncapped profiles are separate and why verified coverage gates training.
