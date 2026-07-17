# Validation status

Status updated on 2026-07-17. Evidence is separated by scale so fixture checks
cannot be mistaken for an official-dataset result.

## Completed fixture and focused-test evidence

The repository's generated fixtures are small and do not download N-BaIoT or
BoT-IoT. They validate code paths and contracts, not production accuracy or
throughput.

- A direct one-epoch CPU out-of-core run has been exercised for both an
  N-BaIoT-shaped fixture and a BoT-IoT-shaped fixture. The integration path
  prepares all fixture rows, trains Main and Tiny BNNs, calibrates open-set,
  cascade, and fixed-FPR thresholds from validation data, records the Boolean
  fast-path contract, evaluates the complete fixture test split, writes
  compressed Parquet predictions, replays temporal state, and produces packed
  edge exports.
- Focused suites cover exact split/shard coverage, train-only three-pass
  preprocessing, deterministic priority sketches, all-train-row epoch
  consumption, checkpoint cursor resume, pre-run validation of cursor, model,
  RNG, optimizer, scheduler, and AMP-scaler state, exact streaming metrics,
  atomic prediction publication/recovery, chronological replay, safe
  checkpoint loading, bootstrap dataset ordering, completed-run reuse, and
  prepare-only behavior.
- The classical in-memory path has separate synthetic integration coverage for
  validation-only fixed-FPR calibration and packed-export parity. It is not used
  by the uncapped Parquet profiles.
- On 2026-07-16, the wrapper-entry and bootstrap-CLI focused gate completed 40
  tests with no failures. It exercised argument forwarding, Python/profile
  selection, CUDA failure policy, help/parser behavior, path validation, package
  data, and official-source registry rules without downloading or training.
- The bootstrap help command, all four documented Windows/Linux full-train and
  prepare-only argument sets, and all nine YAML configs passed parser/config
  loading checks without creating data or starting training.
- The Task 7 CPU integration invoked one bootstrap operation for both generated
  dataset shapes, performed real prepare/train/calibrate/evaluate/export work,
  and passed an unchanged second invocation that reused the same two verified
  runs without preparing or training again.
- On 2026-07-17, `tests.test_full_bootstrap_recovery` completed all 19 tests
  without failures in 3 minutes 39 seconds, including the CPU dual-dataset,
  hard-exit, corrupt-state preflight, and CUDA AMP cases described below.
- Forced subprocess exits after an optimizer step and after a committed
  validation-cache journal both resumed automatically. The resumed results
  matched uninterrupted controls for final metrics, inference contract and
  fingerprint, prediction Parquet bytes, model state/history where applicable,
  and packed edge weights/manifest hashes.
- The CUDA Task 7 smoke ran on the available RTX 4060 Laptop GPU with the CUDA
  12.4 PyTorch build. A one-epoch AMP run was hard-exited after an optimizer
  step, resumed through the bootstrap checkpoint locator, and completed with a
  non-empty scaler state and the deterministic CUDA runtime contract intact.

These checks use generated data with deliberately small row groups, models,
and one-epoch settings. They do not establish official-data accuracy, wall-clock
training time, workstation peak storage, or edge-device performance.

## Final automated validation gates

- The targeted Task 7 integration evidence above is complete. It exercises the
  package bootstrap directly with generated local acquisition fixtures; it does
  not claim that the platform wrapper downloaded either official dataset.
- The final repository-wide pytest run collected 761 tests and completed with
  `749 passed, 12 skipped, 44 warnings` in 32 minutes 5 seconds. The skips are
  platform/permission or optional-environment cases; there were no failures.
- The final shard durability module completed with `89 passed, 1 skipped` in
  22 minutes 39 seconds. The one skip requires Windows symlink privileges.
- Focused static checks passed `compileall`, config loading and validation for
  all nine YAML files, `git diff --check`, Black for the nine task-owned Python
  files, and mypy for the six task-owned production files plus the entry test.
- Repository-wide Black still reports 67 pre-existing files that would be
  reformatted. Full dependency-following mypy reports 49 existing errors across
  dynamic fixtures and imported legacy modules; the task-owned production
  files pass when checked as explicit package bases. Neither broad cleanup was
  applied because it would rewrite unrelated user work.
- Ruff is not installed in the current environment and is recorded as
  unavailable, not passed.

## Production execution not performed

The real 16.7 GB production-scale CSV preparation and training run has **not**
been performed in this validation cycle. No official N-BaIoT/BoT-IoT dataset
hashes, final multi-seed metrics, full-scale elapsed time, or peak disk/RAM/GPU
measurements are claimed. The 69.3 GB BoT-IoT PCAP capture is outside the
project scope and is not part of that pending CSV run.

`configs/nbaiot.yaml` and `configs/botiot.yaml` are capped development profiles.
Only `configs/full/*.yaml` disable row caps and use verified Parquet shards. Do
not relabel a fixture or capped development result as a full-data result.

## Reproduce the automated gate

After installing development dependencies, run:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python -m ruff check --no-cache src tests scripts
git diff --check HEAD
```

The unit suite includes small CPU training fixtures and can take several
minutes. It does not download either research dataset. Record the exact command,
exit code, pass/skip count, and environment for a paper artifact.

Before a paper-quality run, also verify:

1. the full configs contain no development row caps;
2. source, split, preprocessing, shard, checkpoint, and inference fingerprints
   all match the final bootstrap and run reports;
3. fixed-FPR thresholds come from validation and numeric metrics cover every
   test row;
4. temporal wall-clock results use real timestamps and a verified continuous
   test episode;
5. final results use the planned multi-seed protocol;
6. packed runtime latency, RAM, and energy are measured on the target edge
   device rather than inferred from workstation PyTorch timing.

## Bootstrap acquisition validation

The source-bootstrap tests use only small local fixtures. They cover preflight,
official-source policy, local BoT-IoT directory handling, safe ZIP/RAR policy,
bounded schema inspection, immutable manifests, state reuse, source-change
invalidation, explicit restart, nonzero CLI failures, credential-redacted
atomic reports, and cleanup-debt accounting. The out-of-core preparation tests
add uncapped fixture totals, held-out-value leakage isolation, idempotent strict
revalidation, tamper failure, and exact shard coverage for both dataset shapes.

Run the acquisition gate directly:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_bootstrap_entry tests.test_bootstrap_cli tests.test_bootstrap_state tests.test_bootstrap_preflight tests.test_bootstrap_download tests.test_bootstrap_extract tests.test_bootstrap_inspect tests.test_bootstrap_integration -v
python -m unittest tests.test_out_of_core_source tests.test_out_of_core_split tests.test_out_of_core_quantiles tests.test_out_of_core_preprocess tests.test_out_of_core_shard tests.test_out_of_core_prepare -v
python -m unittest tests.test_out_of_core_dataset tests.test_out_of_core_training tests.test_out_of_core_calibration tests.test_out_of_core_metrics tests.test_out_of_core_run -v
```

Before using official full data, independently confirm free disk, the selected
Torch/CUDA profile, and the manually obtained BoT-IoT license terms. The
academic acknowledgement flag is not a license grant. PCAP is deliberately out
of scope, and full CSV preparation/training is expected to be long-running.
Preparation counts every accepted model-ready CSV row exactly once. Quantile-
derived medians, robust scales, encoder thresholds, and benign-distance values
remain deterministic approximations. Retain their capacity, seed, algorithm,
and probabilistic CDF error metadata; the implementation does not claim a
deterministic quantile value-error bound.

Safety claims are limited to untrusted network/archive content and cooperative
writers in a trusted workspace. They do not cover malicious same-account
parent-namespace or hardlink mutation. The report lists retained cleanup debt;
BitGuard does not remove those paths automatically.
