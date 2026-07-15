# Validation status

Validation performed on 2026-07-10:

- Python source, scripts, and tests pass `compileall`.
- 12 data, preprocessing, cascade, temporal-state, and streaming checks pass.
- Logistic Regression end-to-end smoke run succeeds, including data loading,
  attack-held-out split, train-only preprocessing, open-set calibration,
  metrics, plots, capture-aware temporal replay, and action output.
- The local validation container did not contain PyTorch. The STE/BinaryLinear
  test was therefore skipped here; installation declares `torch>=2.2`, and the
  neural smoke/export parity tests must be run in the actual training
  environment before using final BNN results.

Run locally:

```bash
python -m pip install -e '.[dev,plots]'
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests

bitguard make-demo --output data/demo.csv --rows 12000 --seed 2309
bitguard train --config configs/demo.yaml
```

Before a paper-quality run, also verify:

1. no development row caps on operational evaluation data;
2. the test capture is continuous and timestamps are real;
3. all neural/export parity tests pass with the selected PyTorch version;
4. final results use the planned multi-seed protocol;
5. packed runtime latency/RAM/energy are measured on the target edge device.

The repository does not bundle N-BaIoT or BoT-IoT. Dataset hashes and final
full-data/multi-seed results cannot be produced until those files are placed at
the configured paths. `configs/nbaiot.yaml` and `configs/botiot.yaml` are
row-capped development profiles. The separate `configs/full/*.yaml` profiles
disable every row cap and use verified Parquet shards; do not relabel a capped
development run as a full-data result.

## Bootstrap acquisition validation

The source-bootstrap tests use only small local fixtures; they do not download
research data. They cover preflight, official-source policy, local BoT-IoT
directory handling, safe ZIP extraction, bounded schema inspection, immutable
manifests, state reuse, source-change invalidation, explicit restart, nonzero
CLI failures, credential-redacted atomic reports, and cleanup-debt accounting.
The out-of-core preparation tests additionally prove uncapped source totals,
train-only three-pass preprocessing, held-out-value leakage isolation,
idempotent strict revalidation, tamper failure, and exact shard coverage for
N-BaIoT- and BoT-IoT-shaped fixtures.

Run the acquisition gate directly:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_bootstrap_entry tests.test_bootstrap_cli tests.test_bootstrap_state tests.test_bootstrap_preflight tests.test_bootstrap_download tests.test_bootstrap_extract tests.test_bootstrap_inspect tests.test_bootstrap_integration -v
python -m unittest tests.test_out_of_core_source tests.test_out_of_core_split tests.test_out_of_core_quantiles tests.test_out_of_core_preprocess tests.test_out_of_core_shard tests.test_out_of_core_prepare -v
```

Before using official full data, independently confirm free disk, the selected
Torch/CUDA profile, and the manually obtained BoT-IoT license terms. The
academic acknowledgement flag is not a license grant. PCAP is deliberately out
of scope, and full CSV preparation/training is expected to be long-running.
Preparation counts every accepted model-ready CSV row exactly once, while
quantile-derived medians/scales/encoder thresholds remain deterministic
approximations whose capacity and confidence metadata must be retained.

Safety claims are limited to untrusted network/archive content and cooperative
writers in a trusted workspace. They do not cover malicious same-account
parent-namespace or hardlink mutation. The report lists retained cleanup debt;
BitGuard does not remove those paths automatically.
