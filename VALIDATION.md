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

## Bootstrap acquisition validation

The source-bootstrap tests use only small local fixtures; they do not download
research data. They cover preflight, official-source policy, local BoT-IoT
directory handling, safe ZIP extraction, bounded schema inspection, immutable
manifests, state reuse, source-change invalidation, explicit restart, nonzero
CLI failures, credential-redacted atomic reports, and cleanup-debt accounting.

Run the acquisition gate directly:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_bootstrap_entry tests.test_bootstrap_cli tests.test_bootstrap_state tests.test_bootstrap_preflight tests.test_bootstrap_download tests.test_bootstrap_extract tests.test_bootstrap_inspect tests.test_bootstrap_integration -v
```

Before using official full data, independently confirm free disk, the selected
Torch/CUDA profile, and the manually obtained BoT-IoT license terms. The
academic acknowledgement flag is not a license grant. PCAP is deliberately out
of scope, and full CSV preparation/training is expected to be long-running.

Safety claims are limited to untrusted network/archive content and cooperative
writers in a trusted workspace. They do not cover malicious same-account
parent-namespace or hardlink mutation. The report lists retained cleanup debt;
BitGuard does not remove those paths automatically.
