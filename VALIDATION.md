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
