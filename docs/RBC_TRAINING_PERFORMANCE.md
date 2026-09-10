# RBC training-loop optimization

The optimizer, number of epochs, shuffled minibatches, class weighting, selection
partitions, and stopping policy are retained. Changes remove repeated computation:

- A frozen BNN's final binary activations are computed once for multi-epoch type
  training and cached as int8. The type head receives the same float values and
  minibatches. Detector weights and BatchNorm statistics remain frozen.
- `rbc.type_cache_max_bytes` defaults to 268435456 (256 MiB). A larger representation
  uses uncached computation. Zero disables caching; one-epoch training avoids its
  setup cost. For N rows and a final width of 32, the cache requires N*32 bytes.
- Detection training/validation omits the unused type head. Class weights and
  the benign fraction are calculated once; sequential inference uses tensor slices.
- Group-loss means are aggregated with `unique`/`index_add`/`bincount`, replacing
  a Python loop that searched the whole batch once per group. Weighted means,
  validation errors, and tied-maximum gradient sharing retain their semantics.
  Floating-point reduction order can differ; value/gradient equivalence is tested
  with numerical tolerances rather than promising identical complete training runs.
- The Parquet loader caches its scheduled chunk count using immutable shard
  metadata and the mixing budget. Changes invalidate the cache; row order,
  file identity checks, expected chunk validation, and resume cursors are unchanged.

The runner records `training_seconds` and `validation_seconds` in `history.json`.
`type_training.json` records cache size/setup time, elapsed time, and optimizer steps.

CPU measurements on 2026-09-09 (PyTorch 2.13.0+cpu, one thread):

| Measured section | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Frozen type training, 16,384 rows, 5 epochs, batch 2,048 | 87.10 ms | 41.32 ms | 2.11x |
| Weighted group loss + backward, batch 2,048, 64 groups | 7.4341 ms | 0.4126 ms | 18.02x |
| Weighted group loss + backward, batch 2,048, 256 groups | 29.2813 ms | 0.4642 ms | 63.08x |

These are section-level synthetic benchmarks, not full dataset training speedups.
Type training uses five timed repetitions after warm-up, alternating measurement
order; cache construction is included and final model tensors match bit-for-bit.
Its cache uses 524,288 bytes. Group measurements use medians of 80 repetitions.
CUDA is unavailable in this environment; GPU performance is not verified.

Reproduce the type-loop benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_rbc_training.py --output docs/benchmarks/rbc_training_cpu.json
```

Validation: 49 RBC/cache tests passed (3 subtests); streaming dataset/training tests
had 35 passes and one existing error-message assertion failure. That failure also
reproduces with the original uncached chunk-count function: the linked-root attack
is rejected with `unsafe prepared shard parent path`, while the test expects
`manifest root`. No security check or test expectation was weakened. Ruff and
Python compilation passed. The full unrelated recovery suite was not rerun.
