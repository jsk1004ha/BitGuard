"""Compare frozen-type training with identical inputs, batches, and optimizer steps."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bitguard_bnn.rbc import _batches, _train_types
from bitguard_bnn.rbc_model import DetectionFirstBNN, conditional_type_loss


def original_loop(model, values, target, types, args):
    optimizer = torch.optim.Adam(model.type_parameters(), lr=.001)
    rng = np.random.default_rng(29)
    for _ in range(args.epochs):
        model.train()
        for idx in _batches(len(values), args.batch_size, rng):
            if not bool(target[idx].any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = conditional_type_loss(model(values[idx]).type_logits, types[idx],
                                         attack_mask=target[idx].bool())
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    base = DetectionFirstBNN(64, [64, 32], attack_classes=4)
    base.freeze_detector()
    values = torch.where(torch.randn(args.rows, 64) > 0, 1., -1.)
    target = (torch.arange(args.rows) % 3 != 0).float()
    types = torch.arange(args.rows) % 4
    timings = {"original": [], "cached": []}
    report = None
    # Warm both paths, then alternate timing order to reduce order/thermal bias.
    for repeat in range(args.repeats + 1):
        states = {}
        order = ("original", "cached") if repeat % 2 == 0 else ("cached", "original")
        for name in order:
            model = copy.deepcopy(base)
            start = time.perf_counter()
            if name == "original":
                original_loop(model, values, target, types, args)
            else:
                report = _train_types(model, values, target, types, epochs=args.epochs,
                                      batch_size=args.batch_size, seed=29, lr=.001)
            elapsed = time.perf_counter() - start
            if repeat:
                timings[name].append(elapsed)
            states[name] = model.state_dict()
        for key in states["original"]:
            torch.testing.assert_close(states["original"][key], states["cached"][key],
                                       rtol=0, atol=0)
    medians = {name: statistics.median(times) for name, times in timings.items()}
    result = {"scope": "frozen type-head training, generated CPU data, includes cache construction",
              "rows": args.rows, "epochs": args.epochs, "batch_size": args.batch_size,
              "threads": 1, "torch": str(torch.__version__), "python": platform.python_version(),
              "processor": platform.processor(), "repeats": args.repeats,
              "seconds": timings, "median_seconds": medians,
              "speedup": medians["original"] / medians["cached"],
              "cache_bytes": report["cache_bytes"], "parameter_updates_bitwise_equal": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
