"""Chunked raw-feature inference using a fixed RBC deployment artifact."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .rbc_edge import PackedRBCRuntime
from .rbc_encoder import RiskWeightedComparisonEncoder


def predict_csv(run, source, destination, *, chunk_size=4096):
    run, source, destination = Path(run), Path(source), Path(destination)
    encoder = RiskWeightedComparisonEncoder.load(run / "encoder.json")
    runtime = PackedRBCRuntime.load(run / "edge.npz")
    calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("threshold") is None:
        raise ValueError("run has no calibrated operating point")
    if destination.exists():
        raise FileExistsError(destination)
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        for frame in pd.read_csv(source, chunksize=chunk_size):
            prediction = runtime.predict(encoder.transform(frame))
            scores = prediction.detection_scores[:, 0]
            result = pd.DataFrame({"attack_score": scores,
                                   "attack_suspected": scores > calibration["threshold"],
                                   "attack_type_index": prediction.attack_types})
            labels = runtime.metadata.get("type_labels")
            if labels:
                result["type_estimate"] = [labels[int(index)] for index in prediction.attack_types]
            result.to_csv(handle, index=False, header=rows == 0)
            rows += len(frame)
    return {"rows": rows, "output": str(destination.resolve()), "automatic_blocking": False}
