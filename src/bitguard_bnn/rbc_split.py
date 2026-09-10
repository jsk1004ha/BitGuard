"""Four disjoint research partitions; labels never rebalance time splits."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def four_way_split(frame, *, strategy="temporal", held_out_devices=(),
                   held_out_attacks=(), fractions=(.60, .15, .10, .15)):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (4,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("four positive split fractions must sum to one")
    if frame.empty or "row_uid" not in frame or frame.row_uid.duplicated().any():
        raise ValueError("unique row_uid values are required")
    names = ("train", "selection", "calibration", "test")
    parts = {name: [] for name in names}

    def divide(rows, shares, destinations, key):
        rows = rows.sort_values([key, "row_uid"], kind="stable")
        values = rows[key].to_numpy()
        boundaries = [0]
        for fraction in np.cumsum(shares)[:-1]:
            position = min(int(len(rows) * fraction), len(rows))
            # Keep identical timestamps/sequence indices on one side of the boundary.
            while 0 < position < len(rows) and values[position] == values[position - 1]:
                position += 1
            boundaries.append(position)
        boundaries.append(len(rows))
        for name, left, right in zip(destinations, boundaries, boundaries[1:]):
            parts[name].append(rows.iloc[left:right])

    if strategy == "temporal":
        if "timestamp" not in frame or not np.isfinite(frame.timestamp).all():
            raise ValueError("temporal split requires verified finite timestamps")
        divide(frame, fractions, names, "timestamp")
    elif strategy == "device":
        if not held_out_devices:
            raise ValueError("device split requires explicit held_out_devices")
        held = frame.device_id.astype(str).isin([str(value) for value in held_out_devices])
        if not held.any():
            raise ValueError("held_out_devices do not occur in this dataset")
        parts["test"].append(frame[held])
        for _, rows in frame[~held].groupby(["device_id", "source_file"], sort=True):
            divide(rows, fractions[:3] / fractions[:3].sum(), names[:3], "sequence_index")
    else:
        raise ValueError("RBC split strategy must be temporal or device")
    result = {name: pd.concat(rows, ignore_index=True) if rows else frame.iloc[:0].copy()
              for name, rows in parts.items()}
    # Excluded attacks are withheld from all fitting, selection, and calibration.
    # Earlier excluded rows are discarded, never moved across time boundaries.
    for name in names[:3]:
        rows = result[name]
        excluded = rows.behavior_label.isin(held_out_attacks)
        if "raw_attack" in rows:
            excluded |= rows.raw_attack.isin(held_out_attacks)
        result[name] = rows[~excluded].reset_index(drop=True)
    test = result["test"]
    held_attack = test.behavior_label.isin(held_out_attacks)
    if "raw_attack" in test:
        held_attack |= test.raw_attack.isin(held_out_attacks)
    result["test"] = test.copy()
    result["test"]["held_out_attack"] = held_attack
    result["test"]["original_behavior_label"] = test.behavior_label
    result["test"].loc[held_attack, "behavior_label"] = "unknown_like"
    for name, rows in result.items():
        if rows.empty:
            raise ValueError(f"{name} partition is empty; revise the preregistered split")
    for name in ("train", "selection"):
        benign = result[name].behavior_label.eq("benign")
        if not benign.any() or benign.all():
            raise ValueError(f"{name} requires benign and attack support")
    if not result["calibration"].behavior_label.eq("benign").any():
        raise ValueError("calibration requires benign support")
    return result


def split_manifest(parts):
    return {name: {"rows": len(frame),
                   "row_uid_sha256": hashlib.sha256(
                       "\n".join(sorted(frame.row_uid.astype(str))).encode()).hexdigest(),
                   "class_counts": frame.behavior_label.value_counts().to_dict(),
                   "devices": sorted(frame.device_id.astype(str).unique().tolist())}
            for name, frame in parts.items()}
