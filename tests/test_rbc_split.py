import numpy as np
import pandas as pd
import pytest
import torch

from bitguard_bnn.rbc_split import four_way_split
from bitguard_bnn.losses import FocalLoss


def frame():
    return pd.DataFrame({"row_uid": [str(i) for i in range(100)],
                         "timestamp": np.arange(100) // 2,
                         "behavior_label": ["benign", "scan_like"] * 50,
                         "device_id": "d", "source_file": "f",
                         "sequence_index": np.arange(100)})


def test_four_disjoint_time_partitions_and_ties():
    parts = four_way_split(frame())
    assert sum(map(len, parts.values())) == 100
    previous = -1
    seen = set()
    for part in parts.values():
        assert part.timestamp.min() > previous
        previous = part.timestamp.max()
        assert not seen.intersection(part.row_uid)
        seen.update(part.row_uid)


def test_unknown_never_enters_fitting_partitions():
    rows = frame()
    rows.loc[[3, 63, 83, 93], "behavior_label"] = "unknown_like"
    parts = four_way_split(rows, held_out_attacks=["unknown_like"])
    assert all(not p.behavior_label.eq("unknown_like").any()
               for key, p in parts.items() if key != "test")
    assert parts["test"].behavior_label.eq("unknown_like").sum() == 1


def test_missing_time_fails_closed():
    with pytest.raises(ValueError, match="timestamps"):
        four_way_split(frame().drop(columns="timestamp"))


def test_held_out_subtype_is_unknown_even_when_behavior_is_known():
    rows = frame()
    rows["raw_attack"] = rows.behavior_label
    rows.loc[[3, 93], "raw_attack"] = "novel_scan"
    parts = four_way_split(rows, held_out_attacks=["novel_scan"])
    assert "3" not in set(parts["train"].row_uid)
    novel = parts["test"].loc[parts["test"].row_uid.eq("93")].iloc[0]
    assert novel.behavior_label == "unknown_like"
    assert novel.original_behavior_label == "scan_like"


def test_focal_weight_scale_invariant():
    logits = torch.tensor([[1., 2.], [3., -1.], [-1., 1.]])
    target = torch.tensor([0, 1, 1])
    weights = torch.tensor([1., 4.])
    torch.testing.assert_close(FocalLoss(weights)(logits, target),
                               FocalLoss(weights * 100)(logits, target))
    torch.testing.assert_close(FocalLoss(weights, gamma=0)(logits, target),
                               torch.nn.functional.cross_entropy(logits, target, weight=weights))
