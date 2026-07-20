from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.data import (
    LoadedDataset,
    _RowBudget,
    _read_csv_chunks,
    load_dataset,
    make_split,
)
from bitguard_bnn.demo import generate_demo
from bitguard_bnn.preprocess import FeaturePreprocessor


def demo_config(root: Path) -> dict:
    import copy

    config = copy.deepcopy(DEFAULTS)
    config["_project_root"] = str(root)
    config["_config_path"] = str(root / "config.yaml")
    config["dataset"].update(
        {
            "type": "csv",
            "path": "demo.csv",
            "label_column": "behavior_label",
            "raw_attack_column": "raw_attack",
            "device_column": "device_id",
            "time_column": "timestamp",
        }
    )
    config["split"].update(
        {"strategy": "attack", "held_out_attacks": ["novel_lowrate"], "seed": 2309}
    )
    config["preprocess"].update(
        {"feature_budget": 16, "selection": "cost_aware", "encoder": "thermometer", "thermometer_bits": 2}
    )
    return config


class DataAndPreprocessTest(unittest.TestCase):
    def test_chunk_reader_concatenates_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            pd.DataFrame({"value": range(9)}).to_csv(path, index=False)
            real_concat = pd.concat
            with patch("bitguard_bnn.data.pd.concat", wraps=real_concat) as concat:
                result = _read_csv_chunks(path, chunk_size=2, max_rows=None, seed=2309)
            self.assertEqual(len(result), 9)
            self.assertEqual(concat.call_count, 1)

    def test_dataset_loaded_row_limit_fails_before_unbounded_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_demo(root / "demo.csv", rows=1_000, seed=2309)
            config = demo_config(root)
            config["dataset"]["max_loaded_rows"] = 100
            with self.assertRaisesRegex(MemoryError, "max_loaded_rows"):
                load_dataset(config)

    def test_loaded_row_limit_stops_consuming_chunks_early(self) -> None:
        yielded = 0

        class Chunks:
            closed = False

            def __iter__(self):
                nonlocal yielded
                for index in range(5):
                    yielded += 1
                    yield pd.DataFrame({"value": [index]})

            def close(self):
                self.closed = True

        chunks = Chunks()

        with (
            patch("bitguard_bnn.data.pd.read_csv", return_value=chunks),
            self.assertRaisesRegex(MemoryError, "max_loaded_rows"),
        ):
            _read_csv_chunks(
                Path("unused.csv"),
                chunk_size=1,
                max_rows=None,
                seed=2309,
                row_budget=_RowBudget(2),
            )
        self.assertEqual(yielded, 3)
        self.assertTrue(chunks.closed)

    def test_attack_split_has_no_unknown_train_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_demo(root / "demo.csv", rows=2_000, seed=2309)
            config = demo_config(root)
            dataset = load_dataset(config)
            split = make_split(dataset, config)
            self.assertNotIn("novel_lowrate", set(split.train["raw_attack"]))
            self.assertIn("unknown_like", set(split.test["behavior_label"]))
            self.assertEqual(split.manifest["row_uid_overlap"]["train_test"], 0)

    def test_train_only_preprocessor_and_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_demo(root / "demo.csv", rows=2_000, seed=2309)
            config = demo_config(root)
            dataset = load_dataset(config)
            split = make_split(dataset, config)
            processor = FeaturePreprocessor(config).fit(split.train, dataset.feature_columns)
            transformed = processor.transform(split.validation)
            self.assertEqual(transformed.shape[1], 32)
            self.assertEqual(len(processor.selected_features), 16)
            self.assertNotIn("unknown_like", processor.active_labels)
            self.assertTrue(np.isfinite(transformed).all())

    def test_all_nan_feature_is_removed_without_name_shift(self) -> None:
        config = demo_config(Path("."))
        config["preprocess"].update({"feature_budget": 1, "encoder": "sign", "selection": "f_score"})
        train = pd.DataFrame(
            {
                "empty": [np.nan] * 8,
                "signal": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
                "behavior_label": ["benign"] * 4 + ["scan_like"] * 4,
            }
        )
        processor = FeaturePreprocessor(config).fit(train, ["empty", "signal"])
        self.assertEqual(processor.candidate_features, ["signal"])
        self.assertEqual(processor.selected_features, ["signal"])
        self.assertEqual(processor.transform(train).shape, (8, 1))

    def test_sequence_split_is_contiguous_per_source(self) -> None:
        config = demo_config(Path("."))
        config["split"]["strategy"] = "sequence"
        config["dataset"]["max_rows_per_file"] = None
        config["dataset"]["max_rows_per_class"] = None
        rows = []
        for source, label in (("capture_a", "benign"), ("capture_b", "scan_like")):
            for index in range(40):
                rows.append(
                    {
                        "row_uid": f"{source}-{index}",
                        "source_file": source,
                        "sequence_index": index,
                        "device_id": "device",
                        "raw_attack": label,
                        "behavior_label": label,
                        "timestamp": np.nan,
                        "dataset": "fixture",
                        "x": float(index),
                    }
                )
        dataset = LoadedDataset(
            pd.DataFrame(rows), ["x"], {"has_wall_clock_time": False}
        )
        split = make_split(dataset, config)
        counts = split.test.groupby("source_file").size().to_dict()
        self.assertEqual(counts, {"capture_a": 6, "capture_b": 6})
        self.assertGreaterEqual(split.test.groupby("source_file")["sequence_index"].min().min(), 34)
        self.assertTrue(split.manifest["temporal_continuity"])


if __name__ == "__main__":
    unittest.main()
