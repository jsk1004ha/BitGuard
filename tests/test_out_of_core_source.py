from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.data import (
    load_botiot,
    load_dataset,
    load_generic_csv,
    load_nbaiot,
)
from bitguard_bnn.out_of_core.source import iter_normalized_chunks


def _base_config(root: Path, dataset_type: str, path: str) -> dict:
    config = copy.deepcopy(DEFAULTS)
    config["_project_root"] = str(root)
    config["_config_path"] = str(root / "config.yaml")
    config["dataset"].update(
        {
            "type": dataset_type,
            "path": path,
            "chunk_size": 2,
            "max_rows_per_file": None,
            "max_rows_per_class": None,
            "max_loaded_rows": None,
            "drop_columns": [],
            "label_map": {},
        }
    )
    config["experiment"]["seed"] = 2309
    return config


class NormalizedSourceIteratorTest(unittest.TestCase):
    def _assert_parity(self, config: dict) -> None:
        legacy_loaders = {
            "csv": load_generic_csv,
            "nbaiot": load_nbaiot,
            "botiot": load_botiot,
        }
        legacy = legacy_loaders[config["dataset"]["type"]](config)
        loaded = load_dataset(config)
        expected = legacy.frame.sort_values("row_uid").reset_index(drop=True)
        materialized = (
            loaded.frame.sort_values("row_uid").reset_index(drop=True)
        )
        chunks = list(iter_normalized_chunks(config))
        actual = pd.concat([chunk.frame for chunk in chunks], ignore_index=True)
        actual = actual.sort_values("row_uid").reset_index(drop=True)
        pd.testing.assert_frame_equal(materialized[expected.columns], expected)
        pd.testing.assert_frame_equal(actual[expected.columns], expected)
        self.assertEqual(loaded.feature_columns, legacy.feature_columns)
        self.assertEqual(loaded.provenance, legacy.provenance)

    def test_generic_csv_chunks_match_in_memory_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "behavior_label": ["benign", "scan_like", "benign", "scan_like"],
                    "raw_attack": ["benign", "scan", "benign", "scan"],
                    "device_id": ["a", "a", "b", "b"],
                    "timestamp": [1, 2, 3, 4],
                    "x": [0.5, 1.5, 2.5, 3.5],
                }
            ).to_csv(root / "generic.csv", index=False)
            config = _base_config(root, "csv", "generic.csv")

            self._assert_parity(config)

    def test_nbaiot_chunks_match_in_memory_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "nbaiot"
            (dataset / "device_a").mkdir(parents=True)
            (dataset / "device_b" / "scan_attacks").mkdir(parents=True)
            pd.DataFrame({"f1": [1, 2, 3], "f2": [0.1, 0.2, 0.3]}).to_csv(
                dataset / "device_a" / "benign.csv", index=False
            )
            pd.DataFrame({"f1": [4, 5], "f2": [0.4, 0.5]}).to_csv(
                dataset / "device_b" / "scan_attacks" / "scan.csv", index=False
            )
            config = _base_config(root, "nbaiot", "nbaiot")

            self._assert_parity(config)

    def test_botiot_chunks_match_in_memory_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "category": ["Normal", "DDoS", "Normal", "Reconnaissance"],
                    "subcategory": ["Normal", "TCP", "Normal", "Service_Scan"],
                    "saddr": ["a", "a", "b", "b"],
                    "stime": [1, 2, 3, 4],
                    "rate": [0.1, 5.0, 0.2, 3.0],
                }
            ).to_csv(root / "botiot.csv", index=False)
            config = _base_config(root, "botiot", "botiot.csv")

            self._assert_parity(config)

    def test_iterator_does_not_concat_source_chunks_and_reports_exact_offsets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "behavior_label": ["benign"] * 5,
                    "raw_attack": ["benign"] * 5,
                    "device_id": ["device"] * 5,
                    "timestamp": range(5),
                    "x": range(5),
                }
            ).to_csv(root / "rows.csv", index=False)
            config = _base_config(root, "csv", "rows.csv")

            with patch(
                "bitguard_bnn.data.pd.concat",
                side_effect=AssertionError("iterator concatenated source chunks"),
            ):
                chunks = list(iter_normalized_chunks(config))

            self.assertEqual([chunk.source_row_start for chunk in chunks], [0, 2, 4])
            self.assertEqual(
                [chunk.source_relative_path for chunk in chunks],
                ["rows.csv", "rows.csv", "rows.csv"],
            )
            self.assertEqual(
                [chunk.frame["sequence_index"].tolist() for chunk in chunks],
                [[0, 1], [2, 3], [4]],
            )

    def test_sampling_caps_match_loader_and_uncapped_ignores_only_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "behavior_label": ["benign", "scan_like"] * 6,
                    "raw_attack": ["benign", "scan"] * 6,
                    "device_id": ["device"] * 12,
                    "timestamp": range(12),
                    "x": range(12),
                }
            ).to_csv(root / "rows.csv", index=False)
            config = _base_config(root, "csv", "rows.csv")
            config["dataset"].update(
                {
                    "max_rows_per_file": 8,
                    "max_rows_per_class": 3,
                    "max_loaded_rows": 100,
                }
            )

            expected_uids = set(load_generic_csv(config).frame["row_uid"])
            capped_uids = {
                uid
                for chunk in iter_normalized_chunks(config)
                for uid in chunk.frame["row_uid"]
            }
            uncapped = list(
                iter_normalized_chunks(config, apply_sampling_caps=False)
            )

            self.assertEqual(capped_uids, expected_uids)
            self.assertEqual(sum(len(chunk.frame) for chunk in uncapped), 12)

            config["dataset"]["max_loaded_rows"] = 5
            with self.assertRaisesRegex(MemoryError, "max_loaded_rows"):
                list(iter_normalized_chunks(config))
            uncapped = list(
                iter_normalized_chunks(config, apply_sampling_caps=False)
            )
            self.assertEqual(sum(len(chunk.frame) for chunk in uncapped), 12)

    def test_uncapped_iterator_still_enforces_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"x": [1, 2]}).to_csv(root / "rows.csv", index=False)
            config = _base_config(root, "csv", "rows.csv")

            with self.assertRaisesRegex(ValueError, "label column"):
                list(iter_normalized_chunks(config, apply_sampling_caps=False))


if __name__ == "__main__":
    unittest.main()
