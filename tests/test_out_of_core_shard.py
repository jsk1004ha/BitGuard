from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from bitguard_bnn.out_of_core.source import NormalizedChunk


def _rows(count: int = 36) -> list[dict[str, object]]:
    labels = ("benign", "scan_like", "flood_like")
    rows: list[dict[str, object]] = []
    for index in range(count):
        source = f"device-{index % 3}/part.csv"
        label = labels[index % len(labels)]
        rows.append(
            {
                "row_uid": f"{index:064x}",
                "dataset": "fixture",
                "source_file": source,
                "sequence_index": index // 3,
                "device_id": f"device-{index % 3}",
                "raw_attack": "normal" if label == "benign" else label,
                "behavior_label": label,
                "timestamp": float(index),
                "f1": float(index),
                "f2": float(index % 7),
                "unused": float(index * 100),
            }
        )
    return rows


def _chunks(rows: list[dict[str, object]], size: int = 5) -> list[NormalizedChunk]:
    chunks: list[NormalizedChunk] = []
    for start in range(0, len(rows), size):
        group = rows[start : start + size]
        chunks.append(
            NormalizedChunk(
                frame=pd.DataFrame(group),
                source_relative_path=str(group[0]["source_file"]),
                source_row_start=int(str(group[0]["sequence_index"])),
            )
        )
    return chunks


def _split(
    rows: list[dict[str, object]], root: Path, strategy: str = "random"
):
    from bitguard_bnn.out_of_core.split import build_split_plan

    return build_split_plan(
        _chunks(rows, 4),
        {
            "dataset": {"path": "ignored"},
            "split": {
                "strategy": strategy,
                "train_fraction": 0.5,
                "validation_fraction": 0.25,
                "test_fraction": 0.25,
                "seed": 11,
                "held_out_devices": [],
                "held_out_attacks": ["scan_like"] if strategy == "attack" else [],
            },
        },
        root / "split",
        max_rows_per_run=3,
        merge_fan_in=2,
    )


class OutOfCoreShardTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, object]], root: Path, **overrides: Any):
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        split = _split(rows, root)
        plan = write_parquet_shards(
            _chunks(list(reversed(rows)), 7),
            split,
            ("f1", "f2"),
            root / "prepared",
            dataset_name="fixture data",
            preprocessing_fingerprint="preprocess-v1",
            shard_target_rows=3,
            max_rows_per_run=4,
            merge_fan_in=2,
            **overrides,
        )
        return split, plan

    def test_writes_exact_partitioned_coverage_and_semantic_manifest(self) -> None:
        from bitguard_bnn.out_of_core.shard import verify_shard_manifest

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            manifest = verify_shard_manifest(plan.manifest_path, split_plan=split)
            entries = manifest["entries"]
            observed: list[str] = []
            by_bucket: dict[tuple[str, str], list[int]] = {}
            for entry in entries:
                path = plan.manifest_path.parent / entry["path"]
                frame = pq.read_table(path).to_pandas()
                observed.extend(frame["row_uid"].astype(str))
                self.assertNotIn("unused", frame.columns)
                self.assertEqual(
                    list(pq.ParquetFile(path).schema_arrow.names),
                    [
                        "row_uid",
                        "source_file",
                        "sequence_index",
                        "device_id",
                        "raw_attack",
                        "behavior_label",
                        "timestamp",
                        "f1",
                        "f2",
                    ],
                )
                self.assertTrue((frame["dataset"] == "fixture_data").all())
                self.assertTrue((frame["split"] == entry["split"]).all())
                self.assertTrue((frame["behavior_label"] == entry["label"]).all())
                by_bucket.setdefault((entry["split"], entry["label"]), []).append(
                    int(entry["rows"])
                )
            self.assertEqual(sorted(observed), sorted(str(row["row_uid"]) for row in rows))
            self.assertEqual(sum(int(entry["rows"]) for entry in entries), len(rows))
            for sizes in by_bucket.values():
                self.assertTrue(all(size == 3 for size in sizes[:-1]))
                self.assertLessEqual(sizes[-1], 3)
            self.assertEqual(manifest["preprocessing_fingerprint"], "preprocess-v1")
            self.assertEqual(manifest["split_fingerprint"], split.fingerprint)
            self.assertEqual(len(manifest["fingerprint"]), 64)
            self.assertLessEqual(
                int(manifest["resource_usage"]["max_merge_fan_in_observed"]), 2
            )
            self.assertEqual(
                int(manifest["shard_contract"]["merge_read_rows"]), 1024
            )
            self.assertEqual(
                manifest["source_coverage"],
                {f"device-{index}/part.csv": 12 for index in range(3)},
            )

    def test_idempotent_reuse_requires_manifest_and_all_hashes(self) -> None:
        from bitguard_bnn.out_of_core.shard import verify_shard_manifest

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, first = self._write(rows, root)
            _, second = self._write(rows, root)
            self.assertEqual(first.fingerprint, second.fingerprint)
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            shard = first.manifest_path.parent / manifest["entries"][0]["path"]
            original = shard.read_bytes()
            shard.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                verify_shard_manifest(first.manifest_path, split_plan=split)
            with self.assertRaisesRegex(RuntimeError, "immutable|checksum"):
                self._write(rows, root)

    def test_missing_extra_and_duplicate_uids_are_rejected(self) -> None:
        rows = _rows()
        cases: list[tuple[str, list[dict[str, object]], str]] = []
        missing = [dict(row) for row in rows[:-1]]
        cases.append(("missing", missing, "missing|coverage"))
        extra = [dict(row) for row in rows]
        extra.append({**dict(rows[-1]), "row_uid": "f" * 64, "sequence_index": 999})
        cases.append(("extra", extra, "extra|coverage"))
        duplicate = [dict(row) for row in rows]
        duplicate[-1]["row_uid"] = duplicate[0]["row_uid"]
        cases.append(("duplicate", duplicate, "duplicate"))
        mismatched_label = [dict(row) for row in rows]
        mismatched_label[-1]["behavior_label"] = "benign"
        cases.append(("label", mismatched_label, "behavior_label mismatch"))
        for name, source_rows, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                from bitguard_bnn.out_of_core.shard import write_parquet_shards

                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    write_parquet_shards(
                        _chunks(source_rows, 6),
                        split,
                        ("f1", "f2"),
                        root / "prepared",
                        dataset_name="fixture",
                        preprocessing_fingerprint="preprocess-v1",
                        shard_target_rows=3,
                        max_rows_per_run=3,
                        merge_fan_in=2,
                    )
                prepared = root / "prepared"
                self.assertFalse(any(prepared.rglob("*.partial")) if prepared.exists() else False)
                self.assertFalse((prepared / "shard_manifest.json").exists())

    def test_manifest_tamper_schema_drift_and_unlisted_artifacts_are_rejected(self) -> None:
        from bitguard_bnn.out_of_core.shard import verify_shard_manifest

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            payload = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
            payload["counts"]["train"] += 1
            plan.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                verify_shard_manifest(plan.manifest_path, split_plan=split)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            stray = (
                plan.manifest_path.parent
                / "dataset=fixture_data"
                / "split=train"
                / "label=benign"
                / "part-stray.parquet"
            )
            stray.parent.mkdir(parents=True, exist_ok=True)
            stray.write_bytes(b"stray")
            with self.assertRaisesRegex(RuntimeError, "unlisted"):
                verify_shard_manifest(plan.manifest_path, split_plan=split)

    def test_invalid_limits_and_feature_contract_fail_before_consuming_chunks(self) -> None:
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        class Unconsumed:
            def __iter__(self):
                raise AssertionError("chunks were consumed")

        with tempfile.TemporaryDirectory() as temp:
            split = _split(_rows(), Path(temp))
            for kwargs, message in (
                ({"shard_target_rows": 0}, "shard_target_rows"),
                ({"max_rows_per_run": 0}, "max_rows_per_run"),
                ({"merge_fan_in": 1}, "merge_fan_in"),
                ({"merge_read_rows": 0}, "merge_read_rows"),
            ):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                    write_parquet_shards(
                        Unconsumed(),
                        split,
                        ("f1", "f2"),
                        Path(temp) / "prepared",
                        dataset_name="fixture",
                        preprocessing_fingerprint="preprocess-v1",
                        **kwargs,
                    )
            with self.assertRaisesRegex(ValueError, "selected_features"):
                write_parquet_shards(
                    Unconsumed(),
                    split,
                    (),
                    Path(temp) / "prepared",
                    dataset_name="fixture",
                    preprocessing_fingerprint="preprocess-v1",
                )

    def test_attack_split_allows_only_its_declared_unknown_test_relabel(self) -> None:
        from bitguard_bnn.out_of_core.shard import (
            verify_shard_manifest,
            write_parquet_shards,
        )

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root, "attack")
            plan = write_parquet_shards(
                _chunks(rows, 9),
                split,
                ("f1", "f2"),
                root / "prepared",
                dataset_name="fixture",
                preprocessing_fingerprint="preprocess-v1",
                shard_target_rows=10,
                max_rows_per_run=12,
                merge_fan_in=2,
                merge_read_rows=4,
            )
            manifest = verify_shard_manifest(plan.manifest_path, split_plan=split)
        self.assertGreater(
            int(manifest["class_counts"]["test"].get("unknown_like", 0)), 0
        )

    def test_manifest_publication_failure_rolls_back_only_owned_shards(self) -> None:
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            with patch(
                "bitguard_bnn.out_of_core.shard._write_manifest_no_replace",
                side_effect=OSError("injected manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "manifest failure"):
                    write_parquet_shards(
                        _chunks(rows, 9),
                        split,
                        ("f1", "f2"),
                        prepared,
                        dataset_name="fixture",
                        preprocessing_fingerprint="preprocess-v1",
                        shard_target_rows=10,
                        max_rows_per_run=12,
                        merge_fan_in=2,
                        merge_read_rows=4,
                    )
            self.assertFalse((prepared / "shard_manifest.json").exists())
            self.assertEqual(list(prepared.rglob("*.parquet")), [])
            self.assertEqual(list(prepared.rglob("*.partial")), [])

    def test_unmanifested_existing_shard_is_never_replaced(self) -> None:
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            sentinel = (
                prepared
                / "dataset=fixture"
                / "split=train"
                / "label=benign"
                / "part-00000000.parquet"
            )
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"do-not-replace")
            with self.assertRaisesRegex(RuntimeError, "incomplete|unsafe"):
                write_parquet_shards(
                    _chunks(rows, 9),
                    split,
                    ("f1", "f2"),
                    prepared,
                    dataset_name="fixture",
                    preprocessing_fingerprint="preprocess-v1",
                    shard_target_rows=10,
                    max_rows_per_run=12,
                    merge_fan_in=2,
                    merge_read_rows=4,
                )
            self.assertEqual(sentinel.read_bytes(), b"do-not-replace")
            self.assertFalse((prepared / "shard_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
