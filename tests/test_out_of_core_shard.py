from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
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


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            raise unittest.SkipTest(f"directory symlinks unavailable: {exc}") from exc
        created = subprocess.run(
            ["cmd", "/d", "/c", f"mklink /J {link} {target}"],
            capture_output=True,
            check=False,
            text=True,
        )
        if created.returncode != 0:
            raise unittest.SkipTest(
                "directory links unavailable: "
                f"{created.stdout} {created.stderr}"
            ) from exc


def _remove_directory_link(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rmdir()


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

    def test_manifest_distinguishes_selected_and_boolean_materialized_features(self) -> None:
        from bitguard_bnn.out_of_core.shard import (
            verify_shard_manifest,
            write_parquet_shards,
        )

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            plan = write_parquet_shards(
                _chunks(rows),
                split,
                ("f1",),
                root / "prepared",
                dataset_name="fixture",
                preprocessing_fingerprint="preprocess-v2",
                materialized_features=("f1", "f2"),
                boolean_fast_path_features=("f2", "not_in_source"),
                missing_boolean_fast_path_features=("not_in_source",),
                shard_target_rows=5,
                max_rows_per_run=6,
                merge_fan_in=2,
                merge_read_rows=3,
            )
            manifest = verify_shard_manifest(plan.manifest_path, split_plan=split)
            self.assertEqual(manifest["selected_features"], ["f1"])
            self.assertEqual(manifest["materialized_features"], ["f1", "f2"])
            self.assertEqual(
                manifest["boolean_fast_path"],
                {
                    "configured_features": ["f2", "not_in_source"],
                    "available_features": ["f2"],
                    "missing_features": ["not_in_source"],
                },
            )
            first = plan.manifest_path.parent / manifest["entries"][0]["path"]
            self.assertEqual(pq.ParquetFile(first).schema_arrow.names[-2:], ["f1", "f2"])

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
            self.assertLessEqual(
                int(manifest["resource_usage"]["max_merge_input_rows_buffered"]),
                int(manifest["resource_usage"]["merge_input_rows_buffered_limit"]),
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

    def test_unsigned_resource_claim_tamper_breaks_manifest_fingerprint(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            payload = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
            payload["resource_usage"]["temporary_bytes_peak"] += 1
            plan.manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                shard_module.verify_shard_manifest(
                    plan.manifest_path, split_plan=split
                )

    def test_resigned_invalid_resource_claims_are_rejected_fail_closed(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            original = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
            shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)
            cases = (
                ("boolean run count", "run_count", True),
                (
                    "run maximum exceeds coverage",
                    "max_run_rows",
                    int(original["coverage"]["rows"]) + 1,
                ),
                ("negative temporary bytes", "temporary_bytes_peak", -1),
            )
            for name, field, value in cases:
                payload = json.loads(json.dumps(original))
                payload["resource_usage"][field] = value
                payload["fingerprint"] = shard_module.stable_fingerprint(
                    shard_module._manifest_semantics(payload)
                )
                plan.manifest_path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.subTest(name=name), self.assertRaisesRegex(
                    RuntimeError, "resource"
                ):
                    shard_module.verify_shard_manifest(
                        plan.manifest_path, split_plan=split
                    )

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

    def test_attack_relabel_rejects_replayed_source_outside_declared_heldout_group(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root, "attack")
            replayed = [dict(row) for row in rows]
            heldout = next(
                row for row in replayed if row["raw_attack"] == "scan_like"
            )
            heldout["raw_attack"] = "flood_like"
            with self.assertRaisesRegex(RuntimeError, "held-out attack|raw_attack"):
                write_parquet_shards(
                    _chunks(replayed, 9),
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
            membership = split.membership_path
            moved = membership.with_name("membership-handle-check.parquet")
            membership.rename(moved)
            moved.rename(membership)
            self.assertFalse(any((root / "prepared").rglob("*.partial")))

    def test_shard_path_rejects_linked_parent_and_resolved_root_escape(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "prepared"
            root.mkdir()
            outside = parent / "outside"
            shard = outside / "split=train" / "label=benign" / "part-00000000.parquet"
            shard.parent.mkdir(parents=True)
            shard.write_bytes(b"fixture")
            linked = root / "linked"
            _make_directory_link(linked, outside)
            relative = (
                "linked/split=train/label=benign/part-00000000.parquet"
            )
            with self.assertRaisesRegex(RuntimeError, "link|unsafe"):
                shard_module._safe_entry_path(root, relative)
            junction_name = "is_junction"
            junction_patch = (
                patch.object(Path, junction_name, return_value=False)
                if hasattr(Path, junction_name)
                else nullcontext()
            )
            with (
                patch.object(Path, "is_symlink", return_value=False),
                junction_patch,
                self.assertRaisesRegex(RuntimeError, "escape|beneath|root"),
            ):
                shard_module._safe_entry_path(root, relative)
            self.assertEqual(shard.read_bytes(), b"fixture")
            _remove_directory_link(linked)

    @unittest.skipUnless(os.name == "nt", "Windows native separator regression")
    def test_safe_entry_path_rejects_hidden_native_separator_junction(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "prepared"
            canonical_relative = (
                "dataset=fixture/split=train/label=benign/part-00000000.parquet"
            )
            canonical = root.joinpath(*canonical_relative.split("/"))
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical")
            outside = parent / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"outside")
            target = root / "target"
            hidden_shard = (
                target
                / "child"
                / "split=train"
                / "label=benign"
                / "part-00000000.parquet"
            )
            hidden_shard.parent.mkdir(parents=True)
            hidden_shard.write_bytes(b"hidden")
            links = root / "links"
            links.mkdir()
            alias = links / "alias"
            _make_directory_link(alias, target)
            hidden = (
                "dataset=fixture\\..\\links\\alias\\child/"
                "split=train/label=benign/part-00000000.parquet"
            )
            try:
                self.assertEqual(
                    shard_module._safe_entry_path(root, canonical_relative),
                    canonical.resolve(strict=True),
                )
                with self.assertRaisesRegex(RuntimeError, "canonical"):
                    shard_module._safe_entry_path(root, hidden)
                self.assertEqual(sentinel.read_bytes(), b"outside")
            finally:
                _remove_directory_link(alias)

    def test_safe_entry_path_rejects_noncanonical_drive_and_control_forms(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical_relative = (
                "dataset=fixture/split=train/label=benign/part-00000000.parquet"
            )
            canonical = root.joinpath(*canonical_relative.split("/"))
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical")
            self.assertEqual(
                shard_module._safe_entry_path(root, canonical_relative),
                canonical.resolve(strict=True),
            )
            invalid = (
                canonical_relative.replace("/", "//", 1),
                canonical_relative.replace("/split=", "/./split=", 1),
                canonical_relative + "/",
                "C:/outside/part-00000000.parquet",
                "dataset=fixture/\x1f/split=train/part-00000000.parquet",
                canonical_relative.replace("/", "\\", 1),
            )
            for value in invalid:
                with self.subTest(value=repr(value)), self.assertRaisesRegex(
                    RuntimeError, "canonical"
                ):
                    shard_module._safe_entry_path(root, value)

    @unittest.skipUnless(os.name == "nt", "Windows native separator regression")
    def test_manifest_rejects_recomputed_dataset_with_hidden_native_traversal(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            split, plan = self._write(rows, parent)
            shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)
            prepared = plan.manifest_path.parent
            outside = parent / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_bytes(b"outside")
            payload = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
            original_dataset = str(payload["dataset"])
            target = prepared / "target"
            shutil.copytree(
                prepared / f"dataset={original_dataset}", target / "child"
            )
            links = prepared / "links"
            links.mkdir()
            alias = links / "alias"
            _make_directory_link(alias, target)
            hidden_dataset = f"{original_dataset}\\..\\links\\alias\\child"
            payload["dataset"] = hidden_dataset
            original_prefix = f"dataset={original_dataset}/"
            hidden_prefix = f"dataset={hidden_dataset}/"
            for entry in payload["entries"]:
                entry["path"] = str(entry["path"]).replace(
                    original_prefix, hidden_prefix, 1
                )
            payload["fingerprint"] = shard_module.stable_fingerprint(
                shard_module._manifest_semantics(payload)
            )
            plan.manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "dataset"):
                    shard_module.verify_shard_manifest(
                        plan.manifest_path, split_plan=split
                    )
                self.assertEqual(sentinel.read_bytes(), b"outside")
            finally:
                _remove_directory_link(alias)

    def test_large_row_group_is_read_in_bounded_batches_and_closes_on_close(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        schema = pa.schema(
            [
                pa.field("row_uid", pa.string(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("behavior_label", pa.string(), nullable=False),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "one-large-row-group.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "row_uid": f"{index:064x}",
                            "split": "train",
                            "behavior_label": "benign",
                        }
                        for index in range(41)
                    ],
                    schema=schema,
                ),
                path,
                row_group_size=41,
            )
            real_parquet_file = pq.ParquetFile
            observed: list[int] = []

            class InstrumentedParquetFile:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self._delegate = real_parquet_file(*args, **kwargs)

                def iter_batches(self, *args: Any, **kwargs: Any):
                    self.assert_batch_size(kwargs.get("batch_size"))
                    for batch in self._delegate.iter_batches(*args, **kwargs):
                        observed.append(batch.num_rows)
                        yield batch

                @staticmethod
                def assert_batch_size(value: object) -> None:
                    if value != 3:
                        raise AssertionError(f"unbounded batch size: {value}")

            tracker = shard_module._ResourceTracker(root, 100, 3)
            with patch.object(shard_module.pq, "ParquetFile", InstrumentedParquetFile):
                iterator = shard_module._iter_records(path, 3, tracker=tracker)
                first = next(iterator)
                iterator.close()
            self.assertEqual(first["row_uid"], f"{0:064x}")
            self.assertTrue(observed)
            self.assertLessEqual(max(observed), 3)
            self.assertLessEqual(tracker.max_merge_input_rows_buffered, 3)
            moved = root / "moved.parquet"
            path.rename(moved)
            moved.rename(path)

    def test_membership_replacement_after_validation_aborts_cleanly_and_retries(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            membership = split.membership_path
            original_bytes = membership.read_bytes()
            table = pq.read_table(membership)
            swapped_rows = table.to_pylist()
            swap: tuple[int, int] | None = None
            for left in range(len(swapped_rows)):
                for right in range(left + 1, len(swapped_rows)):
                    if swapped_rows[left]["split"] != swapped_rows[right]["split"]:
                        swap = (left, right)
                        break
                if swap is not None:
                    break
            self.assertIsNotNone(swap)
            assert swap is not None
            left, right = swap
            swapped_rows[left]["split"], swapped_rows[right]["split"] = (
                swapped_rows[right]["split"],
                swapped_rows[left]["split"],
            )
            replacement = root / "replacement-membership.parquet"
            pq.write_table(
                pa.Table.from_pylist(swapped_rows, schema=table.schema),
                replacement,
                compression="zstd",
            )
            real_validate = shard_module._validate_split_plan

            def replace_after_validation(plan: object):
                manifest = real_validate(plan)
                os.replace(replacement, membership)
                return manifest

            prepared = root / "prepared"

            def write():
                return shard_module.write_parquet_shards(
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

            try:
                with patch.object(
                    shard_module,
                    "_validate_split_plan",
                    side_effect=replace_after_validation,
                ):
                    with self.assertRaisesRegex(RuntimeError, "membership|checksum"):
                        write()
            finally:
                membership.write_bytes(original_bytes)
            self.assertFalse((prepared / "shard_manifest.json").exists())
            self.assertEqual(list(prepared.rglob("*.parquet")), [])
            self.assertEqual(list(prepared.rglob("*.partial")), [])
            retried = write()
            shard_module.verify_shard_manifest(retried.manifest_path, split_plan=split)

    def test_verifier_membership_replacement_after_validation_aborts_and_retries(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            membership = split.membership_path
            original_membership = membership.read_bytes()
            replacement = root / "reencoded-membership.parquet"
            table = pq.read_table(membership)
            pq.write_table(table, replacement, compression="gzip", row_group_size=2)
            self.assertNotEqual(replacement.read_bytes(), original_membership)
            manifest_bytes = plan.manifest_path.read_bytes()
            shard_bytes = {
                str(entry["path"]): (plan.manifest_path.parent / entry["path"]).read_bytes()
                for entry in json.loads(manifest_bytes)["entries"]
            }
            sentinel = root / "outside-sentinel.txt"
            sentinel.write_bytes(b"outside")
            real_validate = shard_module._validate_split_plan

            def replace_after_validation(candidate: object):
                manifest = real_validate(candidate)
                os.replace(replacement, membership)
                return manifest

            try:
                with patch.object(
                    shard_module,
                    "_validate_split_plan",
                    side_effect=replace_after_validation,
                ):
                    with self.assertRaisesRegex(RuntimeError, "membership|checksum"):
                        shard_module.verify_shard_manifest(
                            plan.manifest_path, split_plan=split
                        )
            finally:
                membership.write_bytes(original_membership)
            self.assertEqual(plan.manifest_path.read_bytes(), manifest_bytes)
            for relative, expected in shard_bytes.items():
                self.assertEqual(
                    (plan.manifest_path.parent / relative).read_bytes(), expected
                )
            self.assertEqual(sentinel.read_bytes(), b"outside")
            self.assertEqual(
                list(plan.manifest_path.parent.glob(".verify-shards-*.partial")), []
            )
            shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)

    def test_verifier_shard_swap_after_hash_uses_no_unverified_bytes(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            original_manifest = plan.manifest_path.read_bytes()
            payload = json.loads(original_manifest)
            entry = payload["entries"][0]
            shard = plan.manifest_path.parent / entry["path"]
            original_shard = shard.read_bytes()
            marker = original_shard.rfind(b"parquet-cpp-arrow")
            self.assertGreaterEqual(marker, 0)
            first = bytearray(original_shard)
            second = bytearray(original_shard)
            first[marker] = ord("q")
            second[marker] = ord("r")
            first_bytes = bytes(first)
            second_bytes = bytes(second)
            shard.write_bytes(first_bytes)
            replacement = root / "replacement-shard.parquet"
            replacement.write_bytes(second_bytes)
            self.assertEqual(pq.read_table(shard).num_rows, int(entry["rows"]))
            self.assertEqual(
                pq.read_table(replacement).num_rows, int(entry["rows"])
            )
            entry["sha256"] = hashlib.sha256(first_bytes).hexdigest()
            payload["fingerprint"] = shard_module.stable_fingerprint(
                shard_module._manifest_semantics(payload)
            )
            forged_manifest = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            plan.manifest_path.write_bytes(forged_manifest)
            sentinel = root / "outside-sentinel.txt"
            sentinel.write_bytes(b"outside")
            real_sha256 = shard_module._sha256_file
            replaced = False
            observed_after_failure: bytes | None = None

            def replace_after_hash(candidate: Path) -> str:
                nonlocal replaced
                digest = real_sha256(candidate)
                if not replaced and candidate.resolve() == shard.resolve():
                    os.replace(replacement, shard)
                    replaced = True
                return digest

            try:
                with patch.object(
                    shard_module, "_sha256_file", side_effect=replace_after_hash
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "snapshot|checksum|identity"
                    ):
                        shard_module.verify_shard_manifest(
                            plan.manifest_path, split_plan=split
                        )
                observed_after_failure = shard.read_bytes()
                self.assertEqual(plan.manifest_path.read_bytes(), forged_manifest)
                self.assertEqual(sentinel.read_bytes(), b"outside")
                self.assertEqual(
                    list(plan.manifest_path.parent.glob(".verify-shards-*.partial")),
                    [],
                )
            finally:
                shard.write_bytes(original_shard)
                plan.manifest_path.write_bytes(original_manifest)
            self.assertEqual(observed_after_failure, second_bytes)
            shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)

    def test_manifest_publication_failure_rolls_back_only_owned_shards(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

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
                    shard_module.write_parquet_shards(
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
            self.assertFalse((prepared / "dataset=fixture").exists())
            retried = shard_module.write_parquet_shards(
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
            shard_module.verify_shard_manifest(retried.manifest_path, split_plan=split)

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
