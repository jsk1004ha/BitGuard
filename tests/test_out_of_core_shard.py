from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bitguard_bnn.out_of_core.manifest import SplitPlan
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
) -> SplitPlan:
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
                "directory links unavailable: " f"{created.stdout} {created.stderr}"
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

    def _run_crashing_builder(
        self,
        *,
        split: SplitPlan,
        prepared: Path,
        seam: str,
        exit_code: int,
        seam_body: str,
        cwd: Path | None = None,
        row_count: int = 36,
    ) -> subprocess.CompletedProcess[str]:
        repository = Path(__file__).resolve().parents[1]
        child = f"""
import os
from pathlib import Path
from bitguard_bnn.out_of_core import shard as shard_module
from bitguard_bnn.out_of_core.manifest import SplitPlan
from tests.test_out_of_core_shard import _chunks, _rows

split = SplitPlan(
    strategy={getattr(split, "strategy")!r},
    train_count={getattr(split, "train_count")},
    validation_count={getattr(split, "validation_count")},
    test_count={getattr(split, "test_count")},
    membership_path=Path({str(getattr(split, "membership_path"))!r}),
    fingerprint={getattr(split, "fingerprint")!r},
)
def crash_seam(*args):
{seam_body}
setattr(shard_module, {seam!r}, crash_seam)
shard_module.write_parquet_shards(
    _chunks(_rows({row_count}), 7), split, ("f1", "f2"), Path({str(prepared)!r}),
    dataset_name="fixture", preprocessing_fingerprint="preprocess-v1",
    shard_target_rows=3, record_batch_rows=2, max_rows_per_run=4,
    merge_fan_in=2, merge_read_rows=2,
)
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(repository / "src"), str(repository))
        )
        result = subprocess.run(
            [sys.executable, "-c", child],
            cwd=repository if cwd is None else cwd,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            exit_code,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        return result

    def _retry_crashed_build(
        self, rows: list[dict[str, object]], split: SplitPlan, prepared: Path
    ):
        from bitguard_bnn.out_of_core import shard as shard_module

        return shard_module.write_parquet_shards(
            _chunks(rows, 7),
            split,
            ("f1", "f2"),
            prepared,
            dataset_name="fixture",
            preprocessing_fingerprint="preprocess-v1",
            shard_target_rows=3,
            record_batch_rows=2,
            max_rows_per_run=4,
            merge_fan_in=2,
            merge_read_rows=2,
        )

    def test_manifest_distinguishes_selected_and_boolean_materialized_features(
        self,
    ) -> None:
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
            self.assertEqual(
                pq.ParquetFile(first).schema_arrow.names[-2:], ["f1", "f2"]
            )

    def test_recomputed_manifest_rejects_unknown_algorithm_versions(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(_rows(), root)
            payload = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
            payload["algorithm_versions"]["shards"] = "attacker-controlled-v99"
            payload["fingerprint"] = shard_module.stable_fingerprint(
                shard_module._manifest_semantics(payload)
            )
            plan.manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "algorithm"):
                shard_module.verify_shard_manifest(
                    plan.manifest_path,
                    split_plan=split,
                )

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
            self.assertEqual(
                sorted(observed), sorted(str(row["row_uid"]) for row in rows)
            )
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
            self.assertEqual(int(manifest["shard_contract"]["merge_read_rows"]), 1024)
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
                shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)

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
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(RuntimeError, "resource"),
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
                self.assertFalse(
                    any(prepared.rglob("*.partial")) if prepared.exists() else False
                )
                self.assertFalse((prepared / "shard_manifest.json").exists())

    def test_manifest_tamper_schema_drift_and_unlisted_artifacts_are_rejected(
        self,
    ) -> None:
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

    def test_invalid_limits_and_feature_contract_fail_before_consuming_chunks(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core.shard import write_parquet_shards

        class Unconsumed:
            def __iter__(self):
                raise AssertionError("chunks were consumed")

        with tempfile.TemporaryDirectory() as temp:
            split = _split(_rows(), Path(temp))
            invalid_cases: tuple[tuple[dict[str, Any], str], ...] = (
                ({"shard_target_rows": 0}, "shard_target_rows"),
                ({"max_rows_per_run": 0}, "max_rows_per_run"),
                ({"merge_fan_in": 1}, "merge_fan_in"),
                ({"merge_read_rows": 0}, "merge_read_rows"),
            )
            for kwargs, message in invalid_cases:
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, message),
                ):
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
            heldout = next(row for row in replayed if row["raw_attack"] == "scan_like")
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
            relative = "linked/split=train/label=benign/part-00000000.parquet"
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
                with (
                    self.subTest(value=repr(value)),
                    self.assertRaisesRegex(RuntimeError, "canonical"),
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
            shutil.copytree(prepared / f"dataset={original_dataset}", target / "child")
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

    def test_large_row_group_is_read_in_bounded_batches_and_closes_on_close(
        self,
    ) -> None:
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

            def replace_after_validation(plan: SplitPlan):
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
                str(entry["path"]): (
                    plan.manifest_path.parent / entry["path"]
                ).read_bytes()
                for entry in json.loads(manifest_bytes)["entries"]
            }
            sentinel = root / "outside-sentinel.txt"
            sentinel.write_bytes(b"outside")
            real_validate = shard_module._validate_split_plan

            def replace_after_validation(candidate: SplitPlan):
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
            self.assertEqual(pq.read_table(replacement).num_rows, int(entry["rows"]))
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

    def test_verification_cleanup_preserves_replacement_work_tree(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            replacement = root / "replacement-verification-work"
            replacement.mkdir()
            sentinel = replacement / "foreign-sentinel.bin"
            sentinel.write_bytes(b"preserve")
            displaced = root / "owned-verification-work"
            observed_work: Path | None = None

            def replace_before_cleanup(work: Path) -> None:
                nonlocal observed_work
                if observed_work is not None:
                    return
                observed_work = work
                work.replace(displaced)
                replacement.replace(work)

            with (
                patch.object(
                    shard_module,
                    "_before_verification_work_cleanup",
                    side_effect=replace_before_cleanup,
                ),
                self.assertRaisesRegex(RuntimeError, "verification work identity"),
            ):
                shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)

            self.assertIsNotNone(observed_work)
            assert observed_work is not None
            self.assertEqual((observed_work / sentinel.name).read_bytes(), b"preserve")
            self.assertTrue(displaced.is_dir())

    @unittest.skipUnless(os.name == "nt", "Windows handle cleanup regression")
    def test_verification_cleanup_preserves_post_quarantine_replacement(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split, plan = self._write(rows, root)
            replacement = root / "replacement-verification-quarantine"
            replacement.mkdir()
            sentinel = replacement / "foreign-sentinel.bin"
            sentinel.write_bytes(b"preserve")
            displaced = root / "owned-verification-quarantine"
            observed_quarantine: Path | None = None

            def replace_after_quarantine(work: Path, quarantine: Path) -> None:
                del work
                nonlocal observed_quarantine
                if observed_quarantine is not None:
                    return
                observed_quarantine = quarantine
                quarantine.replace(displaced)
                replacement.replace(quarantine)

            with (
                patch.object(
                    shard_module,
                    "_after_verification_work_quarantine_boundary",
                    side_effect=replace_after_quarantine,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "verification work quarantine identity"
                ),
            ):
                shard_module.verify_shard_manifest(plan.manifest_path, split_plan=split)

            self.assertIsNotNone(observed_quarantine)
            assert observed_quarantine is not None
            self.assertEqual(
                (observed_quarantine / sentinel.name).read_bytes(), b"preserve"
            )
            self.assertTrue(displaced.is_dir())

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

    def test_process_exit_after_parquet_write_recovers_owned_work_with_parity(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            repository = Path(__file__).resolve().parents[1]
            child = f"""
import os
from pathlib import Path
from bitguard_bnn.out_of_core import shard as shard_module
from bitguard_bnn.out_of_core.manifest import SplitPlan
from tests.test_out_of_core_shard import _chunks, _rows

split = SplitPlan(
    strategy={split.strategy!r},
    train_count={split.train_count},
    validation_count={split.validation_count},
    test_count={split.test_count},
    membership_path=Path({str(split.membership_path)!r}),
    fingerprint={split.fingerprint!r},
)
def hard_exit(_index, _partial):
    os._exit(87)
shard_module._after_staged_write_boundary = hard_exit
shard_module.write_parquet_shards(
    _chunks(_rows(), 7), split, ("f1", "f2"), Path({str(prepared)!r}),
    dataset_name="fixture", preprocessing_fingerprint="preprocess-v1",
    shard_target_rows=3, record_batch_rows=2, max_rows_per_run=4,
    merge_fan_in=2, merge_read_rows=2,
)
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repository / "src")
            interrupted = subprocess.run(
                [sys.executable, "-c", child],
                cwd=repository,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(
                interrupted.returncode,
                87,
                msg=f"stdout={interrupted.stdout}\nstderr={interrupted.stderr}",
            )
            stale = list(prepared.glob(".shards-*.partial"))
            self.assertEqual(len(stale), 1)
            self.assertTrue(
                any(
                    path.name.endswith(".parquet.partial")
                    for path in stale[0].rglob("*")
                )
            )
            self.assertTrue((prepared / ".shard-build-transaction.json").is_file())
            self.assertTrue((stale[0] / shard_module._SHARD_WORK_OWNER_FILE).is_file())

            # Matching bytes are not sufficient ownership proof: a replacement
            # directory entry must be preserved and rejected by instance.
            owner = stale[0] / shard_module._SHARD_WORK_OWNER_FILE
            displaced_owner = root / "displaced-owned-marker.json"
            replacement_owner = root / "same-bytes-foreign-marker.json"
            owner_bytes = owner.read_bytes()
            owner.replace(displaced_owner)
            replacement_owner.write_bytes(owner_bytes)
            replacement_owner.replace(owner)
            with self.assertRaisesRegex(RuntimeError, "owner marker identity"):
                shard_module.write_parquet_shards(
                    _chunks(rows, 7),
                    split,
                    ("f1", "f2"),
                    prepared,
                    dataset_name="fixture",
                    preprocessing_fingerprint="preprocess-v1",
                    shard_target_rows=3,
                    record_batch_rows=2,
                    max_rows_per_run=4,
                    merge_fan_in=2,
                    merge_read_rows=2,
                )
            self.assertEqual(owner.read_bytes(), owner_bytes)
            self.assertTrue(stale[0].is_dir())
            owner.unlink()
            displaced_owner.replace(owner)

            recovered = shard_module.write_parquet_shards(
                _chunks(rows, 7),
                split,
                ("f1", "f2"),
                prepared,
                dataset_name="fixture",
                preprocessing_fingerprint="preprocess-v1",
                shard_target_rows=3,
                record_batch_rows=2,
                max_rows_per_run=4,
                merge_fan_in=2,
                merge_read_rows=2,
            )
            clean = shard_module.write_parquet_shards(
                _chunks(rows, 7),
                split,
                ("f1", "f2"),
                root / "clean",
                dataset_name="fixture",
                preprocessing_fingerprint="preprocess-v1",
                shard_target_rows=3,
                record_batch_rows=2,
                max_rows_per_run=4,
                merge_fan_in=2,
                merge_read_rows=2,
            )
            self.assertEqual(recovered.fingerprint, clean.fingerprint)
            self.assertEqual(
                recovered.manifest_path.read_bytes(), clean.manifest_path.read_bytes()
            )
            self.assertFalse((prepared / ".shard-build-transaction.json").exists())
            self.assertEqual(list(prepared.glob(".shards-*.partial")), [])
            shard_module.verify_shard_manifest(
                recovered.manifest_path, split_plan=split, merge_fan_in=2
            )

    def test_recovery_rejects_same_byte_transaction_replacement(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_staged_write_boundary",
                exit_code=87,
                seam_body="    os._exit(87)",
            )
            journal = prepared / ".shard-build-transaction.json"
            displaced = root / "owned-journal.json"
            replacement = root / "replacement-journal.json"
            original = journal.read_bytes()
            journal.replace(displaced)
            replacement.write_bytes(original)
            replacement.replace(journal)
            with self.assertRaisesRegex(RuntimeError, "transaction.*identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(journal.read_bytes(), original)
            self.assertEqual(len(list(prepared.glob(".shards-*.partial"))), 1)
            journal.unlink()
            displaced.replace(journal)
            recovered = self._retry_crashed_build(rows, split, prepared)
            shard_module.verify_shard_manifest(
                recovered.manifest_path, split_plan=split, merge_fan_in=2
            )

    def test_every_public_link_boundary_recovers_with_parity(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            clean = self._retry_crashed_build(rows, split, root / "clean")
            entry_count = len(json.loads(clean.manifest_path.read_bytes())["entries"])
            for boundary in range(entry_count):
                prepared = root / f"prepared-{boundary}"
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam="_after_public_shard_link_boundary",
                    exit_code=88,
                    seam_body=(
                        f"    if args[0] == {boundary}:\n" "        os._exit(88)"
                    ),
                )
                self.assertTrue(any((prepared / "dataset=fixture").rglob("*.parquet")))
                recovered = self._retry_crashed_build(rows, split, prepared)
                self.assertEqual(
                    recovered.manifest_path.read_bytes(),
                    clean.manifest_path.read_bytes(),
                )
                self.assertFalse((prepared / ".shard-build-transaction.json").exists())
                self.assertEqual(list(prepared.glob(".shards-*.partial")), [])
                shard_module.verify_shard_manifest(
                    recovered.manifest_path, split_plan=split, merge_fan_in=2
                )

    def test_public_recovery_preserves_replaced_and_missing_prefixes(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)

            replaced_output = root / "replaced"
            self._run_crashing_builder(
                split=split,
                prepared=replaced_output,
                seam="_after_public_shard_link_boundary",
                exit_code=88,
                seam_body="    os._exit(88)",
            )
            published = next((replaced_output / "dataset=fixture").rglob("*.parquet"))
            displaced = root / "owned-public.parquet"
            replacement = root / "replacement-public.parquet"
            public_bytes = published.read_bytes()
            published.replace(displaced)
            replacement.write_bytes(public_bytes)
            replacement.replace(published)
            with self.assertRaisesRegex(RuntimeError, "published shard identity"):
                self._retry_crashed_build(rows, split, replaced_output)
            self.assertEqual(published.read_bytes(), public_bytes)

            missing_output = root / "missing"
            self._run_crashing_builder(
                split=split,
                prepared=missing_output,
                seam="_after_public_shard_link_boundary",
                exit_code=88,
                seam_body="    if args[0] == 1:\n        os._exit(88)",
            )
            public_paths = sorted(
                (missing_output / "dataset=fixture").rglob("*.parquet")
            )
            self.assertGreaterEqual(len(public_paths), 2)
            later_bytes = public_paths[1].read_bytes()
            public_paths[0].unlink()
            with self.assertRaisesRegex(RuntimeError, "published shard prefix"):
                self._retry_crashed_build(rows, split, missing_output)
            self.assertEqual(public_paths[1].read_bytes(), later_bytes)

    def test_cleanup_inventory_preserves_external_sentinel_after_crash(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_before_owned_shard_work_cleanup",
                exit_code=89,
                seam_body=(
                    "    (args[0] / 'external-sentinel.bin').write_bytes(b'foreign')\n"
                    "    os._exit(89)"
                ),
            )
            work = next(prepared.glob(".shards-*.partial"))
            sentinel = work / "external-sentinel.bin"
            with self.assertRaisesRegex(RuntimeError, "cleanup inventory"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(sentinel.read_bytes(), b"foreign")
            self.assertTrue(work.is_dir())
            sentinel.unlink()
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())

    def test_cleanup_root_rmdir_preserves_a_late_foreign_sentinel(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            inserted: list[Path] = []

            def insert_after_owned_entry_cleanup(work: Path) -> None:
                sentinel = work / "late-foreign-sentinel.bin"
                sentinel.write_bytes(b"foreign")
                inserted.append(sentinel)

            with patch.object(
                shard_module,
                "_before_owned_work_root_removal",
                side_effect=insert_after_owned_entry_cleanup,
                create=True,
            ):
                with self.assertRaises(OSError):
                    self._retry_crashed_build(rows, split, prepared)

            self.assertEqual(len(inserted), 1)
            self.assertEqual(inserted[0].read_bytes(), b"foreign")
            self.assertTrue(inserted[0].parent.is_dir())

    def test_transaction_temporary_initialization_boundaries_are_recoverable(
        self,
    ) -> None:
        rows = _rows()
        seams = (
            ("_after_transaction_temporary_intent_boundary", 96),
            ("_after_transaction_temporary_created_anchor_boundary", 97),
            ("_after_transaction_temporary_payload_boundary", 98),
        )
        for seam, exit_code in seams:
            with self.subTest(seam=seam), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                for _ in range(2):
                    self._run_crashing_builder(
                        split=split,
                        prepared=prepared,
                        seam=seam,
                        exit_code=exit_code,
                        seam_body=f"    os._exit({exit_code})",
                    )
                    self.assertLessEqual(
                        len(
                            list(
                                prepared.glob(
                                    "..shard-build-transaction.json.*.partial"
                                )
                            )
                        ),
                        1,
                    )
                recovered = self._retry_crashed_build(rows, split, prepared)
                self.assertTrue(recovered.manifest_path.is_file())
                self.assertEqual(
                    list(prepared.glob("..shard-build-transaction.json.*.partial")),
                    [],
                )

    def test_created_transaction_temporary_replacement_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_transaction_temporary_created_anchor_boundary",
                exit_code=97,
                seam_body="    os._exit(97)",
            )
            temporary = next(prepared.glob("..shard-build-transaction.json.*.partial"))
            displaced = root / "owned-empty-transaction.partial"
            replacement = root / "foreign-empty-transaction.partial"
            temporary.replace(displaced)
            replacement.write_bytes(b"")
            replacement.replace(temporary)
            with self.assertRaisesRegex(RuntimeError, "temporary identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(temporary.read_bytes(), b"")

    def test_created_transaction_temporary_same_inode_tamper_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_transaction_temporary_created_anchor_boundary",
                exit_code=97,
                seam_body="    os._exit(97)",
            )
            temporary = next(prepared.glob("..shard-build-transaction.json.*.partial"))
            temporary.write_bytes(b"foreign")
            with self.assertRaisesRegex(RuntimeError, "temporary identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(temporary.read_bytes(), b"foreign")

    def test_owner_marker_initialization_boundaries_are_recoverable(self) -> None:
        rows = _rows()
        seams = (
            ("_after_owner_marker_intent_boundary", 99),
            ("_after_owner_marker_created_anchor_boundary", 100),
            ("_after_owner_marker_payload_boundary", 101),
        )
        for seam, exit_code in seams:
            with self.subTest(seam=seam), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam=seam,
                    exit_code=exit_code,
                    seam_body=f"    os._exit({exit_code})",
                )
                recovered = self._retry_crashed_build(rows, split, prepared)
                self.assertTrue(recovered.manifest_path.is_file())

    def test_created_owner_marker_replacement_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_created_anchor_boundary",
                exit_code=100,
                seam_body="    os._exit(100)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            initializing = work / ".bitguard-shard-owner.initializing"
            displaced = root / "owned-empty-owner"
            replacement = root / "foreign-empty-owner"
            initializing.replace(displaced)
            replacement.write_bytes(b"")
            replacement.replace(initializing)
            with self.assertRaisesRegex(RuntimeError, "owner marker identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(initializing.read_bytes(), b"")

    def test_created_owner_marker_same_inode_tamper_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_created_anchor_boundary",
                exit_code=100,
                seam_body="    os._exit(100)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            initializing = work / ".bitguard-shard-owner.initializing"
            initializing.write_bytes(b"foreign")
            with self.assertRaisesRegex(RuntimeError, "owner marker identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(initializing.read_bytes(), b"foreign")

    def test_lock_initialization_boundaries_are_recoverable_and_bounded(self) -> None:
        rows = _rows()
        seams = (
            ("_after_lock_initializing_boundary", 102),
            ("_after_lock_publish_link_boundary", 103),
        )
        for seam, exit_code in seams:
            with self.subTest(seam=seam), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                repetitions = 1 if seam == "_after_lock_publish_link_boundary" else 2
                for _ in range(repetitions):
                    self._run_crashing_builder(
                        split=split,
                        prepared=prepared,
                        seam=seam,
                        exit_code=exit_code,
                        seam_body=f"    os._exit({exit_code})",
                    )
                    self.assertLessEqual(
                        len(
                            list(
                                prepared.glob(
                                    "..shard-build-transaction.lock.*.initializing"
                                )
                            )
                        ),
                        1,
                    )
                recovered = self._retry_crashed_build(rows, split, prepared)
                self.assertTrue(recovered.manifest_path.is_file())
                self.assertEqual(
                    list(
                        prepared.glob("..shard-build-transaction.lock.*.initializing")
                    ),
                    [],
                )

    def test_simultaneous_first_lock_initializers_retire_the_loser_crash_safely(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            barrier = root / "barrier"
            prepared.mkdir()
            barrier.mkdir()
            child = """
import os
import time
from pathlib import Path
from bitguard_bnn.out_of_core import shard as shard_module

prepared = Path({prepared!r})
barrier = Path({barrier!r})
crash_retirement = {crash_retirement!r}

def synchronize_creation(lock):
    (barrier / f"ready-{{os.getpid()}}").write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while len(list(barrier.glob("ready-*"))) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("lock initializer barrier timed out")
        time.sleep(0.01)

def order_publication(initializing):
    (barrier / f"initialized-{{os.getpid()}}").write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while len(list(barrier.glob("initialized-*"))) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("lock initialization barrier timed out")
        time.sleep(0.01)
    if not crash_retirement:
        time.sleep(0.10)

def crash_after_retiring(initializing, retiring):
    if crash_retirement:
        os._exit(123)

shard_module._before_lock_initializer_creation = synchronize_creation
shard_module._after_lock_initializing_boundary = order_publication
shard_module._after_lock_loser_retirement_boundary = crash_after_retiring
handle = shard_module._acquire_shard_transaction_lock(
    prepared / ".shard-build-transaction.lock"
)
shard_module._release_shard_transaction_lock(handle)
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(repository / "src"), str(repository))
            )
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        child.format(
                            prepared=str(prepared),
                            barrier=str(barrier),
                            crash_retirement=crash_retirement,
                        ),
                    ],
                    cwd=repository,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for crash_retirement in (True, False)
            ]
            results = [process.communicate(timeout=30) for process in processes]
            return_codes = [process.returncode for process in processes]
            self.assertEqual(
                sorted(return_codes),
                [0, 123],
                msg="\n".join(
                    f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
                    for process, (stdout, stderr) in zip(
                        processes, results, strict=True
                    )
                ),
            )
            self.assertEqual(
                len(list(prepared.glob("..shard-build-transaction.lock.*.retiring"))),
                0,
            )

            from bitguard_bnn.out_of_core import shard as shard_module

            handle = shard_module._acquire_shard_transaction_lock(
                prepared / ".shard-build-transaction.lock"
            )
            shard_module._release_shard_transaction_lock(handle)
            self.assertEqual(
                list(prepared.glob("..shard-build-transaction.lock.*.initializing")),
                [],
            )
            self.assertEqual(
                list(prepared.glob("..shard-build-transaction.lock.*.retiring")), []
            )

    def test_foreign_precreated_lock_retirement_hardlink_is_preserved(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "prepared"
            prepared.mkdir()
            lock = prepared / ".shard-build-transaction.lock"
            handle = shard_module._acquire_shard_transaction_lock(lock)
            shard_module._release_shard_transaction_lock(handle)
            initializing = prepared / (f".{lock.name}.{'a' * 32}.initializing")
            with initializing.open("xb") as candidate:
                status = os.fstat(candidate.fileno())
                payload = {
                    "schema_version": shard_module._SHARD_LOCK_SCHEMA,
                    "lock_instance": shard_module._status_instance(status),
                    "lock_path": str(lock),
                    "owner_token": "b" * 32,
                    "journal": None,
                    "owner": None,
                    "manifest": None,
                    "temporaries": [],
                    "retiring_initializers": [],
                }
                candidate.write(
                    b"\0" + shard_module.canonical_json_bytes(payload) + b"\n"
                )
                candidate.flush()
                os.fsync(candidate.fileno())
            retiring = shard_module._lock_retiring_path(initializing)
            try:
                os.link(initializing, retiring)
            except OSError as exc:
                raise unittest.SkipTest(f"hard links unavailable: {exc}") from exc
            expected = initializing.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "retirement"):
                shard_module._acquire_shard_transaction_lock(lock)
            self.assertEqual(initializing.read_bytes(), expected)
            self.assertEqual(retiring.read_bytes(), expected)

    def test_lock_initialization_replacement_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_lock_initializing_boundary",
                exit_code=102,
                seam_body="    os._exit(102)",
            )
            initializing = next(
                prepared.glob("..shard-build-transaction.lock.*.initializing")
            )
            displaced = root / "owned-lock-initializing"
            replacement = root / "foreign-lock-initializing"
            contents = initializing.read_bytes()
            initializing.replace(displaced)
            replacement.write_bytes(contents)
            replacement.replace(initializing)
            with self.assertRaisesRegex(RuntimeError, "lock initialization identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(initializing.read_bytes(), contents)

    def test_relative_then_absolute_output_recovers_same_transaction(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            self._run_crashing_builder(
                split=split,
                prepared=Path("prepared"),
                seam="_after_staged_write_boundary",
                exit_code=87,
                seam_body="    os._exit(87)",
                cwd=root,
            )
            recovered = self._retry_crashed_build(rows, split, root / "prepared")
            self.assertTrue(recovered.manifest_path.is_file())

    def test_owner_publication_anchor_rejects_same_byte_replacement(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_replace_boundary",
                exit_code=95,
                seam_body="    os._exit(95)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            owner = work / ".bitguard-shard-owner.json"
            displaced = root / "owned-owner.json"
            replacement = root / "replacement-owner.json"
            owner_bytes = owner.read_bytes()
            owner.replace(displaced)
            replacement.write_bytes(owner_bytes)
            replacement.replace(owner)
            with self.assertRaisesRegex(RuntimeError, "owner marker identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(owner.read_bytes(), owner_bytes)
            owner.unlink()
            displaced.replace(owner)
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())

    def test_owner_initializing_marker_retirement_survives_repeated_hard_exits(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_replace_boundary",
                exit_code=95,
                seam_body="    os._exit(95)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            owner = work / ".bitguard-shard-owner.json"
            lock = prepared / ".shard-build-transaction.lock"

            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_retirement_intent_boundary",
                exit_code=117,
                seam_body="    os._exit(117)",
            )
            with lock.open("r+b", buffering=0) as handle:
                lock_anchor, _, _ = shard_module._read_lock_storage(handle)
            self.assertEqual(lock_anchor["owner"]["phase"], "retiring")
            self.assertEqual(
                os.path.normcase(lock_anchor["owner"]["record"]["retiring_path"]),
                os.path.normcase(str(owner)),
            )
            self.assertTrue(owner.is_file())

            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_retirement_removal_boundary",
                exit_code=118,
                seam_body="    os._exit(118)",
            )
            with lock.open("r+b", buffering=0) as handle:
                lock_anchor, _, _ = shard_module._read_lock_storage(handle)
            self.assertEqual(lock_anchor["owner"]["phase"], "retiring")
            self.assertFalse(owner.exists())

            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_retirement_progress_boundary",
                exit_code=119,
                seam_body="    os._exit(119)",
            )
            with lock.open("r+b", buffering=0) as handle:
                lock_anchor, _, _ = shard_module._read_lock_storage(handle)
            self.assertIsNone(lock_anchor["owner"])
            self.assertTrue(work.is_dir())

            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())
            self.assertEqual(list(prepared.glob(".shards-*.partial")), [])

    def test_retiring_owner_marker_hardlink_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_replace_boundary",
                exit_code=95,
                seam_body="    os._exit(95)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            owner = work / ".bitguard-shard-owner.json"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_retirement_intent_boundary",
                exit_code=117,
                seam_body="    os._exit(117)",
            )
            external = root / "owner-hardlink.json"
            try:
                os.link(owner, external)
            except OSError as exc:
                raise unittest.SkipTest(f"hard links unavailable: {exc}") from exc
            expected = owner.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "owner marker identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(owner.read_bytes(), expected)
            self.assertEqual(external.read_bytes(), expected)

    def test_created_owner_marker_absence_without_retirement_progress_is_rejected(
        self,
    ) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_created_anchor_boundary",
                exit_code=100,
                seam_body="    os._exit(100)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            initializing = work / ".bitguard-shard-owner.initializing"
            initializing.unlink()
            with self.assertRaisesRegex(RuntimeError, "disappeared.*progress"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertFalse(initializing.exists())

    def test_owner_retirement_quarantine_preserves_a_last_moment_replacement(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_replace_boundary",
                exit_code=95,
                seam_body="    os._exit(95)",
            )
            work = next(prepared.glob(".shards-*.partial"))
            owner = work / ".bitguard-shard-owner.json"
            displaced = root / "owned-owner.json"
            replacement = root / "replacement-owner.json"
            replacement.write_bytes(owner.read_bytes())
            replacement_inode = replacement.stat().st_ino

            def replace_before_quarantine(path: Path) -> None:
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_owner_marker_retirement_quarantine_boundary",
                    side_effect=replace_before_quarantine,
                ),
                self.assertRaisesRegex(RuntimeError, "owner marker identity"),
            ):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(owner.is_file())
            self.assertEqual(owner.stat().st_ino, replacement_inode)
            self.assertTrue(displaced.is_file())

    def test_owner_retirement_final_removal_preserves_replacement(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_owner_marker_replace_boundary",
                exit_code=95,
                seam_body="    os._exit(95)",
            )
            replacement = root / "replacement-owner.json"
            replacement.write_bytes(b"foreign-owner")
            displaced = root / "displaced-owner.json"
            invoked = False

            def replace_at_final_removal(path: Path, kind: str) -> None:
                nonlocal invoked
                if invoked or kind != "file" or path.parent != prepared:
                    return
                invoked = True
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_identity_bound_entry_removal",
                    side_effect=replace_at_final_removal,
                ),
                self.assertRaisesRegex(RuntimeError, "foreign file replaced"),
            ):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(invoked)
            retained = list(prepared.glob(".bitguard-retired-*.file"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"foreign-owner")

    def test_anchored_journal_temporary_is_bounded_and_replacement_is_preserved(
        self,
    ) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_before_shard_transaction_replace",
                exit_code=94,
                seam_body="    os._exit(94)",
            )
            debris = list(prepared.glob("..shard-build-transaction.json.*.partial"))
            self.assertEqual(len(debris), 1)
            anchored = debris[0]
            displaced = root / "owned-transaction.partial"
            replacement = root / "replacement-transaction.partial"
            contents = anchored.read_bytes()
            anchored.replace(displaced)
            replacement.write_bytes(contents)
            replacement.replace(anchored)
            with self.assertRaisesRegex(RuntimeError, "transaction temporary identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(anchored.read_bytes(), contents)
            anchored.unlink()
            displaced.replace(anchored)
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())
            self.assertEqual(
                list(prepared.glob("..shard-build-transaction.json.*.partial")), []
            )

    def test_foreign_lock_entries_are_preserved_without_writes(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            prepared.mkdir()
            external = root / "external-empty.bin"
            external.write_bytes(b"")
            lock = prepared / ".shard-build-transaction.lock"
            try:
                os.link(external, lock)
            except OSError as exc:
                raise unittest.SkipTest(f"hard links unavailable: {exc}") from exc
            with self.assertRaisesRegex(RuntimeError, "lock"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(external.read_bytes(), b"")
            self.assertEqual(lock.read_bytes(), b"")

    def test_retired_entries_are_recorded_and_replacements_fail_closed(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "prepared"
            prepared.mkdir()
            lock_path = prepared / ".shard-build-transaction.lock"
            transaction_lock = shard_module._acquire_shard_transaction_lock(lock_path)
            try:
                retired_file = shard_module._retired_entry_path(prepared, "file")
                anchor = shard_module._read_lock_anchor(transaction_lock)
                anchor["temporaries"] = [
                    {
                        "phase": "retiring",
                        "quarantine_path": str(retired_file),
                    }
                ]
                shard_module._write_lock_anchor(transaction_lock, anchor)
                retired_file.write_bytes(b"owned payload")
                file_identity = shard_module.file_identity(retired_file)
                shard_module._retire_regular_file_without_pathname_delete(
                    retired_file, file_identity, transaction_lock
                )
                shard_module._retire_regular_file_without_pathname_delete(
                    retired_file, file_identity, transaction_lock
                )
                anchor = shard_module._read_lock_anchor(transaction_lock)
                anchor["temporaries"] = []
                shard_module._write_lock_anchor(transaction_lock, anchor)
                self.assertEqual(len(anchor["retired_entries"]), 1)
                self.assertEqual(retired_file.read_bytes(), b"")
            finally:
                shard_module._release_shard_transaction_lock(transaction_lock)

            displaced = prepared / "owned-retired.file"
            retired_file.replace(displaced)
            retired_file.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "retired shard file identity"):
                shard_module._acquire_shard_transaction_lock(lock_path)
            self.assertEqual(retired_file.read_bytes(), b"")
            self.assertTrue(displaced.is_file())

    def test_unrecorded_retired_entry_is_rejected(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "prepared"
            prepared.mkdir()
            lock_path = prepared / ".shard-build-transaction.lock"
            transaction_lock = shard_module._acquire_shard_transaction_lock(lock_path)
            shard_module._release_shard_transaction_lock(transaction_lock)
            foreign = prepared / (".bitguard-retired-" + "a" * 32 + ".file")
            foreign.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "unrecorded retired shard"):
                shard_module._acquire_shard_transaction_lock(lock_path)
            self.assertEqual(foreign.read_bytes(), b"")

    def test_same_byte_lock_replacement_is_rejected_and_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            completed = self._retry_crashed_build(rows, split, prepared)
            lock = prepared / ".shard-build-transaction.lock"
            displaced = root / "owned-lock"
            replacement = root / "replacement-lock"
            lock_bytes = lock.read_bytes()
            lock.replace(displaced)
            replacement.write_bytes(lock_bytes)
            replacement.replace(lock)
            with self.assertRaisesRegex(RuntimeError, "lock identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(lock.read_bytes(), lock_bytes)
            self.assertTrue(completed.manifest_path.is_file())

    def test_symlink_lock_is_rejected_without_touching_target(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            prepared.mkdir()
            external = root / "external-lock-target.bin"
            external.write_bytes(b"external")
            lock = prepared / ".shard-build-transaction.lock"
            try:
                lock.symlink_to(external)
            except OSError as exc:
                raise unittest.SkipTest(f"file symlinks unavailable: {exc}") from exc
            with self.assertRaisesRegex(RuntimeError, "lock"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(external.read_bytes(), b"external")
            self.assertTrue(lock.is_symlink())

    def test_foreign_stale_shard_work_is_preserved_and_rejected(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            foreign = prepared / ".shards-foreign.partial"
            foreign.mkdir(parents=True)
            sentinel = foreign / "foreign.bin"
            sentinel.write_bytes(b"do-not-delete")
            with self.assertRaisesRegex(RuntimeError, "unsafe stale shard work"):
                shard_module.write_parquet_shards(
                    _chunks(rows),
                    split,
                    ("f1", "f2"),
                    prepared,
                    dataset_name="fixture",
                    preprocessing_fingerprint="preprocess-v1",
                    shard_target_rows=3,
                    max_rows_per_run=4,
                    merge_fan_in=2,
                )
            self.assertEqual(sentinel.read_bytes(), b"do-not-delete")

    def test_owned_cleanup_entry_removal_boundaries_resume_with_parity(self) -> None:
        rows = _rows(6)
        seams = (
            ("_before_owned_inventory_entry_removal", 104),
            ("_after_owned_inventory_entry_removal", 105),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            clean = self._retry_crashed_build(rows, split, root / "clean")
            clean_bytes = clean.manifest_path.read_bytes()
            inventory_count: int | None = None
            for seam, exit_code in seams:
                boundary = 0
                while inventory_count is None or boundary < inventory_count:
                    with self.subTest(seam=seam, boundary=boundary):
                        prepared = root / f"prepared-{seam}-{boundary}"
                        seam_body = (
                            "    current = getattr(crash_seam, 'calls', 0)\n"
                            "    crash_seam.calls = current + 1\n"
                            f"    if current == {boundary}:\n"
                            f"        os._exit({exit_code})"
                        )
                        self._run_crashing_builder(
                            split=split,
                            prepared=prepared,
                            seam=seam,
                            exit_code=exit_code,
                            seam_body=seam_body,
                            row_count=len(rows),
                        )
                        journal = json.loads(
                            (prepared / ".shard-build-transaction.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if inventory_count is None:
                            inventory_count = len(journal["cleanup_inventory"])
                        self.assertIsNotNone(journal.get("cleanup_removal_intent"))
                        recovered = self._retry_crashed_build(rows, split, prepared)
                        self.assertEqual(
                            recovered.manifest_path.read_bytes(), clean_bytes
                        )
                        self.assertEqual(list(prepared.glob(".shards-*.partial")), [])
                        shutil.rmtree(prepared)
                    boundary += 1

    def test_public_directory_creation_boundaries_are_recoverable(self) -> None:
        rows = _rows()
        seams = (
            ("_after_public_directory_intent_boundary", 106),
            ("_after_public_directory_created_anchor_boundary", 107),
            ("_after_public_directory_publish_boundary", 116),
        )
        for seam, exit_code in seams:
            for boundary in range(3):
                with (
                    self.subTest(seam=seam, boundary=boundary),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    root = Path(temp)
                    split = _split(rows, root)
                    prepared = root / "prepared"
                    seam_body = (
                        "    current = getattr(crash_seam, 'calls', 0)\n"
                        "    crash_seam.calls = current + 1\n"
                        f"    if current == {boundary}:\n"
                        f"        os._exit({exit_code})"
                    )
                    self._run_crashing_builder(
                        split=split,
                        prepared=prepared,
                        seam=seam,
                        exit_code=exit_code,
                        seam_body=seam_body,
                    )
                    recovered = self._retry_crashed_build(rows, split, prepared)
                    self.assertTrue(recovered.manifest_path.is_file())
                    self.assertFalse(
                        (prepared / ".shard-build-transaction.json").exists()
                    )

    def test_public_directory_rollback_progress_survives_each_hard_exit(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            clean = self._retry_crashed_build(rows, split, root / "clean")
            clean_bytes = clean.manifest_path.read_bytes()
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_publish_boundary",
                exit_code=116,
                seam_body=(
                    "    current = getattr(crash_seam, 'calls', 0)\n"
                    "    crash_seam.calls = current + 1\n"
                    "    if current == 2:\n"
                    "        os._exit(116)"
                ),
            )
            journal = prepared / ".shard-build-transaction.json"
            initial = json.loads(journal.read_text(encoding="utf-8"))
            expected_paths = [
                initial["directory_intent"]["path"],
                *reversed(
                    [record["path"] for record in initial["published_directories"]]
                ),
            ]
            self.assertGreaterEqual(len(expected_paths), 2)

            for removed_count, expected_path in enumerate(expected_paths):
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam="_after_public_directory_rollback_removal_boundary",
                    exit_code=120,
                    seam_body="    os._exit(120)",
                )
                transaction = json.loads(journal.read_text(encoding="utf-8"))
                intent = transaction["directory_rollback_intent"]
                self.assertEqual(intent["path"], expected_path)
                self.assertEqual(len(intent["identity"]), 3)
                self.assertEqual(
                    transaction["directory_rollback_removed"],
                    expected_paths[:removed_count],
                )
                self.assertFalse(Path(expected_path).exists())
                absent = [path for path in expected_paths if not Path(path).exists()]
                self.assertEqual(absent, expected_paths[: removed_count + 1])

            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(recovered.manifest_path.read_bytes(), clean_bytes)
            self.assertFalse(journal.exists())

    def test_public_directory_absence_without_matching_progress_is_rejected(
        self,
    ) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_publish_boundary",
                exit_code=116,
                seam_body=(
                    "    current = getattr(crash_seam, 'calls', 0)\n"
                    "    crash_seam.calls = current + 1\n"
                    "    if current == 2:\n"
                    "        os._exit(116)"
                ),
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            untracked = Path(transaction["directory_intent"]["path"])
            untracked.rmdir()
            with self.assertRaisesRegex(RuntimeError, "directory.*missing.*progress"):
                self._retry_crashed_build(rows, split, prepared)

    def test_public_directory_quarantine_preserves_last_moment_replacement(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_publish_boundary",
                exit_code=116,
                seam_body=(
                    "    current = getattr(crash_seam, 'calls', 0)\n"
                    "    crash_seam.calls = current + 1\n"
                    "    if current == 2:\n"
                    "        os._exit(116)"
                ),
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            directory = Path(transaction["directory_intent"]["path"])
            displaced = root / "owned-directory"
            replacement = root / "replacement-directory"
            replacement.mkdir()
            replacement_inode = replacement.stat().st_ino

            def replace_before_quarantine(path: Path) -> None:
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_public_directory_rollback_quarantine_boundary",
                    side_effect=replace_before_quarantine,
                ),
                self.assertRaisesRegex(RuntimeError, "directory identity"),
            ):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(directory.is_dir())
            self.assertEqual(directory.stat().st_ino, replacement_inode)
            self.assertTrue(displaced.is_dir())

    def test_public_directory_final_removal_preserves_replacement(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_publish_boundary",
                exit_code=116,
                seam_body=(
                    "    current = getattr(crash_seam, 'calls', 0)\n"
                    "    crash_seam.calls = current + 1\n"
                    "    if current == 2:\n"
                    "        os._exit(116)"
                ),
            )
            replacement = root / "replacement-directory"
            replacement.mkdir()
            replacement_inode = replacement.stat().st_ino
            displaced = root / "displaced-directory"
            invoked = False
            replaced_path: Path | None = None

            def replace_at_final_removal(path: Path, kind: str) -> None:
                nonlocal invoked, replaced_path
                if invoked or kind != "directory":
                    return
                invoked = True
                replaced_path = path
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_identity_bound_entry_removal",
                    side_effect=replace_at_final_removal,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "foreign directory replaced|published directory identity changed",
                ),
            ):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(invoked)
            self.assertIsNotNone(replaced_path)
            assert replaced_path is not None
            self.assertTrue(replaced_path.is_dir())
            self.assertEqual(replaced_path.stat().st_ino, replacement_inode)
            self.assertTrue(displaced.is_dir())

    def test_public_artifact_final_removal_preserves_replacement(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows(6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_shard_source_removal",
                exit_code=114,
                seam_body="    os._exit(114)",
                row_count=len(rows),
            )
            published = next((prepared / "dataset=fixture").rglob("*.parquet"))
            replacement = root / "replacement-public.parquet"
            replacement.write_bytes(b"foreign-public")
            displaced = root / "displaced-public.parquet"
            invoked = False

            def replace_at_final_removal(path: Path, kind: str) -> None:
                nonlocal invoked
                if invoked or kind != "file" or path.suffix != ".parquet":
                    return
                invoked = True
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_identity_bound_entry_removal",
                    side_effect=replace_at_final_removal,
                ),
                self.assertRaisesRegex(RuntimeError, "foreign file replaced"),
            ):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(invoked)
            self.assertEqual(published.read_bytes(), b"foreign-public")

    def test_legacy_v1_publishing_journal_migrates_directory_progress(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_publish_boundary",
                exit_code=116,
                seam_body="    os._exit(116)",
            )
            journal = prepared / ".shard-build-transaction.json"
            transaction = json.loads(journal.read_text(encoding="utf-8"))
            transaction.pop("directory_rollback_removed")
            transaction.pop("directory_rollback_intent")
            journal.write_bytes(shard_module.canonical_json_bytes(transaction) + b"\n")
            record = shard_module._anchored_file_record(journal, transaction)
            lock = prepared / ".shard-build-transaction.lock"
            with lock.open("r+b", buffering=0) as handle:
                anchor = shard_module._read_lock_anchor(handle)
                anchor["journal"] = {"phase": "active", "accepted": [record]}
                shard_module._write_lock_anchor(handle, anchor)
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())

    def test_anchored_public_directory_replacement_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_created_anchor_boundary",
                exit_code=107,
                seam_body="    os._exit(107)",
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            intent = transaction["directory_intent"]
            directory = Path(intent["staging_path"])
            displaced = root / "owned-public-directory"
            directory.replace(displaced)
            directory.mkdir()
            with self.assertRaisesRegex(RuntimeError, "directory identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(directory.is_dir())

    def test_public_directory_intent_does_not_claim_a_foreign_directory(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_intent_boundary",
                exit_code=106,
                seam_body="    os._exit(106)",
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            directory = Path(transaction["directory_intent"]["path"])
            directory.mkdir()
            foreign_identity = directory.stat().st_ino
            with self.assertRaisesRegex(RuntimeError, "directory identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(directory.is_dir())
            self.assertEqual(directory.stat().st_ino, foreign_identity)

    def test_anchored_public_directory_nonempty_entry_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_directory_created_anchor_boundary",
                exit_code=107,
                seam_body="    os._exit(107)",
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            directory = Path(transaction["directory_intent"]["staging_path"])
            sentinel = directory / "foreign.bin"
            sentinel.write_bytes(b"preserve")
            with self.assertRaisesRegex(RuntimeError, "foreign published entry"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_manifest_temporary_boundaries_are_recoverable_and_bounded(self) -> None:
        rows = _rows()
        seams = (
            ("_after_manifest_temporary_intent_boundary", 108),
            ("_after_manifest_temporary_created_anchor_boundary", 109),
            ("_after_manifest_temporary_payload_boundary", 110),
            ("_after_manifest_publish_link_boundary", 111),
        )
        for seam, exit_code in seams:
            with self.subTest(seam=seam), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                repetitions = (
                    1 if seam == "_after_manifest_publish_link_boundary" else 2
                )
                for _ in range(repetitions):
                    self._run_crashing_builder(
                        split=split,
                        prepared=prepared,
                        seam=seam,
                        exit_code=exit_code,
                        seam_body=f"    os._exit({exit_code})",
                    )
                    self.assertLessEqual(
                        len(list(prepared.glob(".shard_manifest.json.*.partial"))),
                        1,
                    )
                recovered = self._retry_crashed_build(rows, split, prepared)
                self.assertTrue(recovered.manifest_path.is_file())
                self.assertEqual(
                    list(prepared.glob(".shard_manifest.json.*.partial")), []
                )

    def test_manifest_commit_boundaries_preserve_exact_publication(self) -> None:
        rows = _rows()
        seams = (
            ("_after_manifest_publish_link_boundary", 124),
            ("_before_journal_retirement_quarantine_boundary", 125),
        )
        for seam, exit_code in seams:
            with self.subTest(seam=seam), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam=seam,
                    exit_code=exit_code,
                    seam_body=f"    os._exit({exit_code})",
                )
                manifest = prepared / "shard_manifest.json"
                self.assertTrue(manifest.is_file())
                expected = {
                    path.relative_to(prepared).as_posix(): path.read_bytes()
                    for path in sorted(
                        (prepared / "dataset=fixture").rglob("*.parquet")
                    )
                }
                expected["shard_manifest.json"] = manifest.read_bytes()

                recovered = self._retry_crashed_build(rows, split, prepared)

                self.assertTrue(recovered.manifest_path.is_file())
                observed = {
                    path.relative_to(prepared).as_posix(): path.read_bytes()
                    for path in sorted(
                        (prepared / "dataset=fixture").rglob("*.parquet")
                    )
                }
                observed["shard_manifest.json"] = manifest.read_bytes()
                self.assertEqual(observed, expected)
                self.assertFalse((prepared / ".shard-build-transaction.json").exists())

    def test_committed_retirement_revalidates_artifact_bytes_before_clear(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_before_journal_retirement_quarantine_boundary",
                exit_code=125,
                seam_body="    os._exit(125)",
            )
            journal = prepared / ".shard-build-transaction.json"
            transaction = json.loads(journal.read_text(encoding="utf-8"))
            artifact = transaction["artifacts"][0]
            final = prepared.joinpath(*str(artifact["relative"]).split("/"))
            expected_identity = shard_module.file_identity(final)
            status = final.stat()
            with final.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 0xFF]))
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                final,
                ns=(int(status.st_atime_ns), int(status.st_mtime_ns)),
            )
            self.assertEqual(shard_module.file_identity(final), expected_identity)

            with self.assertRaisesRegex(
                RuntimeError, "manifest commit artifact changed"
            ):
                self._retry_crashed_build(rows, split, prepared)

            self.assertTrue(journal.is_file())
            self.assertTrue((prepared / "shard_manifest.json").is_file())
            lock_path = prepared / ".shard-build-transaction.lock"
            with lock_path.open("r+b", buffering=0) as lock:
                anchor = shard_module._read_lock_anchor(lock)
            self.assertEqual(anchor["manifest"]["phase"], "committed")
            self.assertEqual(anchor["journal"]["phase"], "retiring")

    def test_anchorless_publishing_retirement_migrates_evidence_before_clear(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_before_journal_retirement_quarantine_boundary",
                exit_code=125,
                seam_body="    os._exit(125)",
            )
            journal = prepared / ".shard-build-transaction.json"
            transaction = json.loads(journal.read_text(encoding="utf-8"))
            lock_path = prepared / ".shard-build-transaction.lock"
            with lock_path.open("r+b", buffering=0) as lock:
                anchor = shard_module._read_lock_anchor(lock)
                anchor["manifest"] = None
                shard_module._write_lock_anchor(lock, anchor)

            artifact = transaction["artifacts"][0]
            final = prepared.joinpath(*str(artifact["relative"]).split("/"))
            expected_identity = shard_module.file_identity(final)
            status = final.stat()
            with final.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 0xFF]))
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                final,
                ns=(int(status.st_atime_ns), int(status.st_mtime_ns)),
            )
            self.assertEqual(shard_module.file_identity(final), expected_identity)

            with self.assertRaisesRegex(
                RuntimeError, "manifest commit artifact changed"
            ):
                self._retry_crashed_build(rows, split, prepared)

            self.assertTrue(journal.is_file())
            self.assertTrue((prepared / "shard_manifest.json").is_file())
            with lock_path.open("r+b", buffering=0) as lock:
                anchor = shard_module._read_lock_anchor(lock)
            self.assertEqual(anchor["manifest"]["phase"], "published")
            self.assertEqual(anchor["journal"]["phase"], "retiring")

    def test_created_manifest_temporary_replacement_is_preserved(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_manifest_temporary_created_anchor_boundary",
                exit_code=109,
                seam_body="    os._exit(109)",
            )
            partial = next(prepared.glob(".shard_manifest.json.*.partial"))
            displaced = root / "owned-manifest.partial"
            replacement = root / "foreign-manifest.partial"
            partial.replace(displaced)
            replacement.write_bytes(b"")
            replacement.replace(partial)
            with self.assertRaisesRegex(RuntimeError, "manifest temporary identity"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertEqual(partial.read_bytes(), b"")

    def test_active_manifest_temporary_replacement_and_tamper_are_preserved(
        self,
    ) -> None:
        rows = _rows()
        for mutation in ("replacement", "tamper"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam="_after_manifest_temporary_payload_boundary",
                    exit_code=110,
                    seam_body="    os._exit(110)",
                )
                partial = next(prepared.glob(".shard_manifest.json.*.partial"))
                original = partial.read_bytes()
                if mutation == "replacement":
                    displaced = root / "owned-manifest.partial"
                    replacement = root / "foreign-manifest.partial"
                    partial.replace(displaced)
                    replacement.write_bytes(original)
                    replacement.replace(partial)
                else:
                    partial.write_bytes(original + b" ")
                with self.assertRaisesRegex(
                    RuntimeError, "manifest temporary identity"
                ):
                    self._retry_crashed_build(rows, split, prepared)
                self.assertTrue(partial.is_file())

    def test_cleanup_progress_rejects_unjournaled_missing_and_replacement(self) -> None:
        rows = _rows()
        for mutation in ("missing", "replacement"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = _split(rows, root)
                prepared = root / "prepared"
                self._run_crashing_builder(
                    split=split,
                    prepared=prepared,
                    seam="_before_owned_inventory_entry_removal",
                    exit_code=104,
                    seam_body="    os._exit(104)",
                )
                transaction = json.loads(
                    (prepared / ".shard-build-transaction.json").read_text(
                        encoding="utf-8"
                    )
                )
                work = Path(transaction["work"])
                intent = transaction["cleanup_removal_intent"]
                candidate_entry = next(
                    entry
                    for entry in transaction["cleanup_inventory"]
                    if entry["kind"] == "file" and entry["path"] != intent
                )
                candidate = work.joinpath(*Path(candidate_entry["path"]).parts)
                original = candidate.read_bytes()
                displaced = root / "displaced-cleanup-entry"
                candidate.replace(displaced)
                if mutation == "replacement":
                    replacement = root / "foreign-cleanup-entry"
                    replacement.write_bytes(original)
                    replacement.replace(candidate)
                with self.assertRaisesRegex(RuntimeError, "cleanup inventory changed"):
                    self._retry_crashed_build(rows, split, prepared)
                if mutation == "replacement":
                    self.assertEqual(candidate.read_bytes(), original)
                self.assertTrue(work.is_dir())

    def test_intent_only_work_root_never_claims_a_foreign_empty_directory(
        self,
    ) -> None:
        rows = _rows(6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_work_directory_creation_intent_boundary",
                exit_code=112,
                seam_body="    os._exit(112)",
                row_count=len(rows),
            )
            transaction = json.loads(
                (prepared / ".shard-build-transaction.json").read_text(encoding="utf-8")
            )
            work = Path(transaction["work"])
            work.mkdir()
            identity = work.stat().st_ino
            with self.assertRaisesRegex(RuntimeError, "ownership initialization"):
                self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(work.is_dir())
            self.assertEqual(work.stat().st_ino, identity)

    def test_manifest_recovery_entry_removal_is_itself_crash_resumable(self) -> None:
        rows = _rows(6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_manifest_temporary_payload_boundary",
                exit_code=110,
                seam_body="    os._exit(110)",
                row_count=len(rows),
            )
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_manifest_recovery_entry_removal",
                exit_code=113,
                seam_body="    os._exit(113)",
                row_count=len(rows),
            )
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())
            self.assertEqual(list(prepared.glob(".shard_manifest.json.*.partial")), [])

    def test_public_artifact_rollback_is_itself_crash_resumable(self) -> None:
        rows = _rows(6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_shard_source_removal",
                exit_code=114,
                seam_body="    os._exit(114)",
                row_count=len(rows),
            )
            self._run_crashing_builder(
                split=split,
                prepared=prepared,
                seam="_after_public_artifact_rollback_removal",
                exit_code=115,
                seam_body="    os._exit(115)",
                row_count=len(rows),
            )
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())
            self.assertFalse((prepared / ".shard-build-transaction.json").exists())

    def test_directory_anchor_exception_cleans_latest_owned_state(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        rows = _rows(6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = _split(rows, root)
            prepared = root / "prepared"
            with patch.object(
                shard_module,
                "_after_public_directory_created_anchor_boundary",
                side_effect=RuntimeError("directory-anchor-failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "directory-anchor-failure"):
                    self._retry_crashed_build(rows, split, prepared)
            self.assertFalse((prepared / ".shard-build-transaction.json").exists())
            self.assertEqual(list(prepared.glob(".shards-*.partial")), [])
            self.assertEqual(list(prepared.glob(".*.directory.partial")), [])
            recovered = self._retry_crashed_build(rows, split, prepared)
            self.assertTrue(recovered.manifest_path.is_file())

    def test_identity_bound_quarantine_restores_a_final_syscall_replacement(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "owned.bin"
            quarantine = root / "quarantine.bin"
            displaced = root / "displaced-owned.bin"
            replacement = root / "foreign.bin"
            source.write_bytes(b"owned")
            replacement.write_bytes(b"foreign")
            expected = shard_module.file_identity(source)
            invoked = False

            def replace_at_move(path: Path, kind: str) -> None:
                nonlocal invoked
                if invoked or path != source or kind != "file":
                    return
                invoked = True
                path.replace(displaced)
                replacement.replace(path)

            with (
                patch.object(
                    shard_module,
                    "_before_identity_bound_entry_removal",
                    side_effect=replace_at_move,
                ),
                self.assertRaisesRegex(RuntimeError, "identity-bound file changed"),
            ):
                shard_module._move_entry_to_quarantine_exact(
                    source,
                    quarantine,
                    expected,
                    kind="file",
                )
            self.assertTrue(invoked)
            self.assertEqual(source.read_bytes(), b"foreign")
            self.assertEqual(displaced.read_bytes(), b"owned")
            self.assertFalse(quarantine.exists())

    def test_owner_marker_publication_is_no_clobber(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                work = output / ".shards-owned.partial"
                work.mkdir()
                owner = work / shard_module._SHARD_WORK_OWNER_FILE
                initializing = work / shard_module._SHARD_WORK_OWNER_INITIALIZING_FILE
                payload = {
                    "schema_version": 1,
                    "transaction_id": "a" * 32,
                    "request_fingerprint": "b" * 64,
                }

                def install_foreign(_initializing: Path) -> None:
                    owner.write_bytes(b"foreign-owner")

                with (
                    patch.object(
                        shard_module,
                        "_after_owner_marker_payload_boundary",
                        side_effect=install_foreign,
                    ),
                    self.assertRaisesRegex(RuntimeError, "already exists"),
                ):
                    shard_module._write_shard_owner_marker(
                        owner, initializing, payload, lock
                    )
                self.assertEqual(owner.read_bytes(), b"foreign-owner")
                self.assertTrue(initializing.is_file())
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_lock_publication_is_no_clobber(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock_path = output / ".shard-build-transaction.lock"

            def install_foreign(_initializing: Path) -> None:
                lock_path.write_bytes(b"foreign-lock")

            with (
                patch.object(
                    shard_module,
                    "_after_lock_initializing_boundary",
                    side_effect=install_foreign,
                ),
                self.assertRaises(RuntimeError),
            ):
                shard_module._acquire_shard_transaction_lock(lock_path)
            self.assertEqual(lock_path.read_bytes(), b"foreign-lock")
            self.assertEqual(len(list(output.glob(".*.initializing"))), 1)

    def test_manifest_publication_is_no_clobber(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                manifest = output / "shard_manifest.json"

                def install_foreign(_partial: Path) -> None:
                    manifest.write_bytes(b"foreign-manifest")

                with (
                    patch.object(
                        shard_module,
                        "_after_manifest_temporary_payload_boundary",
                        side_effect=install_foreign,
                    ),
                    self.assertRaisesRegex(RuntimeError, "already exists"),
                ):
                    shard_module._write_manifest_no_replace(
                        manifest, {"schema_version": "test"}, lock
                    )
                self.assertEqual(manifest.read_bytes(), b"foreign-manifest")
                self.assertEqual(
                    len(list(output.glob(".shard_manifest.json.*.partial"))), 1
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_journal_replacement_is_bounded_after_many_updates(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            journal = output / ".shard-build-transaction.json"
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                payload = {"output_root": str(output), "progress": 0}
                identity = shard_module._write_shard_transaction(journal, payload, lock)
                for progress in range(1, 129):
                    payload = {"output_root": str(output), "progress": progress}
                    identity = shard_module._replace_shard_transaction(
                        journal, payload, identity, lock
                    )
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8")), payload
                )
                anchor = shard_module._read_lock_anchor(lock)
                self.assertEqual(anchor["journal"]["phase"], "active")
                self.assertEqual(anchor["retired_entries"], [])
                self.assertEqual(list(output.glob(".bitguard-retired-*")), [])
                self.assertEqual(
                    list(output.glob("..shard-build-transaction.json.*.partial")),
                    [],
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_lock_wal_recovers_torn_generation_and_rejects_unsafe_sequences(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock_path = output / ".shard-build-transaction.lock"
            lock = shard_module._acquire_shard_transaction_lock(lock_path)
            try:
                anchor = shard_module._read_lock_anchor(lock)
                anchor["test_generation"] = "older"
                shard_module._write_lock_anchor(lock, anchor)
                anchor["test_generation"] = "newer"
                shard_module._write_lock_anchor(lock, anchor)
                _, _, newest_slot = shard_module._read_lock_storage(lock)
                self.assertIsNotNone(newest_slot)
                assert newest_slot is not None
                lock.seek(shard_module._lock_wal_slot_offset(newest_slot) + 8 + 8 + 8)
                checksum_byte = lock.read(1)
                lock.seek(-1, os.SEEK_CUR)
                shard_module._write_all(lock, bytes([checksum_byte[0] ^ 0xFF]))
                lock.flush()
                os.fsync(lock.fileno())
                recovered = shard_module._read_lock_anchor(lock)
                self.assertEqual(recovered["test_generation"], "older")
            finally:
                shard_module._release_shard_transaction_lock(lock)

        for corruption in ("same_generation", "generation_gap"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as temp,
            ):
                output = Path(temp)
                lock_path = output / ".shard-build-transaction.lock"
                lock = shard_module._acquire_shard_transaction_lock(lock_path)
                try:
                    anchor, generation, slot = shard_module._read_lock_storage(lock)
                    forged = dict(anchor)
                    forged["forged"] = corruption
                    forged_generation = (
                        generation
                        if corruption == "same_generation"
                        else generation + 2
                    )
                    other_slot = 1 if slot == 0 else 0
                    lock.seek(shard_module._lock_wal_slot_offset(other_slot))
                    shard_module._write_all(
                        lock,
                        shard_module._lock_wal_record(forged, forged_generation),
                    )
                    lock.flush()
                    os.fsync(lock.fileno())
                    with self.assertRaisesRegex(
                        RuntimeError, "generations conflict|generation gap"
                    ):
                        shard_module._read_lock_storage(lock)
                finally:
                    shard_module._release_shard_transaction_lock(lock)

    def test_lock_wal_capacity_rejection_preserves_prior_generation(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                before = shard_module._read_lock_anchor(lock)
                before["v2_generation"] = "authoritative"
                shard_module._write_lock_anchor(lock, before)
                oversized = dict(before)
                oversized["oversized"] = (
                    "x" * shard_module._SHARD_LOCK_WAL_SLOT_CAPACITY
                )
                with self.assertRaisesRegex(RuntimeError, "fixed WAL capacity"):
                    shard_module._write_lock_anchor(lock, oversized)
                self.assertEqual(shard_module._read_lock_anchor(lock), before)
                lock.seek(1)
                shard_module._write_all(
                    lock,
                    shard_module.canonical_json_bytes({"legacy": "must-not-win"})
                    + b"\n",
                )
                lock.flush()
                os.fsync(lock.fileno())
                self.assertEqual(shard_module._read_lock_anchor(lock), before)
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_unanchored_replacement_payload_cannot_authorize_retirement(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            journal = output / ".shard-build-transaction.json"
            old_payload = {"output_root": str(output), "progress": 0}
            journal.write_bytes(shard_module.canonical_json_bytes(old_payload) + b"\n")
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            debt = shard_module._retired_entry_path(output, "file")
            try:
                old_record = shard_module._anchored_file_record(journal, old_payload)
                malicious = {
                    "output_root": str(output),
                    "cleanup_quarantine_intent": {
                        "kind": "file",
                        "quarantine_path": str(debt),
                    },
                }
                anchor = shard_module._read_lock_anchor(lock)
                anchor["journal"] = {
                    "phase": "replace_intended",
                    "accepted": [old_record],
                    "replacement": {
                        "journal_path": str(output / "wrong-journal.json"),
                        "old": old_record,
                        "stable_instance": shard_module._entry_instance(journal),
                        "new_payload": malicious,
                        "new_fingerprint": shard_module.stable_fingerprint(malicious),
                    },
                }
                shard_module._write_lock_anchor(lock, anchor)
                debt.write_bytes(b"foreign")
                with self.assertRaisesRegex(
                    RuntimeError, "unrecorded retired shard entry"
                ):
                    shard_module._read_lock_anchor(lock)
            finally:
                debt.unlink(missing_ok=True)
                shard_module._release_shard_transaction_lock(lock)

    def test_anchored_replacement_authorizes_torn_journal_recovery(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            journal = output / ".shard-build-transaction.json"
            old_payload = {"output_root": str(output), "progress": 0}
            journal.write_bytes(shard_module.canonical_json_bytes(old_payload) + b"\n")
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            debt = shard_module._retired_entry_path(output, "file")
            try:
                old_record = shard_module._anchored_file_record(journal, old_payload)
                new_payload = {
                    "output_root": str(output),
                    "progress": 1,
                    "cleanup_quarantine_intent": {
                        "kind": "file",
                        "quarantine_path": str(debt),
                    },
                }
                anchor = shard_module._read_lock_anchor(lock)
                anchor["journal"] = {
                    "phase": "rewrite_started",
                    "accepted": [old_record],
                    "replacement": {
                        "journal_path": str(journal),
                        "old": old_record,
                        "stable_instance": old_record["identity"][:3],
                        "new_payload": new_payload,
                        "new_fingerprint": shard_module.stable_fingerprint(new_payload),
                    },
                }
                shard_module._write_lock_anchor(lock, anchor)
                debt.write_bytes(b"authenticated-debt")
                with journal.open("r+b") as handle:
                    handle.seek(0)
                    handle.write(b"{torn")
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())

                self.assertEqual(
                    shard_module._read_lock_anchor(lock)["journal"]["phase"],
                    "rewrite_started",
                )
                shard_module._complete_journal_replacement(journal, lock)
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8")), new_payload
                )
            finally:
                debt.unlink(missing_ok=True)
                shard_module._release_shard_transaction_lock(lock)

    def test_full_length_checksum_torn_initializer_is_skipped_not_accepted(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock_path = output / ".shard-build-transaction.lock"
            torn = output / f".{lock_path.name}.{'b' * 32}.initializing"
            with torn.open("xb") as handle:
                status = os.fstat(handle.fileno())
                payload = {
                    "schema_version": shard_module._SHARD_LOCK_SCHEMA,
                    "lock_instance": shard_module._status_instance(status),
                    "lock_path": str(lock_path),
                    "owner_token": "c" * 32,
                    "journal": None,
                    "owner": None,
                    "manifest": None,
                    "temporaries": [],
                    "retiring_initializers": [],
                    "retired_entries": [],
                    "private_trash": shard_module._ensure_private_trash(output),
                }
                shard_module._write_all(
                    handle,
                    b"\0" + shard_module._lock_wal_record(payload, 1),
                )
                handle.flush()
                os.fsync(handle.fileno())
            with torn.open("r+b") as handle:
                handle.seek(1 + 8 + 8 + 8)
                checksum = handle.read(1)
                handle.seek(-1, os.SEEK_CUR)
                handle.write(bytes([checksum[0] ^ 0xFF]))
                handle.flush()
                os.fsync(handle.fileno())
            expected = torn.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "initialization is incomplete"):
                shard_module._read_initialized_lock(torn, lock_path)

            lock = shard_module._acquire_shard_transaction_lock(lock_path)
            try:
                self.assertEqual(torn.read_bytes(), expected)
                self.assertNotEqual(
                    shard_module.file_identity(torn),
                    shard_module.file_identity(lock_path),
                )
                self.assertEqual(
                    shard_module._read_lock_anchor(lock)["lock_path"], str(lock_path)
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_torn_initializer_is_preserved_while_valid_initializer_wins(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            lock_path = output / ".shard-build-transaction.lock"
            torn = output / f".{lock_path.name}.{'a' * 32}.initializing"
            torn.write_bytes(b"\0" + shard_module._SHARD_LOCK_WAL_MAGIC[:4])
            expected = torn.read_bytes()
            lock = shard_module._acquire_shard_transaction_lock(lock_path)
            try:
                self.assertEqual(torn.read_bytes(), expected)
                self.assertEqual(
                    shard_module._read_lock_anchor(lock)["lock_path"], str(lock_path)
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_private_trash_absence_advances_only_for_authenticated_intent(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                shard_module, "_uses_private_trash_retirement", return_value=True
            ),
        ):
            output = Path(temp)
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                debt = shard_module._retired_entry_path(output, "file")
                anchor = shard_module._read_lock_anchor(lock)
                anchor["temporaries"] = [
                    {"phase": "retiring", "quarantine_path": str(debt)}
                ]
                shard_module._write_lock_anchor(lock, anchor)
                debt.write_bytes(b"owned")
                expected = shard_module.file_identity(debt)
                with (
                    patch.object(
                        shard_module,
                        "_after_private_trash_entry_removal_boundary",
                        side_effect=RuntimeError("crash-after-private-unlink"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "crash-after-private-unlink"),
                ):
                    shard_module._retire_quarantined_entry(
                        debt,
                        expected,
                        kind="file",
                        transaction_lock=lock,
                    )
                self.assertFalse(debt.exists())
                self.assertEqual(
                    shard_module._retire_quarantined_entry(
                        debt,
                        expected,
                        kind="file",
                        transaction_lock=lock,
                    ),
                    "gone",
                )

                foreign_debt = shard_module._retired_entry_path(output, "file")
                anchor = shard_module._read_lock_anchor(lock)
                anchor["temporaries"] = [
                    {
                        "phase": "retiring",
                        "quarantine_path": str(foreign_debt),
                    }
                ]
                shard_module._write_lock_anchor(lock, anchor)
                foreign_debt.write_bytes(b"owned-second")
                foreign_expected = shard_module.file_identity(foreign_debt)
                with (
                    patch.object(
                        shard_module,
                        "_after_private_trash_entry_removal_boundary",
                        side_effect=RuntimeError("crash-after-private-unlink"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "crash-after-private-unlink"),
                ):
                    shard_module._retire_quarantined_entry(
                        foreign_debt,
                        foreign_expected,
                        kind="file",
                        transaction_lock=lock,
                    )
                foreign_debt.write_bytes(b"foreign-replacement")
                with self.assertRaisesRegex(
                    RuntimeError, "quarantined shard file identity changed"
                ):
                    shard_module._retire_quarantined_entry(
                        foreign_debt,
                        foreign_expected,
                        kind="file",
                        transaction_lock=lock,
                    )
                self.assertEqual(foreign_debt.read_bytes(), b"foreign-replacement")
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_private_trash_retirement_is_bounded_without_ledger_growth(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                shard_module, "_uses_private_trash_retirement", return_value=True
            ),
        ):
            output = Path(temp)
            lock_path = output / ".shard-build-transaction.lock"
            lock = shard_module._acquire_shard_transaction_lock(lock_path)
            try:
                for index in range(32):
                    debt = shard_module._retired_entry_path(output, "file")
                    anchor = shard_module._read_lock_anchor(lock)
                    anchor["temporaries"] = [
                        {
                            "phase": "retiring",
                            "quarantine_path": str(debt),
                        }
                    ]
                    shard_module._write_lock_anchor(lock, anchor)
                    debt.write_bytes(f"debt-{index}".encode("ascii"))
                    outcome = shard_module._retire_quarantined_entry(
                        debt,
                        shard_module.file_identity(debt),
                        kind="file",
                        transaction_lock=lock,
                    )
                    self.assertEqual(outcome, "gone")
                    anchor = shard_module._read_lock_anchor(lock)
                    anchor["temporaries"] = []
                    shard_module._write_lock_anchor(lock, anchor)
                final_anchor = shard_module._read_lock_anchor(lock)
                self.assertEqual(final_anchor["retired_entries"], [])
                self.assertEqual(
                    list((output / shard_module._SHARD_PRIVATE_TRASH).iterdir()), []
                )
                self.assertLessEqual(
                    lock_path.stat().st_size,
                    1
                    + shard_module._SHARD_LOCK_WAL_SLOT_COUNT
                    * shard_module._SHARD_LOCK_WAL_SLOT_CAPACITY,
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_journal_rewrite_rejects_a_final_boundary_replacement(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            journal = output / ".shard-build-transaction.json"
            displaced = output / "owned-journal.json"
            replacement = output / "foreign-journal.json"
            replacement.write_bytes(b"foreign-journal")
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                old_payload = {"output_root": str(output), "progress": 0}
                identity = shard_module._write_shard_transaction(
                    journal, old_payload, lock
                )

                def replace_at_open(_journal: Path) -> None:
                    journal.replace(displaced)
                    replacement.replace(journal)

                with (
                    patch.object(
                        shard_module,
                        "_before_journal_inplace_rewrite_boundary",
                        side_effect=replace_at_open,
                    ),
                    self.assertRaisesRegex(RuntimeError, "identity changed"),
                ):
                    shard_module._replace_shard_transaction(
                        journal,
                        {"output_root": str(output), "progress": 1},
                        identity,
                        lock,
                    )
                self.assertEqual(journal.read_bytes(), b"foreign-journal")
                self.assertEqual(
                    json.loads(displaced.read_text(encoding="utf-8")), old_payload
                )
            finally:
                shard_module._release_shard_transaction_lock(lock)

    def test_journal_rewrite_rejects_a_new_hardlink_before_writing(self) -> None:
        from bitguard_bnn.out_of_core import shard as shard_module

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            journal = output / ".shard-build-transaction.json"
            alias = output / "journal-alias.json"
            lock = shard_module._acquire_shard_transaction_lock(
                output / ".shard-build-transaction.lock"
            )
            try:
                old_payload = {"output_root": str(output), "progress": 0}
                identity = shard_module._write_shard_transaction(
                    journal, old_payload, lock
                )

                def add_hardlink(_journal: Path) -> None:
                    os.link(journal, alias)

                with (
                    patch.object(
                        shard_module,
                        "_before_journal_inplace_payload_write_boundary",
                        side_effect=add_hardlink,
                    ),
                    self.assertRaisesRegex(RuntimeError, "identity changed"),
                ):
                    shard_module._replace_shard_transaction(
                        journal,
                        {"output_root": str(output), "progress": 1},
                        identity,
                        lock,
                    )
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8")), old_payload
                )
                self.assertEqual(alias.read_bytes(), journal.read_bytes())
            finally:
                shard_module._release_shard_transaction_lock(lock)


if __name__ == "__main__":
    unittest.main()
