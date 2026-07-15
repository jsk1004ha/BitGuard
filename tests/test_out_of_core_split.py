from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from bitguard_bnn.out_of_core.manifest import SplitPlan
from bitguard_bnn.out_of_core.source import NormalizedChunk


def _config(
    strategy: str,
    *,
    held_out_devices: list[str] | None = None,
    held_out_attacks: list[str] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    return {
        "dataset": {"path": "C:/machine-specific/data.csv"},
        "split": {
            "strategy": strategy,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "seed": seed,
            "held_out_devices": held_out_devices or [],
            "held_out_attacks": held_out_attacks or [],
        },
    }


def _rows(count: int = 20) -> list[dict[str, object]]:
    labels = ["benign", "scan_like", "flood_like", "exfil_like"]
    attacks = ["benign", "scan", "flood", "exfil"]
    result: list[dict[str, object]] = []
    for index in range(count):
        label = labels[index % len(labels)]
        result.append(
            {
                "row_uid": f"{index:064x}",
                "source_file": f"device-{index % 3}/part.csv",
                "sequence_index": index,
                "behavior_label": label,
                "raw_attack": attacks[index % len(attacks)],
                "device_id": f"device-{index % 3}",
                "timestamp": float((index * 7) % count),
                "feature": float(index),
            }
        )
    return result


def _chunks(
    rows: list[dict[str, object]], sizes: tuple[int, ...] = (5, 3, 7)
) -> list[NormalizedChunk]:
    chunks: list[NormalizedChunk] = []
    offset = 0
    size_index = 0
    while offset < len(rows):
        size = sizes[size_index % len(sizes)]
        group = rows[offset : offset + size]
        chunks.append(
            NormalizedChunk(
                pd.DataFrame(group),
                str(group[0]["source_file"]),
                int(str(group[0]["sequence_index"])),
            )
        )
        offset += len(group)
        size_index += 1
    return chunks


def _membership(plan: SplitPlan) -> pd.DataFrame:
    return pq.read_table(plan.membership_path).to_pandas()


class OutOfCoreSplitTests(unittest.TestCase):
    def _build(
        self,
        rows: list[dict[str, object]],
        config: dict[str, Any],
        output: Path,
        *,
        sizes: tuple[int, ...] = (5, 3, 7),
        max_rows_per_run: int = 4,
        merge_fan_in: int = 32,
        source_manifest_fingerprint: str | None = None,
    ) -> SplitPlan:
        from bitguard_bnn.out_of_core.split import build_split_plan

        return build_split_plan(
            _chunks(rows, sizes),
            config,
            output,
            max_rows_per_run=max_rows_per_run,
            merge_fan_in=merge_fan_in,
            source_manifest_fingerprint=source_manifest_fingerprint,
        )

    def test_time_split_matches_exact_stable_reference_across_unsorted_chunks(self) -> None:
        rows = _rows(20)
        rows = rows[::2] + rows[1::2]
        with tempfile.TemporaryDirectory() as temp:
            plan = self._build(rows, _config("time"), Path(temp))
            actual = _membership(plan).set_index("row_uid")["split"].to_dict()

        ordered = sorted(rows, key=lambda row: (row["timestamp"], row["row_uid"]))
        expected = {}
        for rank, row in enumerate(ordered):
            expected[str(row["row_uid"])] = (
                "train" if rank < 14 else "validation" if rank < 17 else "test"
            )
        self.assertEqual(actual, expected)
        self.assertEqual(
            (plan.train_count, plan.validation_count, plan.test_count), (14, 3, 3)
        )

    def test_time_split_uses_uid_to_break_equal_timestamps(self) -> None:
        rows = _rows(20)
        for row in rows:
            row["timestamp"] = 1.0
        rows.reverse()
        with tempfile.TemporaryDirectory() as temp:
            membership = _membership(
                self._build(rows, _config("time"), Path(temp))
            ).set_index("row_uid")
        ordered_uids = sorted(str(row["row_uid"]) for row in rows)
        self.assertTrue((membership.loc[ordered_uids[:14], "split"] == "train").all())
        self.assertTrue(
            (membership.loc[ordered_uids[14:17], "split"] == "validation").all()
        )
        self.assertTrue((membership.loc[ordered_uids[17:], "split"] == "test").all())

    def test_random_is_reproducible_across_input_order_and_chunking(self) -> None:
        rows = _rows(80)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = self._build(rows, _config("random"), Path(first), sizes=(1,))
            right = self._build(
                list(reversed(rows)),
                _config("random"),
                Path(second),
                sizes=(13, 2),
            )
            left_frame = _membership(left)
            right_frame = _membership(right)
        pd.testing.assert_frame_equal(left_frame, right_frame)
        self.assertEqual(left.fingerprint, right.fingerprint)
        self.assertEqual(
            left_frame.groupby("split").size().to_dict(),
            {"test": 12, "train": 56, "validation": 12},
        )
        for _, group in left_frame.groupby("behavior_label"):
            self.assertEqual(group.groupby("split").size().to_dict(), {
                "test": 3,
                "train": 14,
                "validation": 3,
            })

    def test_device_holds_all_configured_devices_out_and_defaults_last_sorted(self) -> None:
        rows = _rows(60)
        with tempfile.TemporaryDirectory() as temp:
            explicit = _membership(
                self._build(
                    rows,
                    _config("device", held_out_devices=["device-1"]),
                    Path(temp),
                )
            )
        explicit = explicit.merge(
            pd.DataFrame(rows)[["row_uid", "device_id"]], on="row_uid", validate="one_to_one"
        )
        self.assertEqual(set(explicit.loc[explicit["device_id"] == "device-1", "split"]), {"test"})
        self.assertNotIn("test", set(explicit.loc[explicit["device_id"] != "device-1", "split"]))

        with tempfile.TemporaryDirectory() as temp:
            defaulted = _membership(self._build(rows, _config("device"), Path(temp)))
        defaulted = defaulted.merge(
            pd.DataFrame(rows)[["row_uid", "device_id"]], on="row_uid", validate="one_to_one"
        )
        self.assertEqual(
            set(defaulted.loc[defaulted["device_id"] == "device-2", "split"]),
            {"test"},
        )

    def test_attack_marks_held_and_existing_unknown_as_unknown_test(self) -> None:
        rows = _rows(80)
        rows[0]["behavior_label"] = "unknown_like"
        rows[0]["raw_attack"] = "mystery"
        with tempfile.TemporaryDirectory() as temp:
            plan = self._build(
                rows,
                _config("attack", held_out_attacks=["scan"]),
                Path(temp),
            )
            membership = _membership(plan)
            manifest = json.loads(
                plan.membership_path.with_suffix(".manifest.json").read_text("utf-8")
            )
        membership = membership.merge(
            pd.DataFrame(rows)[["row_uid", "raw_attack", "behavior_label"]].rename(
                columns={"behavior_label": "original_behavior_label"}
            ),
            on="row_uid",
            validate="one_to_one",
        )
        forced = membership[\
            (membership["raw_attack"] == "scan")
            | (membership["original_behavior_label"] == "unknown_like")
        ]
        self.assertTrue((forced["split"] == "test").all())
        self.assertTrue((forced["behavior_label"] == "unknown_like").all())
        self.assertEqual(
            int(
                (
                    (membership["split"] == "train")
                    & (membership["behavior_label"] == "unknown_like")
                ).sum()
            ),
            0,
        )
        self.assertGreater(
            int(
                (
                    (membership["split"] == "test")
                    & (membership["behavior_label"] != "unknown_like")
                ).sum()
            ),
            0,
        )
        self.assertEqual(manifest["checks"]["unknown_in_train"], 0)
        self.assertEqual(plan.strategy, "attack")

    def test_missing_and_nonfinite_timestamps_are_rejected(self) -> None:
        for value in (None, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                rows = _rows(20)
                rows[4]["timestamp"] = value
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    self._build(rows, _config("time"), Path(temp))
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_duplicate_uid_and_duplicate_logical_source_coordinate_are_rejected(self) -> None:
        rows = _rows(20)
        duplicate_uid = [dict(row) for row in rows]
        duplicate_uid[-1]["row_uid"] = duplicate_uid[0]["row_uid"]
        duplicate_coordinate = [dict(row) for row in rows]
        duplicate_coordinate[-1]["source_file"] = duplicate_coordinate[0]["source_file"]
        duplicate_coordinate[-1]["sequence_index"] = duplicate_coordinate[0]["sequence_index"]
        for invalid, message in (
            (duplicate_uid, "row_uid"),
            (duplicate_coordinate, "source"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    self._build(invalid, _config("random"), Path(temp))

    def test_absent_configured_group_and_empty_partition_are_rejected(self) -> None:
        rows = _rows(20)
        cases = (
            (_config("device", held_out_devices=["absent"]), "absent"),
            (_config("attack", held_out_attacks=["absent"]), "absent"),
            (_config("time"), "non-empty"),
        )
        for config, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                fixture = rows[:2] if config["split"]["strategy"] == "time" else rows
                with self.assertRaisesRegex(ValueError, message):
                    self._build(fixture, config, Path(temp))

    def test_fingerprint_is_idempotent_relocation_stable_and_content_sensitive(self) -> None:
        rows = _rows(40)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._build(
                rows,
                _config("random"),
                root,
                source_manifest_fingerprint="source-v1",
            )
            second = self._build(
                list(reversed(rows)),
                _config("random"),
                root,
                sizes=(11,),
                source_manifest_fingerprint="source-v1",
            )
            relocated_config = _config("random")
            relocated_config["dataset"]["path"] = "D:/other-machine/data.csv"
            relocated = self._build(
                rows,
                relocated_config,
                root / "relocated",
                source_manifest_fingerprint="source-v1",
            )
            mutated_rows = [dict(row) for row in rows]
            mutated_rows[0]["row_uid"] = "f" * 64
            mutated = self._build(
                mutated_rows,
                _config("random"),
                root / "mutated",
                source_manifest_fingerprint="source-v2",
            )
        self.assertEqual(first.membership_path, second.membership_path)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.fingerprint, relocated.fingerprint)
        self.assertNotEqual(first.fingerprint, mutated.fingerprint)

    def test_runs_are_bounded_cleanup_is_complete_and_concat_is_not_used(self) -> None:
        rows = _rows(47)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("pandas.concat", side_effect=AssertionError("full concat")):
                plan = self._build(
                    rows,
                    _config("random"),
                    root,
                    sizes=(47,),
                    max_rows_per_run=3,
                )
            manifest = json.loads(
                plan.membership_path.with_suffix(".manifest.json").read_text("utf-8")
            )
            published_columns = pq.read_schema(plan.membership_path).names
            leftovers = [path for path in root.rglob("*") if ".partial" in path.name]
        self.assertLessEqual(manifest["resource_usage"]["max_run_rows"], 3)
        self.assertGreater(manifest["resource_usage"]["run_count"], 1)
        self.assertGreater(manifest["resource_usage"]["temporary_bytes_peak"], 0)
        self.assertEqual(
            published_columns,
            ["row_uid", "split", "behavior_label"],
        )
        self.assertEqual(leftovers, [])

    def test_multi_pass_merge_never_exceeds_fan_in_and_preserves_singleton_carries(self) -> None:
        rows = _rows(18)
        with tempfile.TemporaryDirectory() as temp:
            plan = self._build(
                rows,
                _config("time"),
                Path(temp),
                sizes=(18,),
                max_rows_per_run=1,
                merge_fan_in=2,
            )
            manifest = json.loads(
                plan.membership_path.with_suffix(".manifest.json").read_text("utf-8")
            )
            membership = _membership(plan)
        self.assertEqual(len(membership), len(rows))
        self.assertEqual(membership["row_uid"].tolist(), sorted(membership["row_uid"]))
        self.assertLessEqual(
            manifest["resource_usage"]["max_merge_fan_in_observed"], 2
        )
        self.assertEqual(manifest["resource_usage"]["merge_fan_in_limit"], 2)

    def test_manifest_publication_failure_rolls_back_membership_and_partials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "bitguard_bnn.out_of_core.split.write_json_atomic",
                side_effect=OSError("injected manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "manifest failure"):
                    self._build(_rows(20), _config("random"), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_directory_fsync_errors_propagate_on_posix(self) -> None:
        from bitguard_bnn.out_of_core import split as split_module

        with (
            patch.object(split_module.os, "name", "posix"),
            patch.object(split_module.os, "open", return_value=123),
            patch.object(split_module.os, "fsync", side_effect=OSError("fsync failed")),
            patch.object(split_module.os, "close"),
        ):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                split_module._fsync_directory(Path("unused"))

    def test_existing_manifest_semantic_tampering_is_rejected(self) -> None:
        rows = _rows(40)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self._build(rows, _config("random"), root)
            manifest_path = plan.membership_path.with_suffix(".manifest.json")
            valid = json.loads(manifest_path.read_text("utf-8"))

            def increment_class(payload: dict[str, Any]) -> None:
                payload["class_counts"]["train"]["benign"] += 1

            def increment_source(payload: dict[str, Any]) -> None:
                source = sorted(payload["source_coverage"])[0]
                payload["source_coverage"][source]["train"] += 1

            mutations = {
                "schema_version": lambda payload: payload.__setitem__(
                    "schema_version", "tampered"
                ),
                "class_counts": increment_class,
                "source_coverage": increment_source,
                "schema_descriptor": lambda payload: payload.__setitem__(
                    "schema", [{"name": "row_uid", "type": "string", "nullable": True}]
                ),
                "schema_fingerprint": lambda payload: payload.__setitem__(
                    "schema_fingerprint", "0" * 64
                ),
                "algorithm": lambda payload: payload["algorithm_versions"].__setitem__(
                    "split", "tampered"
                ),
                "config": lambda payload: payload.__setitem__(
                    "config_signature", "0" * 64
                ),
                "checks": lambda payload: payload["checks"].__setitem__(
                    "unknown_in_train", 99
                ),
                "semantic_fingerprint": lambda payload: payload.__setitem__(
                    "semantic_fingerprint", "0" * 64
                ),
                "membership_logical": lambda payload: payload["membership"].__setitem__(
                    "logical_digest", "0" * 64
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(field=name):
                    payload = copy.deepcopy(valid)
                    mutate(payload)
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "immutable split output"):
                        self._build(rows, _config("random"), root)
            manifest_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "immutable split output"):
                self._build(rows, _config("random"), root)
            manifest_path.write_text(json.dumps(valid), encoding="utf-8")

    def test_manifest_schema_descriptor_is_ordered_and_preserves_nullability(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan = self._build(_rows(20), _config("random"), Path(temp))
            manifest = json.loads(
                plan.membership_path.with_suffix(".manifest.json").read_text("utf-8")
            )
        self.assertEqual(
            manifest["schema"],
            [
                {"name": "row_uid", "type": "string", "nullable": False},
                {"name": "split", "type": "string", "nullable": False},
                {"name": "behavior_label", "type": "string", "nullable": False},
            ],
        )
        self.assertEqual(len(manifest["semantic_fingerprint"]), 64)

    def test_post_rename_manifest_fsync_failure_rolls_back_both_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "bitguard_bnn.out_of_core.manifest._fsync_directory",
                side_effect=OSError("injected post-rename fsync failure"),
            ):
                with self.assertRaisesRegex(OSError, "post-rename fsync failure"):
                    self._build(_rows(20), _config("random"), root)
            self.assertEqual(list(root.iterdir()), [])

    def test_success_never_ignores_work_tree_cleanup_failure(self) -> None:
        import shutil

        real_rmtree: Any = shutil.rmtree

        def fail_only_when_strict(path: object, *args: object, **kwargs: object) -> None:
            if Path(str(path)).name.startswith(".split-"):
                if kwargs.get("ignore_errors"):
                    return
                raise OSError("injected work cleanup failure")
            real_rmtree(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch(
                "bitguard_bnn.out_of_core.split.shutil.rmtree",
                side_effect=fail_only_when_strict,
            ):
                with self.assertRaisesRegex(OSError, "work cleanup failure"):
                    self._build(_rows(20), _config("random"), root)
            self.assertFalse(any(root.glob("split-membership-*")))

    def test_primary_error_remains_primary_when_cleanup_also_fails(self) -> None:
        import shutil

        real_rmtree: Any = shutil.rmtree

        def fail_only_when_strict(path: object, *args: object, **kwargs: object) -> None:
            if Path(str(path)).name.startswith(".split-"):
                if kwargs.get("ignore_errors"):
                    return
                raise OSError("secondary cleanup failure")
            real_rmtree(path, *args, **kwargs)

        def failing_chunks() -> Any:
            yield _chunks(_rows(4), sizes=(4,))[0]
            raise ValueError("primary inspection failure")

        from bitguard_bnn.out_of_core.split import build_split_plan

        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "bitguard_bnn.out_of_core.split.shutil.rmtree",
                side_effect=fail_only_when_strict,
            ):
                with self.assertRaisesRegex(ValueError, "primary inspection failure") as caught:
                    build_split_plan(
                        failing_chunks(),
                        _config("random"),
                        Path(temp),
                        max_rows_per_run=4,
                    )
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_idempotent_cleanup_fsync_fault_never_removes_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self._build(_rows(20), _config("random"), root)
            manifest_path = plan.membership_path.with_suffix(".manifest.json")
            membership_before = plan.membership_path.read_bytes()
            manifest_before = manifest_path.read_bytes()
            with patch(
                "bitguard_bnn.out_of_core.split._fsync_directory",
                side_effect=OSError("injected idempotent cleanup fsync failure"),
            ):
                with self.assertRaisesRegex(OSError, "idempotent cleanup fsync failure"):
                    self._build(_rows(20), _config("random"), root)
            self.assertEqual(plan.membership_path.read_bytes(), membership_before)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertFalse(any(".partial" in path.name for path in root.iterdir()))

    def test_post_rename_membership_identity_fault_rolls_back_owned_file(self) -> None:
        from bitguard_bnn.out_of_core import split as split_module

        real_file_identity = split_module.file_identity

        def fail_only_for_final_membership(path: Path) -> Any:
            if path.name.startswith("split-membership-"):
                raise PermissionError("injected post-rename identity failure")
            return real_file_identity(path)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(
                split_module,
                "file_identity",
                side_effect=fail_only_for_final_membership,
            ):
                with self.assertRaisesRegex(PermissionError, "identity failure"):
                    self._build(_rows(20), _config("random"), root)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
