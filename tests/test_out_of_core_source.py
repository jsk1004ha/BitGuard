from __future__ import annotations

import copy
import inspect
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.data import (
    LoadedDataset,
    _FrameAccumulator,
    load_botiot,
    load_dataset,
    load_generic_csv,
    load_nbaiot,
)
from bitguard_bnn.out_of_core import source as source_module
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

    def test_csv_and_byte_pass_counts_are_independent_of_labels_and_chunks(
        self,
    ) -> None:
        for label_count in (1, 6):
            for chunk_size in (1, 7):
                for class_cap, expected_passes in ((None, 2), (2, 3)):
                    with (
                        self.subTest(
                            label_count=label_count,
                            chunk_size=chunk_size,
                            class_cap=class_cap,
                        ),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory)
                        path = root / "rows.csv"
                        labels = [f"label_{index}" for index in range(label_count)]
                        pd.DataFrame(
                            {
                                "behavior_label": labels * 4,
                                "timestamp": list(range(label_count * 4)),
                                "x": list(range(label_count * 4)),
                            }
                        ).to_csv(path, index=False)
                        config = _base_config(root, "csv", "rows.csv")
                        config["dataset"]["chunk_size"] = chunk_size
                        config["dataset"]["max_rows_per_class"] = class_cap
                        csv_calls = 0
                        hashed_bytes = 0
                        original_csv = source_module.pd.read_csv
                        original_read = source_module._HashingReader.read

                        def count_csv(*args, **kwargs):
                            nonlocal csv_calls
                            if kwargs.get("chunksize") is not None:
                                csv_calls += 1
                            return original_csv(*args, **kwargs)

                        def count_bytes(reader, size: int = -1):
                            nonlocal hashed_bytes
                            block = original_read(reader, size)
                            hashed_bytes += len(block)
                            return block

                        with (
                            patch.object(source_module.pd, "read_csv", count_csv),
                            patch.object(
                                source_module._HashingReader,
                                "read",
                                count_bytes,
                            ),
                        ):
                            list(iter_normalized_chunks(config))

                        self.assertEqual(csv_calls, expected_passes)
                        self.assertEqual(
                            hashed_bytes,
                            path.stat().st_size * expected_passes,
                        )

    def test_nbaiot_uses_only_plan_and_emit_byte_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset" / "device"
            dataset.mkdir(parents=True)
            path = dataset / "benign.csv"
            pd.DataFrame({"x": list(range(9))}).to_csv(path, index=False)
            config = _base_config(root, "nbaiot", "dataset")
            config["dataset"]["chunk_size"] = 2
            csv_calls = 0
            hashed_bytes = 0
            original_csv = source_module.pd.read_csv
            original_read = source_module._HashingReader.read

            def count_csv(*args, **kwargs):
                nonlocal csv_calls
                if kwargs.get("chunksize") is not None:
                    csv_calls += 1
                return original_csv(*args, **kwargs)

            def count_bytes(reader, size: int = -1):
                nonlocal hashed_bytes
                block = original_read(reader, size)
                hashed_bytes += len(block)
                return block

            with (
                patch.object(source_module.pd, "read_csv", count_csv),
                patch.object(
                    source_module._HashingReader,
                    "read",
                    count_bytes,
                ),
            ):
                list(iter_normalized_chunks(config))

            self.assertEqual(csv_calls, 2)
            self.assertEqual(hashed_bytes, path.stat().st_size * 2)

    def test_source_and_snapshot_byte_reads_are_measured_separately(self) -> None:
        for class_cap, snapshot_passes in ((None, 1), (2, 2)):
            with (
                self.subTest(class_cap=class_cap),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = root / "rows.csv"
                pd.DataFrame(
                    {
                        "behavior_label": ["benign", "scan_like"] * 4,
                        "x": list(range(8)),
                    }
                ).to_csv(path, index=False)
                config = _base_config(root, "csv", "rows.csv")
                config["dataset"]["max_rows_per_class"] = class_cap
                source_bytes = 0
                snapshot_bytes = 0
                original_read = source_module._HashingReader.read

                def count_bytes(reader, size: int = -1):
                    nonlocal source_bytes, snapshot_bytes
                    block = original_read(reader, size)
                    if getattr(reader, "origin", "source") == "source":
                        source_bytes += len(block)
                    else:
                        snapshot_bytes += len(block)
                    return block

                with patch.object(
                    source_module._HashingReader, "read", count_bytes
                ):
                    list(iter_normalized_chunks(config))

                self.assertEqual(source_bytes, path.stat().st_size)
                self.assertEqual(
                    snapshot_bytes, path.stat().st_size * snapshot_passes
                )

    def test_selection_state_is_not_duplicated_in_plan_dataclasses(self) -> None:
        self.assertNotIn(
            "selected_rows", source_module._FilePlan.__dataclass_fields__
        )
        self.assertNotIn(
            "retained_uids", source_module._IterationPlan.__dataclass_fields__
        )
        self.assertNotIn(
            "materialization_order",
            source_module._IterationPlan.__dataclass_fields__,
        )

    def test_disk_selection_index_is_removed_after_success_and_error(self) -> None:
        for fail_during_emit in (False, True):
            with (
                self.subTest(fail_during_emit=fail_during_emit),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = root / "rows.csv"
                index_path = root / "selection.sqlite3"
                pd.DataFrame(
                    {
                        "behavior_label": ["benign", "scan_like"] * 5,
                        "x": list(range(10)),
                    }
                ).to_csv(path, index=False)
                config = _base_config(root, "csv", "rows.csv")
                config["dataset"]["max_rows_per_file"] = 4
                patches = [
                    patch.object(
                        source_module,
                        "_new_selection_path",
                        return_value=index_path,
                        create=True,
                    )
                ]
                if fail_during_emit:
                    patches.append(
                        patch.object(
                            source_module,
                            "_iter_planned_chunks",
                            side_effect=RuntimeError("emit failed"),
                        )
                    )

                with patches[0] as path_factory:
                    if fail_during_emit:
                        with patches[1], self.assertRaisesRegex(
                            RuntimeError, "emit failed"
                        ):
                            list(iter_normalized_chunks(config))
                    else:
                        list(iter_normalized_chunks(config))

                path_factory.assert_called_once_with()
                self.assertFalse(index_path.exists())

    def test_logical_identity_is_stable_when_the_source_tree_moves(self) -> None:
        snapshots: list[tuple[list[int], list[str], list[str]]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_bytes = pd.DataFrame(
                {
                    "behavior_label": ["benign", "scan_like"] * 10,
                    "x": list(range(20)),
                }
            ).to_csv(index=False).encode("utf-8")
            for location in (root / "first", root / "relocated" / "second"):
                dataset = location / "dataset"
                dataset.mkdir(parents=True)
                (dataset / "rows.csv").write_bytes(csv_bytes)
                config = _base_config(location, "csv", "dataset/*.csv")
                config["dataset"]["max_rows_per_file"] = 5

                loaded = load_dataset(config)

                snapshots.append(
                    (
                        sorted(loaded.frame["sequence_index"].astype(int).tolist()),
                        sorted(loaded.frame["row_uid"].astype(str).tolist()),
                        sorted(loaded.frame["source_file"].astype(str).unique()),
                    )
                )

        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[0][2], ["rows.csv"])

    def test_logical_identity_changes_when_bytes_change_at_the_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            config = _base_config(root, "csv", "rows.csv")
            pd.DataFrame(
                {"behavior_label": ["benign"], "x": [1]}
            ).to_csv(path, index=False)
            first = load_dataset(config)
            pd.DataFrame(
                {"behavior_label": ["benign"], "x": [2]}
            ).to_csv(path, index=False)

            second = load_dataset(config)

        self.assertNotEqual(
            first.frame["row_uid"].tolist(), second.frame["row_uid"].tolist()
        )

    def test_iterator_rejects_append_after_the_planning_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            pd.DataFrame(
                {"behavior_label": ["benign"] * 6, "x": list(range(6))}
            ).to_csv(path, index=False)
            config = _base_config(root, "csv", "rows.csv")
            config["dataset"]["max_loaded_rows"] = 6
            iterator = iter_normalized_chunks(config)
            next(iterator)

            with path.open("a", encoding="utf-8") as handle:
                handle.write("benign,999\n")

            with self.assertRaisesRegex(RuntimeError, "source changed"):
                list(iterator)

    def test_iterator_rejects_same_size_rewrite_with_restored_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            pd.DataFrame(
                {"behavior_label": ["benign"] * 6, "x": list(range(6))}
            ).to_csv(path, index=False)
            config = _base_config(root, "csv", "rows.csv")
            original = path.read_bytes()
            rewritten = original.replace(b"benign,3", b"benign,9", 1)
            self.assertEqual(len(rewritten), len(original))
            initial_stat = path.stat()
            original_iter = source_module._iter_planned_chunks

            def rewrite_before_emit(config: dict, plan: object):
                path.write_bytes(rewritten)
                os.utime(
                    path,
                    ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
                )
                yield from original_iter(config, plan)

            with (
                patch.object(
                    source_module,
                    "_iter_planned_chunks",
                    rewrite_before_emit,
                ),
                self.assertRaisesRegex(RuntimeError, "source changed"),
            ):
                list(iter_normalized_chunks(config))

    def test_rewrite_cannot_escape_a_changed_chunk_before_sha_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            header = b"behavior_label,x\n"
            original = header + b"benign," + b"1" * 82 + b"\n"
            original += b"benign," + b"2" * 82 + b"\n"
            rewritten = header + b"benign,1\n" * 20
            self.assertEqual(len(rewritten), len(original))
            path.write_bytes(original)
            initial_stat = path.stat()
            config = _base_config(root, "csv", "rows.csv")
            config["dataset"]["max_loaded_rows"] = 2
            original_iter = source_module._iter_planned_chunks

            def rewrite_before_emit(config: dict, plan: object):
                path.write_bytes(rewritten)
                os.utime(
                    path,
                    ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
                )
                yield from original_iter(config, plan)

            iterator = iter_normalized_chunks(config)
            try:
                with (
                    patch.object(
                        source_module,
                        "_iter_planned_chunks",
                        rewrite_before_emit,
                    ),
                    self.assertRaisesRegex(RuntimeError, "source changed"),
                ):
                    next(iterator)
            finally:
                iterator.close()

    def test_verified_snapshot_is_removed_on_normal_and_early_close(self) -> None:
        for close_early in (False, True):
            with (
                self.subTest(close_early=close_early),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                snapshot_root = root / "private-snapshots"
                pd.DataFrame(
                    {
                        "behavior_label": ["benign"] * 5,
                        "x": list(range(5)),
                    }
                ).to_csv(root / "rows.csv", index=False)
                config = _base_config(root, "csv", "rows.csv")
                iterator = iter_normalized_chunks(config)

                with patch.object(
                    source_module,
                    "_new_snapshot_root",
                    return_value=snapshot_root,
                    create=True,
                ) as root_factory:
                    if close_early:
                        try:
                            next(iterator)
                            self.assertTrue(snapshot_root.is_dir())
                            snapshots = list(
                                snapshot_root.glob("*.verified.csv")
                            )
                            self.assertEqual(len(snapshots), 1)
                            self.assertFalse(
                                snapshots[0].stat().st_mode & stat.S_IWRITE
                            )
                        finally:
                            iterator.close()
                    else:
                        list(iterator)

                root_factory.assert_called_once_with()
                self.assertFalse(snapshot_root.exists())

    def test_snapshot_and_absolute_source_paths_do_not_escape_provenance(
        self,
    ) -> None:
        def contains_path(value: object, path: str) -> bool:
            if isinstance(value, str):
                return path in value
            if isinstance(value, dict):
                return any(
                    contains_path(key, path) or contains_path(item, path)
                    for key, item in value.items()
                )
            if isinstance(value, (list, tuple)):
                return any(contains_path(item, path) for item in value)
            return False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            snapshot_root = root / "private-snapshots"
            pd.DataFrame(
                {"behavior_label": ["benign"], "x": [1]}
            ).to_csv(path, index=False)
            config = _base_config(root, "csv", "rows.csv")

            with patch.object(
                source_module,
                "_new_snapshot_root",
                return_value=snapshot_root,
            ):
                loaded = load_dataset(config)

            self.assertFalse(
                contains_path(loaded.provenance, str(snapshot_root))
            )
            self.assertFalse(
                contains_path(loaded.provenance, str(path.resolve()))
            )

    def test_verified_snapshot_is_removed_on_parse_and_emit_errors(self) -> None:
        for parse_error in (False, True):
            with (
                self.subTest(parse_error=parse_error),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                snapshot_root = root / "private-snapshots"
                path = root / "rows.csv"
                if parse_error:
                    path.write_text(
                        'behavior_label,x\n"unterminated,1\n',
                        encoding="utf-8",
                    )
                else:
                    pd.DataFrame(
                        {"behavior_label": ["benign"], "x": [1]}
                    ).to_csv(path, index=False)
                config = _base_config(root, "csv", "rows.csv")

                with patch.object(
                    source_module,
                    "_new_snapshot_root",
                    return_value=snapshot_root,
                    create=True,
                ) as root_factory:
                    if parse_error:
                        with self.assertRaises(pd.errors.ParserError):
                            list(iter_normalized_chunks(config))
                    else:
                        with (
                            patch.object(
                                source_module,
                                "_iter_planned_chunks",
                                side_effect=RuntimeError("emit failed"),
                            ),
                            self.assertRaisesRegex(RuntimeError, "emit failed"),
                        ):
                            list(iter_normalized_chunks(config))

                root_factory.assert_called_once_with()
                self.assertFalse(snapshot_root.exists())

    def test_iterator_rejects_same_path_replacement_before_emit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            replacement = root / "replacement.csv"
            pd.DataFrame(
                {"behavior_label": ["benign"] * 4, "x": list(range(4))}
            ).to_csv(path, index=False)
            pd.DataFrame(
                {"behavior_label": ["benign"] * 4, "x": list(range(4, 8))}
            ).to_csv(replacement, index=False)
            config = _base_config(root, "csv", "rows.csv")
            original_iter = source_module._iter_planned_chunks

            def replace_before_emit(config: dict, plan: object):
                replacement.replace(path)
                yield from original_iter(config, plan)

            with (
                patch.object(
                    source_module,
                    "_iter_planned_chunks",
                    replace_before_emit,
                ),
                self.assertRaisesRegex(RuntimeError, "source changed"),
            ):
                list(iter_normalized_chunks(config))

    def test_uncapped_iterator_still_enforces_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"x": [1, 2]}).to_csv(root / "rows.csv", index=False)
            config = _base_config(root, "csv", "rows.csv")

            with self.assertRaisesRegex(ValueError, "label column"):
                list(iter_normalized_chunks(config, apply_sampling_caps=False))

    def test_materialized_class_cap_order_matches_legacy_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, base in (("a", 0), ("b", 10)):
                (root / folder).mkdir()
                pd.DataFrame(
                    {
                        "behavior_label": [
                            "benign",
                            "scan_like",
                            "benign",
                            "scan_like",
                        ],
                        "raw_attack": ["benign", "scan", "benign", "scan"],
                        "device_id": [folder] * 4,
                        "timestamp": range(base, base + 4),
                        "x": range(base, base + 4),
                    }
                ).to_csv(root / folder / "rows.csv", index=False)
            config = _base_config(root, "csv", "**/*.csv")
            config["dataset"]["max_rows_per_class"] = 2

            legacy = load_generic_csv(config).frame
            materialized = load_dataset(config).frame

        expected = [
            ("b", 1, "scan_like"),
            ("a", 1, "scan_like"),
            ("a", 2, "benign"),
            ("b", 2, "benign"),
        ]

        def identity(frame: pd.DataFrame) -> list[tuple[str, int, str]]:
            return [
                (Path(source).parent.name, int(index), str(label))
                for source, index, label in zip(
                    frame["source_file"],
                    frame["sequence_index"],
                    frame["behavior_label"],
                )
            ]

        self.assertEqual(identity(legacy), expected)
        self.assertEqual(identity(materialized), expected)

    def test_materialized_order_preserves_groups_when_cap_is_not_exceeded(
        self,
    ) -> None:
        expected = [
            (3, "scan_like"),
            (1, "scan_like"),
            (2, "benign"),
            (0, "benign"),
        ]
        for class_cap in (100, 2):
            with (
                self.subTest(class_cap=class_cap),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                pd.DataFrame(
                    {
                        "behavior_label": [
                            "benign",
                            "scan_like",
                            "benign",
                            "scan_like",
                        ],
                        "x": range(4),
                    }
                ).to_csv(root / "rows.csv", index=False)
                config = _base_config(root, "csv", "rows.csv")
                config["dataset"]["chunk_size"] = 1
                config["dataset"]["max_rows_per_class"] = class_cap

                materialized = load_dataset(config).frame

            self.assertEqual(
                [
                    (int(index), str(label))
                    for index, label in zip(
                        materialized["sequence_index"],
                        materialized["behavior_label"],
                    )
                ],
                expected,
            )

    def test_materialized_order_matches_legacy_at_first_and_later_overflow(
        self,
    ) -> None:
        cases = (
            ("first_overflow", 1, (4,)),
            ("later_overflow", 2, (2, 4)),
        )
        for name, class_cap, file_rows in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                for file_index, rows in enumerate(file_rows):
                    pd.DataFrame(
                        {
                            "behavior_label": [
                                "benign" if index % 2 == 0 else "scan_like"
                                for index in range(rows)
                            ],
                            "x": [file_index * 10 + index for index in range(rows)],
                        }
                    ).to_csv(root / f"{file_index}.csv", index=False)
                config = _base_config(root, "csv", "*.csv")
                config["dataset"]["chunk_size"] = 1
                config["dataset"]["max_rows_per_class"] = class_cap

                source_frames: dict[str, list[pd.DataFrame]] = {}
                for chunk in iter_normalized_chunks(
                    config, apply_sampling_caps=False
                ):
                    source_frames.setdefault(
                        chunk.source_relative_path, []
                    ).append(chunk.frame)
                oracle = _FrameAccumulator(
                    class_cap, int(config["experiment"]["seed"])
                )
                for frames in source_frames.values():
                    oracle.add(pd.concat(frames, ignore_index=True))
                expected = oracle.finish(int(config["experiment"]["seed"]))

                materialized = load_dataset(config).frame

            self.assertEqual(
                materialized["row_uid"].tolist(),
                expected["row_uid"].tolist(),
            )

    def test_header_only_file_is_skipped_when_later_file_has_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00-header.csv").write_text(
                "behavior_label,timestamp,x\n", encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "behavior_label": ["benign", "scan_like"],
                    "timestamp": [1, 2],
                    "x": [3, 4],
                }
            ).to_csv(root / "01-data.csv", index=False)
            config = _base_config(root, "csv", "*.csv")

            chunks = list(iter_normalized_chunks(config))
            loaded = load_dataset(config)

        self.assertEqual(sum(len(chunk.frame) for chunk in chunks), 2)
        self.assertEqual({chunk.source_relative_path for chunk in chunks}, {"01-data.csv"})
        self.assertEqual(len(loaded.frame), 2)
        self.assertEqual(loaded.provenance["files"], 2)

    def test_header_only_schema_frames_influence_only_uncapped_materialization(
        self,
    ) -> None:
        cases = (
            (
                "csv",
                "behavior_label,b,z\n",
                {"behavior_label": ["benign"], "a": [1], "b": [2]},
                load_generic_csv,
            ),
            (
                "nbaiot",
                "b,z\n",
                {"a": [1], "b": [2]},
                load_nbaiot,
            ),
            (
                "botiot",
                "category,b,z\n",
                {"category": ["Normal"], "a": [1], "b": [2]},
                load_botiot,
            ),
        )
        metadata = [
            "dataset",
            "source_file",
            "sequence_index",
            "device_id",
            "raw_attack",
            "behavior_label",
            "timestamp",
            "row_uid",
        ]
        for dataset_type, header, row, typed_loader in cases:
            for class_cap, expected_columns, expected_features in (
                (None, ["b", "z", *metadata, "a"], ["b", "a"]),
                (2, ["a", "b", *metadata], ["a", "b"]),
            ):
                for loader in (load_dataset, typed_loader):
                    with (
                        self.subTest(
                            dataset_type=dataset_type,
                            class_cap=class_cap,
                            loader=loader.__name__,
                        ),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory)
                        if dataset_type == "nbaiot":
                            dataset = root / "dataset"
                            dataset.mkdir()
                            path = "dataset"
                        else:
                            dataset = root
                            path = "*.csv"
                        (dataset / "00-header.csv").write_text(
                            header, encoding="utf-8"
                        )
                        pd.DataFrame(row).to_csv(
                            dataset / "01-data.csv", index=False
                        )
                        config = _base_config(root, dataset_type, path)
                        config["dataset"]["max_rows_per_class"] = class_cap

                        loaded = loader(config)

                        self.assertEqual(
                            loaded.frame.columns.tolist(), expected_columns
                        )
                        self.assertEqual(
                            loaded.feature_columns, expected_features
                        )
                        self.assertEqual(
                            loaded.frame[expected_features]
                            .dtypes.astype(str)
                            .tolist(),
                            ["float32"] * len(expected_features),
                        )
                        if class_cap is None:
                            self.assertEqual(str(loaded.frame["z"].dtype), "object")
                            self.assertTrue(loaded.frame["z"].isna().all())

    def test_only_header_files_report_no_numeric_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "header.csv").write_text(
                "behavior_label,timestamp,x\n", encoding="utf-8"
            )
            config = _base_config(root, "csv", "header.csv")

            with self.assertRaisesRegex(ValueError, "no numeric feature"):
                list(iter_normalized_chunks(config))

    def test_invalid_generic_header_is_enforced_before_later_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "00-invalid.csv"
            invalid.write_text("x\n", encoding="utf-8")
            pd.DataFrame(
                {"behavior_label": ["benign"], "x": [1]}
            ).to_csv(root / "01-valid.csv", index=False)
            config = _base_config(root, "csv", "*.csv")
            pattern = re.escape(
                f"label column 'behavior_label' missing from {invalid}"
            )

            for loader in (load_dataset, load_generic_csv):
                with self.subTest(loader=loader.__name__), self.assertRaisesRegex(
                    ValueError, pattern
                ):
                    loader(config)

    def test_header_only_error_order_matches_legacy_for_type_and_cap(self) -> None:
        cases = (
            ("csv", "behavior_label,x\n", load_generic_csv),
            ("nbaiot", "f1,f2\n", load_nbaiot),
            ("botiot", "category,rate\n", load_botiot),
        )
        for dataset_type, header, typed_loader in cases:
            for class_cap, expected in (
                (None, "no numeric feature columns were found"),
                (2, "dataset contains no rows"),
            ):
                for loader in (load_dataset, typed_loader):
                    with (
                        self.subTest(
                            dataset_type=dataset_type,
                            class_cap=class_cap,
                            loader=loader.__name__,
                        ),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory)
                        if dataset_type == "nbaiot":
                            dataset = root / "dataset"
                            dataset.mkdir()
                            (dataset / "header.csv").write_text(
                                header, encoding="utf-8"
                            )
                            path = "dataset"
                        else:
                            (root / "header.csv").write_text(
                                header, encoding="utf-8"
                            )
                            path = "header.csv"
                        config = _base_config(
                            root, dataset_type, path
                        )
                        config["dataset"]["max_rows_per_class"] = class_cap

                        with self.assertRaisesRegex(
                            ValueError, f"^{expected}$"
                        ):
                            loader(config)

    def test_iterator_options_are_keyword_only(self) -> None:
        parameters = inspect.signature(iter_normalized_chunks).parameters
        self.assertEqual(
            parameters["path_override"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            parameters["apply_sampling_caps"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_directory_override_uses_unique_paths_relative_to_selected_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            selected = root / "selected"
            for folder, value in (("a", 1), ("b", 2)):
                (selected / folder).mkdir(parents=True)
                pd.DataFrame(
                    {"behavior_label": ["benign"], "x": [value]}
                ).to_csv(selected / folder / "rows.csv", index=False)
            config = _base_config(project, "csv", "unused.csv")

            chunks = list(
                iter_normalized_chunks(config, path_override=selected)
            )

        self.assertEqual(
            [chunk.source_relative_path for chunk in chunks],
            ["a/rows.csv", "b/rows.csv"],
        )

    def test_duplicate_normalized_logical_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            pd.DataFrame(
                {"behavior_label": ["benign"], "x": [1]}
            ).to_csv(path, index=False)
            config = _base_config(root, "csv", "*.csv")

            with (
                patch.object(
                    source_module,
                    "resolve_csv_files",
                    return_value=(path, path),
                ),
                self.assertRaisesRegex(ValueError, "duplicate logical source path"),
            ):
                list(iter_normalized_chunks(config))

    def test_typed_loaders_delegate_to_shared_materializer(self) -> None:
        sentinel = LoadedDataset(pd.DataFrame(), [], {})
        cases = (
            (load_nbaiot, "nbaiot"),
            (load_botiot, "botiot"),
            (load_generic_csv, "csv"),
        )
        for loader, dataset_type in cases:
            with self.subTest(dataset_type=dataset_type):
                config = {"dataset": {"type": dataset_type}}
                with patch(
                    "bitguard_bnn.out_of_core.source.load_normalized_dataset",
                    return_value=sentinel,
                ) as shared:
                    self.assertIs(loader(config), sentinel)
                shared.assert_called_once_with(
                    config,
                    None,
                    dataset_type=dataset_type,
                )


if __name__ == "__main__":
    unittest.main()
