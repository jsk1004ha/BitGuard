from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitguard_bnn.bootstrap.inspect import (
    SchemaInspectionError,
    _BoundedLines,
    inspect_csv_dataset,
)


class SchemaInspectionTest(unittest.TestCase):
    def test_physical_line_read_is_limited_before_allocation(self) -> None:
        class RecordingText(io.StringIO):
            def __init__(self, value: str) -> None:
                super().__init__(value)
                self.requested_sizes: list[int] = []

            def readline(self, size: int = -1) -> str:
                self.requested_sizes.append(size)
                return super().readline(size)

        handle = RecordingText("x" * 100 + "\n")
        lines = _BoundedLines(handle, 16)

        with self.assertRaisesRegex(SchemaInspectionError, "record exceeds"):
            next(lines)

        self.assertEqual(handle.requested_sizes, [17])
        self.assertEqual(handle.tell(), 17)

    def test_nbaiot_inspection_is_chunked_deterministic_and_json_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "device_b" / "benign_traffic.csv"
            second = root / "device_a" / "gafgyt_attacks" / "tcp.csv"
            first.parent.mkdir()
            second.parent.mkdir(parents=True)
            first.write_text("mean,std\n1,2\n3,\n", encoding="utf-8")
            second.write_text("std,mean\n5,4\n7,6\n", encoding="utf-8")

            report = inspect_csv_dataset("nbaiot", root, chunk_size=1)

            self.assertEqual(report.total_rows, 4)
            self.assertEqual(report.feature_columns, ("mean", "std"))
            self.assertEqual(report.class_counts, (("benign", 2), ("flood_like", 2)))
            self.assertEqual(report.unique_devices, 2)
            self.assertEqual([item.relative_path for item in report.files], [
                "device_a/gafgyt_attacks/tcp.csv",
                "device_b/benign_traffic.csv",
            ])
            self.assertEqual(report.rejected_rows, 0)
            self.assertEqual(report.as_dict()["total_rows"], 4)

    def test_botiot_required_metadata_labels_devices_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "flows.csv"
            path.write_text(
                "category,subcategory,saddr,stime,bytes,rate\n"
                "Normal,Normal,10.0.0.1,1.5,100,2.0\n"
                "DDoS,TCP,10.0.0.2,2019-01-01T00:00:00Z,200,3.0\n",
                encoding="utf-8",
            )
            report = inspect_csv_dataset(
                "botiot",
                root,
                required_columns=("category", "subcategory", "saddr", "stime"),
                chunk_size=1,
            )
            self.assertEqual(report.feature_columns, ("bytes", "rate"))
            self.assertEqual(report.class_counts, (("benign", 1), ("flood_like", 1)))
            self.assertEqual(report.unique_devices, 2)
            self.assertEqual(report.device_samples, (("10.0.0.1", 1), ("10.0.0.2", 1)))

    def test_required_columns_and_compatible_feature_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.csv").write_text("label,x\nbenign,1\n", encoding="utf-8")
            (root / "two.csv").write_text("label,y\nbenign,2\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaInspectionError, "feature schema mismatch"):
                inspect_csv_dataset("botiot", root, required_columns=("label",))
            with self.assertRaisesRegex(SchemaInspectionError, "missing required columns"):
                inspect_csv_dataset("botiot", root, required_columns=("missing",))

    def test_duplicate_headers_empty_files_and_non_csv_sources_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "duplicate.csv").write_text("A,a\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaInspectionError, "duplicate column"):
                inspect_csv_dataset("nbaiot", root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.csv").write_bytes(b"")
            with self.assertRaisesRegex(SchemaInspectionError, "empty CSV"):
                inspect_csv_dataset("nbaiot", root)

    def test_unparseable_numeric_row_fails_by_default_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "device" ).mkdir()
            (root / "device" / "benign.csv").write_text(
                "x,y\n1,2\nnot-a-number,4\n5,inf\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SchemaInspectionError, "2 rejected rows.*non_numeric_feature"
            ):
                inspect_csv_dataset("nbaiot", root, chunk_size=1)

            report = inspect_csv_dataset(
                "nbaiot", root, chunk_size=1, fail_on_rejected=False
            )
            self.assertEqual(report.total_rows, 3)
            self.assertEqual(report.accepted_rows, 1)
            self.assertEqual(report.rejected_rows, 2)
            self.assertEqual(
                report.rejected_reasons,
                (("non_finite_feature", 1), ("non_numeric_feature", 1)),
            )
            self.assertEqual(len(report.rejected_samples), 2)

    def test_all_missing_numeric_feature_is_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x,y\n1,\n2,NA\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaInspectionError, "no finite values.*y"):
                inspect_csv_dataset("nbaiot", root)

    def test_bad_label_device_and_timestamp_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "flows.csv").write_text(
                "category,subcategory,saddr,stime,x\n"
                ",tcp,device,1,1\n"
                "ddos,tcp,,1,1\n"
                "ddos,tcp,device,not-a-time,1\n",
                encoding="utf-8",
            )
            report = inspect_csv_dataset(
                "botiot",
                root,
                required_columns=("category", "subcategory", "saddr", "stime"),
                fail_on_rejected=False,
            )
            self.assertEqual(report.accepted_rows, 0)
            self.assertEqual(report.rejected_reasons, (
                ("invalid_device", 1),
                ("invalid_label", 1),
                ("invalid_timestamp", 1),
            ))

    def test_row_and_field_ceilings_bound_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x\n" + "z" * 200 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaInspectionError, "record exceeds"):
                inspect_csv_dataset("nbaiot", root, max_record_chars=32)

    def test_multiline_record_total_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text(
                'x\n"1111111111\n2222222222\n3333333333"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(SchemaInspectionError, "record exceeds"):
                inspect_csv_dataset("nbaiot", root, max_record_chars=24)

    def test_process_csv_field_limit_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x\n1\n", encoding="utf-8")
            original = csv.field_size_limit()
            inspect_csv_dataset("nbaiot", root, max_record_chars=original + 1)
            self.assertEqual(csv.field_size_limit(), original)

    def test_source_replacement_race_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            path.write_text("x\n1\n", encoding="utf-8")
            real_lstat = Path.lstat
            calls = 0

            def changed(candidate: Path):
                nonlocal calls
                result = real_lstat(candidate)
                if candidate == path:
                    calls += 1
                    if calls >= 2:
                        values = list(result)
                        values[1] += 1
                        return os.stat_result(values)
                return result

            with (
                patch("pathlib.Path.lstat", changed),
                self.assertRaisesRegex(SchemaInspectionError, "changed during inspection"),
            ):
                inspect_csv_dataset("nbaiot", root)

    def test_symlinked_source_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual.txt"
            actual.write_text("x\n1\n", encoding="utf-8")
            link = root / "rows.csv"
            try:
                os.symlink(actual, link)
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")
            with self.assertRaisesRegex(SchemaInspectionError, "regular non-link file"):
                inspect_csv_dataset("nbaiot", root)


if __name__ == "__main__":
    unittest.main()
