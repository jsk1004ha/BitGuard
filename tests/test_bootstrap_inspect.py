from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import bitguard_bnn.bootstrap.inspect as inspect_module
from bitguard_bnn.data import _coerce_timestamp, _numeric_features

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
            self.assertEqual(
                [item.relative_path for item in report.files],
                [
                    "device_a/gafgyt_attacks/tcp.csv",
                    "device_b/benign_traffic.csv",
                ],
            )
            self.assertEqual(report.rejected_rows, 0)
            self.assertEqual(report.as_dict()["total_rows"], 4)

    def test_botiot_required_metadata_labels_devices_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "flows.csv"
            path.write_text(
                "category,subcategory,saddr,stime,bytes,rate\n"
                "Normal,Normal,10.0.0.1,1.5,100,2.0\n"
                "DDoS,TCP,10.0.0.2,2.5,200,3.0\n",
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

    def test_botiot_classifies_realistic_columns_without_rejecting_strings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "flows.csv"
            path.write_text(
                "category,subcategory,saddr,stime,pkSeqID,seq,proto,daddr,sport,dport,"
                "smac,dmac,bytes,rate,mixed,description\n"
                "Normal,Normal,10.0.0.1,1.5,1,2,tcp,10.0.0.2,1234,80,aa,bb,100,2.0,7,ok\n"
                "DDoS,TCP,10.0.0.3,2.5,3,4,udp,10.0.0.4,53,9999,cc,dd,200,3.0,unknown,bad\n",
                encoding="utf-8",
            )

            report = inspect_csv_dataset("botiot", root, chunk_size=1)

            self.assertEqual(report.accepted_rows, 2)
            self.assertEqual(report.rejected_rows, 0)
            self.assertEqual(report.feature_columns, ("bytes", "mixed", "rate"))
            self.assertEqual(report.unusable_columns, ("description", "proto"))
            self.assertEqual(
                report.excluded_columns,
                (
                    "category",
                    "daddr",
                    "dmac",
                    "dport",
                    "pkSeqID",
                    "saddr",
                    "seq",
                    "smac",
                    "sport",
                    "stime",
                    "subcategory",
                ),
            )
            self.assertEqual(report.files[0].unusable_columns, ("description", "proto"))
            self.assertEqual(
                report.as_dict()["unusable_columns"], ["description", "proto"]
            )

    def test_drop_columns_override_is_casefolded_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "flows.csv").write_text(
                "category,subcategory,saddr,stime,proto,bytes,café,rate\n"
                "Normal,Normal,10.0.0.1,1,tcp,100,3,2.0\n",
                encoding="utf-8",
            )

            report = inspect_csv_dataset(
                "botiot",
                root,
                drop_columns=(
                    " PrOtO ",
                    "BYTES",
                    "cafe\N{COMBINING ACUTE ACCENT}",
                ),
            )

            self.assertEqual(report.feature_columns, ("rate",))
            self.assertEqual(report.unusable_columns, ())
            self.assertIn("bytes", report.excluded_columns)
            self.assertIn("café", report.excluded_columns)
            self.assertIn("proto", report.excluded_columns)

            for invalid in ("proto", b"proto", True, ("proto", "PROTO"), ("",)):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "drop_columns"):
                        inspect_csv_dataset("botiot", root, drop_columns=invalid)  # type: ignore[arg-type]

    def test_feature_schemas_are_compared_after_streaming_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = "category,subcategory,saddr,stime,proto,x\n"
            (root / "one.csv").write_text(
                metadata + "Normal,Normal,device-a,1,tcp,1\n", encoding="utf-8"
            )
            (root / "two.csv").write_text(
                metadata + "Normal,Normal,device-b,2,17,2\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                SchemaInspectionError, "feature schema mismatch"
            ):
                inspect_csv_dataset("botiot", root)

    def test_required_columns_and_compatible_feature_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.csv").write_text("label,x\nbenign,1\n", encoding="utf-8")
            (root / "two.csv").write_text("label,y\nbenign,2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SchemaInspectionError, "feature schema mismatch"
            ):
                inspect_csv_dataset("botiot", root, required_columns=("label",))
            with self.assertRaisesRegex(
                SchemaInspectionError, "missing required columns"
            ):
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

    def test_numeric_candidates_match_coercing_adapter_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "device").mkdir()
            (root / "device" / "benign.csv").write_text(
                "x,y\n1,2\nnot-a-number,4\n5,inf\n", encoding="utf-8"
            )
            report = inspect_csv_dataset("nbaiot", root, chunk_size=1)
            self.assertEqual(report.total_rows, 3)
            self.assertEqual(report.accepted_rows, 3)
            self.assertEqual(report.rejected_rows, 0)
            self.assertEqual(report.feature_columns, ("x", "y"))
            self.assertEqual(report.unusable_columns, ())

    def test_numeric_eligibility_exactly_matches_training_pandas_conversion(
        self,
    ) -> None:
        tokens = (
            "0",
            " +1 ",
            "-2",
            ".5",
            "5.",
            "6e-2",
            "-7E+2",
            "+inf",
            "-INF",
            "Infinity",
            " inf ",
            "nan",
            "+nan",
            "-NaN",
            "1_000",
            "1,000",
            "0x10",
            "0e309",
            "0.0e309",
            ".0e309",
            "00.1e309",
            "0.01e310",
            "1e309",
            "1e-400",
            "99e307",
        )
        columns = tuple(f"value_{index:02d}" for index in range(len(tokens)))
        frame = pd.DataFrame(
            [{column: token for column, token in zip(columns, tokens)}]
        )
        expected_frame = frame.copy()
        expected = tuple(sorted(_numeric_features(expected_frame), key=str.casefold))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow(tokens)

            report = inspect_csv_dataset("nbaiot", root, chunk_size=1)

        self.assertEqual(report.feature_columns, expected)
        observed = {
            token: column in report.feature_columns
            for column, token in zip(columns, tokens)
        }
        self.assertFalse(observed["0e309"])
        self.assertTrue(observed["0.0e309"])
        self.assertTrue(observed[".0e309"])
        self.assertTrue(observed["00.1e309"])
        self.assertTrue(observed["0.01e310"])
        self.assertFalse(observed["1_000"])

    def test_underscore_timestamp_is_rejected_during_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "flows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("category", "subcategory", "saddr", "stime", "x"))
                for index in range(99):
                    writer.writerow(("ddos", "tcp", "device", str(index), "1"))
                writer.writerow(("ddos", "tcp", "device", "1_000", "1"))

            report = inspect_csv_dataset(
                "botiot",
                root,
                required_columns=("category", "subcategory", "saddr", "stime"),
                fail_on_rejected=False,
            )

            self.assertEqual(report.accepted_rows, 99)
            self.assertEqual(report.rejected_reasons, (("invalid_timestamp", 1),))

    def test_timestamp_mode_matches_training_threshold_at_95_percent(self) -> None:
        for numeric_rows, expected_accepted in ((96, 96), (94, 6)):
            with (
                self.subTest(numeric_rows=numeric_rows),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                timestamps = [str(index + 1) for index in range(numeric_rows)]
                timestamps.extend(
                    "2019-01-01T00:00:00Z" for _ in range(100 - numeric_rows)
                )
                expected = _coerce_timestamp(pd.Series(timestamps)).notna()
                with (root / "flows.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("category", "subcategory", "saddr", "stime", "x"))
                    for timestamp in timestamps:
                        writer.writerow(("Normal", "Normal", "device", timestamp, "1"))

                report = inspect_csv_dataset(
                    "botiot", root, chunk_size=7, fail_on_rejected=False
                )

                self.assertEqual(report.accepted_rows, int(expected.sum()))
                self.assertEqual(report.accepted_rows, expected_accepted)
                self.assertEqual(
                    report.rejected_reasons,
                    (("invalid_timestamp", 100 - expected_accepted),),
                )

    def test_timestamp_numeric_mode_rejects_infinity_for_elapsed_time_safety(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = ["1"] * 95 + ["inf"] * 5
            with (root / "flows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("category", "subcategory", "saddr", "stime", "x"))
                for timestamp in rows:
                    writer.writerow(("Normal", "Normal", "device", timestamp, "1"))

            report = inspect_csv_dataset(
                "botiot", root, chunk_size=9, fail_on_rejected=False
            )

            self.assertEqual(report.accepted_rows, 95)
            self.assertEqual(report.rejected_reasons, (("invalid_timestamp", 5),))

    def test_pandas_conversions_never_receive_more_than_one_bounded_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "flows.csv").write_text(
                "category,subcategory,saddr,stime,x,y\n"
                + "".join(
                    f"Normal,Normal,device,2019-01-01T00:00:{index:02d}Z,value,{index}\n"
                    for index in range(23)
                ),
                encoding="utf-8",
            )
            numeric_sizes: list[int] = []
            datetime_sizes: list[int] = []
            real_numeric = pd.to_numeric
            real_datetime = pd.to_datetime

            def bounded_numeric(values, *args, **kwargs):
                numeric_sizes.append(len(values))
                return real_numeric(values, *args, **kwargs)

            def bounded_datetime(values, *args, **kwargs):
                datetime_sizes.append(len(values))
                return real_datetime(values, *args, **kwargs)

            with (
                patch.object(inspect_module.pd, "to_numeric", bounded_numeric),
                patch.object(inspect_module.pd, "to_datetime", bounded_datetime),
            ):
                inspect_csv_dataset("botiot", root, chunk_size=5)

            self.assertTrue(numeric_sizes)
            self.assertLessEqual(max(numeric_sizes), 5)
            self.assertTrue(datetime_sizes)
            self.assertLessEqual(max(datetime_sizes), 5)

    def test_missing_pandas_dependency_reports_actionable_bootstrap_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x\n1\n", encoding="utf-8")
            with (
                patch.object(inspect_module, "pd", None),
                self.assertRaisesRegex(
                    SchemaInspectionError,
                    "pandas.*bootstrap.*locked dependencies",
                ),
            ):
                inspect_csv_dataset("nbaiot", root)

    def test_pandas_divergent_numeric_forms_are_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text(
                "anchor,underscore,grouped,hex,overflow,signed_nan\n"
                '1,1_000,"1,000",0x10,1e309,+nan\n'
                '2,-1_000,"-1,000",-0x10,-1e309,-NaN\n',
                encoding="utf-8",
            )

            report = inspect_csv_dataset("nbaiot", root)

            self.assertEqual(report.feature_columns, ("anchor",))
            self.assertEqual(
                report.unusable_columns,
                ("grouped", "hex", "overflow", "signed_nan", "underscore"),
            )

    def test_only_pandas_divergent_numeric_forms_fail_as_no_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text(
                "underscore,overflow\n1_000,1e309\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                SchemaInspectionError, "no numeric feature columns"
            ):
                inspect_csv_dataset("nbaiot", root)

    def test_all_missing_candidate_is_reported_as_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x,y\n1,\n2,NA\n", encoding="utf-8")
            report = inspect_csv_dataset("nbaiot", root)
            self.assertEqual(report.feature_columns, ("x",))
            self.assertEqual(report.unusable_columns, ("y",))

    def test_no_numeric_feature_columns_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.csv").write_text("x,y\nleft,right\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SchemaInspectionError, "no numeric feature columns"
            ):
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
            self.assertEqual(
                report.rejected_reasons,
                (
                    ("invalid_device", 1),
                    ("invalid_label", 1),
                    ("invalid_timestamp", 1),
                ),
            )

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
                self.assertRaisesRegex(
                    SchemaInspectionError, "changed during inspection"
                ),
            ):
                inspect_csv_dataset("nbaiot", root)

    def test_source_replacement_between_bounded_passes_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.csv"
            path.write_text("x\n1\n", encoding="utf-8")
            real_open = inspect_module._open_pinned_text
            calls = 0

            def replace_before_second_pass(candidate: Path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    replacement = root / "replacement.csv"
                    replacement.write_text("x\n2\n", encoding="utf-8")
                    os.replace(replacement, path)
                return real_open(candidate)

            with (
                patch.object(
                    inspect_module,
                    "_open_pinned_text",
                    replace_before_second_pass,
                ),
                self.assertRaisesRegex(
                    SchemaInspectionError, "changed between inspection passes"
                ),
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
