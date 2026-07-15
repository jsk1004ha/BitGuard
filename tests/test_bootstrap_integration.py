from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bitguard_bnn.bootstrap.cleanup import scan_cleanup_debt
from bitguard_bnn.bootstrap.download import download_file
from bitguard_bnn.bootstrap.extract import extract_zip
from bitguard_bnn.bootstrap.inspect import inspect_csv_dataset
from bitguard_bnn.bootstrap.orchestrator import (
    BootstrapDependencies,
    Stage,
    run_bootstrap,
)
from bitguard_bnn.bootstrap.types import BootstrapOptions


class BootstrapAcquisitionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.runs_root = self.root / "runs"
        self.nbaiot_archive = self.root / "nbaiot.zip"
        with zipfile.ZipFile(self.nbaiot_archive, "w") as archive:
            archive.writestr(
                "device_a/benign_traffic.csv",
                "mean,std\n1,2\n3,4\n",
            )
            archive.writestr(
                "device_b/gafgyt_attacks/tcp.csv",
                "mean,std\n5,6\n7,8\n",
            )
        self.botiot_source = self.root / "official-botiot"
        self.botiot_source.mkdir()
        (self.botiot_source / "flows.csv").write_text(
            "category,subcategory,saddr,stime,bytes,rate\n"
            "Normal,Normal,10.0.0.1,1.5,100,2.0\n"
            "DDoS,TCP,10.0.0.2,2.5,200,3.0\n",
            encoding="utf-8",
        )
        self.options = BootstrapOptions(
            datasets=("nbaiot", "botiot"),
            botiot_source=self.botiot_source.resolve(),
            data_root=self.data_root.resolve(),
            runs_root=self.runs_root.resolve(),
            compute="cpu",
            prepare_only=True,
            install_system_tools=False,
            accepted_botiot_license=True,
            restart_stage=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def dependencies(
        self,
        *,
        downloader=download_file,
        zip_extractor=extract_zip,
        inspector=inspect_csv_dataset,
    ) -> BootstrapDependencies:
        return BootstrapDependencies(
            nbaiot_archive=self.nbaiot_archive,
            available_bytes=10**12,
            compute_resolver=lambda requested: {
                "requested": requested,
                "selected_profile": "cpu",
                "device": "cpu",
            },
            downloader=downloader,
            zip_extractor=zip_extractor,
            inspector=inspector,
        )

    def test_stage_record_has_name_signature_and_run(self) -> None:
        stage = Stage("inspect", lambda: "signature", lambda: ())
        self.assertEqual(stage.name, "inspect")
        self.assertEqual(stage.input_signature(), "signature")
        self.assertEqual(stage.run(), ())

    def test_prepare_only_runs_to_verified_sources_and_reuses_every_stage(self) -> None:
        first = run_bootstrap(
            self.options,
            raw_inputs={
                "botiot_source": "official-botiot",
                "data_root": "data",
                "runs_root": "runs",
            },
            dependencies=self.dependencies(),
        )

        self.assertEqual(first["status"], "sources_verified")
        self.assertEqual(first["last_completed_stage"], "inspect")
        self.assertIsNone(first["failed_stage"])
        state = json.loads(
            (self.data_root / ".bitguard" / "bootstrap-state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(state["stages"]),
            {"preflight", "environment", "acquire", "extract", "inspect"},
        )
        for dataset in ("nbaiot", "botiot"):
            self.assertTrue(Path(first["manifests"][dataset]).is_file())
            self.assertTrue(Path(first["schema_reports"][dataset]).is_file())

        second_zip = Mock(side_effect=AssertionError("ZIP extraction must be reused"))
        second_inspector = Mock(side_effect=AssertionError("inspection must be reused"))
        second_download = Mock(side_effect=AssertionError("HTTP download must be reused"))
        second = run_bootstrap(
            self.options,
            dependencies=self.dependencies(
                downloader=second_download,
                zip_extractor=second_zip,
                inspector=second_inspector,
            ),
        )

        self.assertEqual(second["status"], "sources_verified")
        self.assertEqual(
            second["reused_stages"],
            ["preflight", "environment", "acquire", "extract", "inspect"],
        )
        second_zip.assert_not_called()
        second_download.assert_not_called()
        second_inspector.assert_not_called()

    def test_source_mutation_invalidates_acquire_extract_and_inspect(self) -> None:
        initial = run_bootstrap(self.options, dependencies=self.dependencies())
        self.assertEqual(initial["status"], "sources_verified")

        (self.botiot_source / "flows.csv").write_text(
            "category,subcategory,saddr,stime,bytes,rate\n"
            "Normal,Normal,10.0.0.1,1.5,101,2.0\n"
            "DDoS,TCP,10.0.0.2,2.5,200,3.0\n",
            encoding="utf-8",
        )
        rerun = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(rerun["status"], "sources_verified")
        self.assertIn("acquire", rerun["executed_stages"])
        self.assertIn("extract", rerun["executed_stages"])
        self.assertIn("inspect", rerun["executed_stages"])
        schema = json.loads(Path(rerun["schema_reports"]["botiot"]).read_text(encoding="utf-8"))
        self.assertEqual(schema["accepted_rows"], 2)

    def test_nbaiot_archive_revision_invalidates_downstream_stages(self) -> None:
        initial = run_bootstrap(self.options, dependencies=self.dependencies())
        self.assertEqual(initial["status"], "sources_verified")
        with zipfile.ZipFile(self.nbaiot_archive, "w") as archive:
            archive.writestr(
                "device_a/benign_traffic.csv",
                "mean,std\n9,2\n3,4\n",
            )
            archive.writestr(
                "device_b/gafgyt_attacks/tcp.csv",
                "mean,std\n5,6\n7,8\n",
            )

        rerun = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(rerun["status"], "sources_verified")
        self.assertEqual(rerun["reused_stages"], ["preflight", "environment"])
        self.assertEqual(rerun["executed_stages"], ["acquire", "extract", "inspect"])

    def test_modified_extracted_tree_is_not_silently_reaccepted(self) -> None:
        initial = run_bootstrap(self.options, dependencies=self.dependencies())
        self.assertEqual(initial["status"], "sources_verified")
        raw_csv = next((self.data_root / "raw").glob("nbaiot-*/*/benign_traffic.csv"))
        raw_csv.write_text("mean,std\n99,2\n3,4\n", encoding="utf-8")

        failed = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "extract")
        self.assertIn("does not match its prior verified tree", failed["error"])

    def test_low_disk_fails_before_acquisition_and_still_writes_report(self) -> None:
        failed = run_bootstrap(
            self.options,
            dependencies=BootstrapDependencies(
                nbaiot_archive=self.nbaiot_archive,
                available_bytes=0,
                compute_resolver=lambda _requested: {
                    "selected_profile": "cpu",
                    "device": "cpu",
                },
            ),
        )

        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["last_completed_stage"])
        self.assertEqual(failed["failed_stage"], "preflight")
        self.assertIn("Insufficient disk space", failed["error"])
        self.assertFalse((self.data_root / ".bitguard" / "acquired").exists())
        self.assertTrue((self.data_root / ".bitguard" / "bootstrap-report.json").is_file())

    def test_botiot_pcap_directory_fails_closed_before_copy(self) -> None:
        (self.botiot_source / "capture.pcap").write_bytes(b"not-used")

        failed = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "preflight")
        self.assertIn("PCAP capture input is excluded", failed["error"])
        self.assertFalse((self.data_root / ".bitguard" / "acquired").exists())

    def test_failure_report_is_atomic_redacted_and_restart_is_actionable(self) -> None:
        def fail_zip(*_args, **_kwargs):
            raise RuntimeError(
                "remote https://user:secret@example.invalid/archive.zip?token=abc#fragment failed"
            )

        failed = run_bootstrap(
            self.options,
            raw_inputs={
                "data_root": "https://user:password@example.invalid/data?token=raw#part"
            },
            dependencies=self.dependencies(zip_extractor=fail_zip),
        )

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["last_completed_stage"], "acquire")
        self.assertEqual(failed["failed_stage"], "extract")
        self.assertIn("--restart-stage extract", failed["recovery_command"])
        encoded = json.dumps(failed)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("password", encoded)
        self.assertNotIn("token=abc", encoded)
        self.assertNotIn("token=raw", encoded)
        self.assertIn("https://example.invalid/data", encoded)
        report_path = self.data_root / ".bitguard" / "bootstrap-report.json"
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), failed)
        self.assertEqual(list(report_path.parent.glob("bootstrap-report.json.*.tmp")), [])

        restarted_options = BootstrapOptions(
            **{**self.options.__dict__, "restart_stage": "extract"}
        )
        recovered = run_bootstrap(
            restarted_options,
            dependencies=self.dependencies(),
        )
        self.assertEqual(recovered["status"], "sources_verified")
        self.assertIn("extract", recovered["executed_stages"])


class CleanupDebtTest(unittest.TestCase):
    def test_scan_reports_apparent_and_inode_deduplicated_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retired = root / ".bitguard-retired-one"
            retired.mkdir()
            artifact = retired / "artifact"
            artifact.write_bytes(b"12345")
            hardlink = retired / "alias"
            try:
                os.link(artifact, hardlink)
            except OSError as error:
                self.skipTest(f"hard links unavailable: {error}")
            staging = root / ".bitguard-extract-leftover"
            staging.mkdir()
            (staging / "partial.csv").write_bytes(b"123")
            zero = root / ".bitguard-retired-zero"
            zero.mkdir()
            (zero / "artifact").write_bytes(b"")

            report = scan_cleanup_debt((root,))

            self.assertEqual(report["apparent_bytes"], 13)
            self.assertEqual(report["unique_bytes"], 8)
            self.assertEqual(len(report["artifacts"]), 3)
            self.assertIn("Do not delete automatically", report["recovery_command"])


class GitIgnoreContractTest(unittest.TestCase):
    def test_generated_runtime_paths_are_ignored_but_plans_are_not(self) -> None:
        lines = {
            line.strip()
            for line in (Path(__file__).resolve().parents[1] / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for expected in (".venv/", "data/", "runs/", "__pycache__/", ".pytest_cache/"):
            self.assertIn(expected, lines)
        self.assertNotIn(".omx/", lines)
        self.assertNotIn("docs/superpowers/plans/", lines)


if __name__ == "__main__":
    unittest.main()
