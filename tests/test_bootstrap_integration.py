from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bitguard_bnn.bootstrap.cleanup import scan_cleanup_debt
from bitguard_bnn.bootstrap.download import download_file
from bitguard_bnn.bootstrap.extract import extract_zip
from bitguard_bnn.bootstrap.inspect import inspect_csv_dataset
from bitguard_bnn.bootstrap.orchestrator import (
    BootstrapDependencies,
    Stage,
    _lock_path,
    run_bootstrap,
)
from bitguard_bnn.bootstrap.state import BootstrapWriterLock
from bitguard_bnn.bootstrap.types import BootstrapOptions


class BootstrapAcquisitionIntegrationTest(unittest.TestCase):
    REQUIRED_REPORT_KEYS = {
        "version",
        "status",
        "last_completed_stage",
        "failed_stage",
        "error",
        "recovery_command",
        "inputs",
        "compute",
        "state",
        "manifests",
        "schema_reports",
        "cleanup_debt",
        "executed_stages",
        "reused_stages",
        "threat_model",
        "report_path",
        "reports",
    }

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

    def assert_comprehensive_report(self, report: dict[str, object]) -> None:
        self.assertEqual(set(report), self.REQUIRED_REPORT_KEYS)
        self.assertIsInstance(report["manifests"], dict)
        self.assertIsInstance(report["schema_reports"], dict)
        self.assertIsInstance(report["cleanup_debt"], dict)
        self.assertIsInstance(report["executed_stages"], list)
        self.assertIsInstance(report["reused_stages"], list)
        self.assertEqual(
            set(report["reports"]),
            {
                "bootstrap",
                "preflight",
                "environment",
                "acquisition",
                "extraction",
                "schemas",
            },
        )
        self.assertEqual(report["reports"]["bootstrap"], report["report_path"])
        persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(persisted, report)

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
        self.assert_comprehensive_report(first)
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
        for name in ("preflight", "environment", "acquisition", "extraction"):
            self.assertTrue(Path(first["reports"][name]).is_file())
        self.assertEqual(first["reports"]["schemas"], first["schema_reports"])

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
        self.assert_comprehensive_report(failed)
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

    def test_botiot_zip_pcap_listing_fails_before_final_tree_publication(self) -> None:
        archive_path = self.root / "botiot-with-capture.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("flows.csv", "category,subcategory\nNormal,Normal\n")
            archive.writestr("nested/capture.pcap", b"excluded")
        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "datasets": ("botiot",),
                "botiot_source": archive_path.resolve(),
            }
        )

        failed = run_bootstrap(options, dependencies=self.dependencies())

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "extract")
        self.assertIn("PCAP capture input is excluded", failed["error"])
        raw_root = self.data_root / "raw"
        self.assertEqual(list(raw_root.glob("botiot-*")), [])

    def test_nested_rar_failure_retains_private_candidate_without_final_tree(self) -> None:
        archive_path = self.root / "botiot-with-nested-rar.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("flows.csv", "category,subcategory\nNormal,Normal\n")
            archive.writestr("nested.rar", b"rar")
        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "datasets": ("botiot",),
                "botiot_source": archive_path.resolve(),
            }
        )

        def reject_nested(*_args, **_kwargs):
            raise RuntimeError("PCAP capture input is excluded from nested RAR listing")

        dependencies = BootstrapDependencies(
            nbaiot_archive=self.nbaiot_archive,
            available_bytes=10**12,
            compute_resolver=self.dependencies().compute_resolver,
            rar_extractor=reject_nested,
        )
        failed = run_bootstrap(options, dependencies=dependencies)

        self.assertEqual(failed["status"], "failed")
        self.assertIn("PCAP capture input is excluded", failed["error"])
        raw_root = self.data_root / "raw"
        self.assertEqual(list(raw_root.glob("botiot-*")), [])
        self.assertTrue(list(raw_root.glob(".bitguard-extract-*")))

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

    def test_writer_lock_identity_depends_only_on_data_root(self) -> None:
        alternate = BootstrapOptions(
            **{**self.options.__dict__, "runs_root": (self.root / "other-runs").resolve()}
        )

        self.assertEqual(_lock_path(self.options), _lock_path(alternate))
        with BootstrapWriterLock(_lock_path(self.options)):
            failed = run_bootstrap(
                alternate,
                raw_inputs={
                    "data_root": "https://user:secret@example.invalid/data?token=lock",
                },
                dependencies=self.dependencies(),
            )

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "preflight")
        self.assert_comprehensive_report(failed)
        self.assertIsNone(failed["reports"]["preflight"])
        self.assertEqual(failed["reports"]["schemas"], {})
        encoded = json.dumps(failed)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("token=lock", encoded)

    def test_lock_path_resolution_failure_uses_comprehensive_fallback_report(self) -> None:
        with patch(
            "bitguard_bnn.bootstrap.orchestrator._lock_path",
            side_effect=RuntimeError(
                "bad https://user:secret@example.invalid/root?token=path#fragment"
            ),
        ):
            failed = run_bootstrap(
                self.options,
                raw_inputs={
                    "runs_root": "https://user:secret@example.invalid/runs?token=input"
                },
                dependencies=self.dependencies(),
            )

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "preflight")
        self.assert_comprehensive_report(failed)
        encoded = json.dumps(failed)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("token=path", encoded)
        self.assertNotIn("token=input", encoded)

    def test_recursive_report_redaction_covers_windows_mangled_url_paths(self) -> None:
        mangled = (
            self.root
            / r"https:\user:password@example.invalid\data?token=resolved#fragment"
        )
        unsafe_options = BootstrapOptions(
            **{**self.options.__dict__, "data_root": mangled}
        )
        debt = {
            "artifacts": [{"path": str(mangled), "nested": [str(mangled)]}],
            "apparent_bytes": 0,
            "unique_bytes": 0,
            "recovery_command": f"inspect {mangled}",
        }

        with patch(
            "bitguard_bnn.bootstrap.orchestrator.scan_cleanup_debt",
            return_value=debt,
        ):
            failed = run_bootstrap(
                unsafe_options,
                raw_inputs={"nested": {"paths": [str(mangled)]}},
                dependencies=self.dependencies(),
            )

        self.assertEqual(failed["status"], "failed")
        self.assert_comprehensive_report(failed)
        encoded = json.dumps(failed)
        for secret in ("password", "token=resolved", "#fragment"):
            self.assertNotIn(secret, encoded)
        self.assertNotIn("user:", encoded)

    def test_lock_release_failure_uses_same_comprehensive_report_schema(self) -> None:
        original_release = BootstrapWriterLock.release

        def fail_after_release(lock):
            original_release(lock)
            raise RuntimeError(
                "release https://user:secret@example.invalid/lock?token=release failed"
            )

        with patch.object(BootstrapWriterLock, "release", fail_after_release):
            failed = run_bootstrap(
                self.options,
                raw_inputs={
                    "data_root": "https://user:secret@example.invalid/data?token=input"
                },
                dependencies=self.dependencies(),
            )

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["last_completed_stage"], "inspect")
        self.assertEqual(failed["failed_stage"], "lock-release")
        self.assert_comprehensive_report(failed)
        encoded = json.dumps(failed)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("token=release", encoded)
        self.assertNotIn("token=input", encoded)


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
            escaped = str(retired).replace("'", "''")
            self.assertIn(escaped, report["recovery_command"])
            self.assertIn("Get-Item", report["recovery_command"])
            self.assertIn("-LiteralPath", report["recovery_command"])
            self.assertNotRegex(
                report["recovery_command"], r"(?i)Remove-Item|del(?:ete)?-Item"
            )


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
        self.assertIn(".omx/metrics.json", lines)
        self.assertNotIn(".omx/", lines)
        self.assertNotIn(".omx/project-memory.json", lines)
        self.assertNotIn(".omx/notepad.md", lines)
        self.assertNotIn("docs/superpowers/plans/", lines)

        root = Path(__file__).resolve().parents[1]
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", ".omx/metrics.json"],
            cwd=root,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)
        for retained in (
            ".omx/project-memory.json",
            ".omx/notepad.md",
            ".omx/plans/prd-bootstrap.md",
        ):
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "-q", retained],
                cwd=root,
                check=False,
            )
            self.assertEqual(result.returncode, 1, retained)


if __name__ == "__main__":
    unittest.main()
