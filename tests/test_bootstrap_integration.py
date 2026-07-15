from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from bitguard_bnn.bootstrap.cleanup import scan_cleanup_debt
from bitguard_bnn.bootstrap.download import DownloadResult, download_file
from bitguard_bnn.bootstrap.extract import ExtractionResult, extract_zip
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
        "report_error",
        "lock_release_error",
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
                "acquisition_journals",
                "extraction_journals",
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
        second_download = Mock(
            side_effect=AssertionError("HTTP download must be reused")
        )
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
        schema = json.loads(
            Path(rerun["schema_reports"]["botiot"]).read_text(encoding="utf-8")
        )
        self.assertEqual(schema["accepted_rows"], 2)

    def test_acquisition_journal_reuses_first_dataset_after_later_failure(self) -> None:
        download_calls: list[Path] = []

        def local_download(url, destination, **_kwargs):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = self.nbaiot_archive.read_bytes()
            destination.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            download_calls.append(destination)
            return DownloadResult(
                destination=str(destination),
                byte_size=len(payload),
                sha256=digest,
                resumed=False,
                restarted=False,
                reused=False,
                source_url=url,
                final_response_url=url,
            )

        dependencies = BootstrapDependencies(
            nbaiot_download_size=self.nbaiot_archive.stat().st_size,
            available_bytes=10**12,
            compute_resolver=self.dependencies().compute_resolver,
            downloader=local_download,
        )
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        real_copy_tree = orchestrator_module._copy_tree

        def fail_botiot_copy(source, destination, expected_digest):
            if Path(source) == self.botiot_source.resolve():
                raise RuntimeError("injected BoT-IoT acquisition failure")
            return real_copy_tree(source, destination, expected_digest)

        with patch.object(
            orchestrator_module, "_copy_tree", side_effect=fail_botiot_copy
        ):
            first = run_bootstrap(self.options, dependencies=dependencies)

        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["failed_stage"], "acquire")
        journal = Path(first["reports"]["acquisition_journals"]["nbaiot"])
        self.assertTrue(journal.is_file())
        self.assertEqual(
            json.loads(journal.read_text(encoding="utf-8"))["status"], "completed"
        )
        self.assertEqual(len(download_calls), 1)

        def reject_network(*_args, **_kwargs):
            raise AssertionError("journaled N-BaIoT acquisition must not redownload")

        recovered = run_bootstrap(
            self.options,
            dependencies=BootstrapDependencies(
                nbaiot_download_size=self.nbaiot_archive.stat().st_size,
                available_bytes=10**12,
                compute_resolver=self.dependencies().compute_resolver,
                downloader=reject_network,
            ),
        )

        self.assertEqual(recovered["status"], "sources_verified")
        self.assertEqual(len(download_calls), 1)

    def test_remote_journal_is_bound_to_the_official_source_identity(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        calls: list[str] = []
        revision_two = self.root / "nbaiot-revision-two.zip"
        with zipfile.ZipFile(revision_two, "w") as archive:
            archive.writestr(
                "device_a/benign_traffic.csv",
                "mean,std\n9,10\n11,12\n",
            )
            archive.writestr(
                "device_b/gafgyt_attacks/tcp.csv",
                "mean,std\n13,14\n15,16\n",
            )

        def local_download(url, destination, **_kwargs):
            destination = Path(destination)
            payload = (
                revision_two.read_bytes()
                if url.endswith("revision-two.zip")
                else self.nbaiot_archive.read_bytes()
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            calls.append(url)
            return DownloadResult(
                destination=str(destination),
                byte_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                resumed=False,
                restarted=False,
                reused=False,
                source_url=url,
                final_response_url=url,
            )

        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "datasets": ("nbaiot",),
                "botiot_source": None,
                "accepted_botiot_license": False,
            }
        )
        dependencies = BootstrapDependencies(
            nbaiot_download_size=self.nbaiot_archive.stat().st_size,
            available_bytes=10**12,
            compute_resolver=self.dependencies().compute_resolver,
            downloader=local_download,
        )
        first = run_bootstrap(options, dependencies=dependencies)
        self.assertEqual(first["status"], "sources_verified")
        first_url = calls[0]

        registry = dict(orchestrator_module.load_registry())
        registry["nbaiot"] = replace(
            registry["nbaiot"],
            download_url="https://archive.ics.uci.edu/revision-two.zip",
        )
        with (
            patch.object(orchestrator_module, "load_registry", return_value=registry),
            patch(
                "bitguard_bnn.bootstrap.manifest.load_registry", return_value=registry
            ),
        ):
            second = run_bootstrap(options, dependencies=dependencies)

        self.assertEqual(second["status"], "sources_verified", msg=second["error"])
        self.assertEqual(
            calls,
            [
                first_url,
                "https://archive.ics.uci.edu/revision-two.zip",
            ],
        )
        self.assertNotEqual(first_url, calls[1])

    def test_journal_candidate_cannot_escape_its_private_namespace(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        def local_download(url, destination, **_kwargs):
            destination = Path(destination)
            payload = self.nbaiot_archive.read_bytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return DownloadResult(
                destination=str(destination),
                byte_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                resumed=False,
                restarted=False,
                reused=False,
                source_url=url,
                final_response_url=url,
            )

        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "datasets": ("nbaiot",),
                "botiot_source": None,
                "accepted_botiot_license": False,
            }
        )
        dependencies = BootstrapDependencies(
            nbaiot_download_size=self.nbaiot_archive.stat().st_size,
            available_bytes=10**12,
            compute_resolver=self.dependencies().compute_resolver,
            downloader=local_download,
        )
        first = run_bootstrap(options, dependencies=dependencies)
        self.assertEqual(first["status"], "sources_verified")
        journal = Path(first["reports"]["acquisition_journals"]["nbaiot"])
        record = json.loads(journal.read_text(encoding="utf-8"))
        final_path = Path(record["final_path"])
        external = self.root / "external-matching-archive.zip"
        external.write_bytes(final_path.read_bytes())
        final_path.unlink()
        record.update(status="intent", candidate_path=str(external))
        journal.write_text(json.dumps(record), encoding="utf-8")

        restarted = BootstrapOptions(**{**options.__dict__, "restart_stage": "acquire"})
        with patch.object(orchestrator_module, "_publish_candidate") as publisher:
            failed = run_bootstrap(restarted, dependencies=dependencies)

        self.assertEqual(failed["status"], "failed")
        self.assertIn("private staging namespace", failed["error"])
        self.assertTrue(external.is_file())
        publisher.assert_not_called()

    def test_journal_and_publication_directory_entries_are_fsynced(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        journal = self.root / "journal.json"
        with patch.object(orchestrator_module, "_fsync_parent_directory") as durable:
            orchestrator_module._write_json(journal, {"status": "intent"})
        self.assertEqual(
            durable.call_args_list,
            [call(journal.parent), call(journal)],
        )

        candidate = self.root / ".bitguard-acquire-test-candidate"
        destination = self.root / "published.bin"
        candidate.write_bytes(b"verified")
        with patch.object(orchestrator_module, "_fsync_parent_directory") as durable:
            orchestrator_module._publish_candidate(candidate, destination, "zip")
        durable.assert_any_call(destination)
        self.assertEqual(destination.read_bytes(), b"verified")

    def test_durable_directory_helper_orders_syncs_for_new_and_existing_roots(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        parent = self.root / "protocol-parent"
        parent.mkdir()
        root = parent / "protocol-root"
        events: list[tuple[str, Path]] = []
        real_mkdir = Path.mkdir

        def tracked_mkdir(path, *args, **kwargs):
            if Path(path) == root:
                events.append(("mkdir", root))
            return real_mkdir(path, *args, **kwargs)

        def fallback(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        ensure = getattr(orchestrator_module, "_ensure_durable_directory", fallback)
        with (
            patch.object(Path, "mkdir", tracked_mkdir),
            patch.object(
                orchestrator_module,
                "_fsync_directory",
                side_effect=lambda path: events.append(("fsync-self", Path(path))),
                create=True,
            ),
            patch.object(
                orchestrator_module,
                "_fsync_parent_directory",
                side_effect=lambda path: events.append(("fsync-parent", Path(path))),
            ),
        ):
            ensure(root)

        self.assertEqual(
            events,
            [
                ("mkdir", root),
                ("fsync-self", root),
                ("fsync-parent", root),
            ],
        )
        events.clear()
        with (
            patch.object(Path, "mkdir", tracked_mkdir),
            patch.object(
                orchestrator_module,
                "_fsync_directory",
                side_effect=lambda path: events.append(("fsync-self", Path(path))),
                create=True,
            ),
            patch.object(
                orchestrator_module,
                "_fsync_parent_directory",
                side_effect=lambda path: events.append(("fsync-parent", Path(path))),
            ),
        ):
            ensure(root)
        self.assertEqual(
            events,
            [
                ("fsync-self", root),
                ("fsync-parent", root),
            ],
        )

    def test_durable_directory_retry_reestablishes_sync_after_first_failure(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        root = self.root / "retry-protocol-root"
        with patch.object(
            orchestrator_module,
            "_fsync_parent_directory",
            side_effect=OSError("injected parent fsync failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected parent fsync failure"):
                orchestrator_module._ensure_durable_directory(root)

        self.assertTrue(root.is_dir())
        events: list[tuple[str, Path]] = []
        with (
            patch.object(
                orchestrator_module,
                "_fsync_directory",
                side_effect=lambda path: events.append(("fsync-self", Path(path))),
            ),
            patch.object(
                orchestrator_module,
                "_fsync_parent_directory",
                side_effect=lambda path: events.append(("fsync-parent", Path(path))),
            ),
        ):
            created = orchestrator_module._ensure_durable_directory(root)

        self.assertFalse(created)
        self.assertEqual(
            events,
            [
                ("fsync-self", root),
                ("fsync-parent", root),
            ],
        )

    def test_durable_directory_rejects_symlink_and_reparse_roots(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        root = self.root / "unsafe-protocol-root"
        cases = {
            "symlink": SimpleNamespace(
                st_mode=stat.S_IFLNK | 0o777,
                st_reparse_tag=0,
            ),
            "reparse": SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o755,
                st_reparse_tag=0xA0000003,
            ),
        }
        for name, result in cases.items():
            with (
                self.subTest(name=name),
                patch.object(
                    Path,
                    "lstat",
                    return_value=result,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "non-symlink directory"):
                    orchestrator_module._ensure_durable_directory(root)

    def test_tree_durability_flushes_files_and_directories_bottom_up(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        root = self.root / "durability-tree"
        child = root / "child"
        child.mkdir(parents=True)
        child_file = child / "child.bin"
        root_file = root / "root.bin"
        child_file.write_bytes(b"child")
        root_file.write_bytes(b"root")
        events: list[tuple[str, Path]] = []

        with (
            patch.object(
                orchestrator_module,
                "_fsync_regular_file",
                side_effect=lambda path: events.append(("file", Path(path))),
            ),
            patch.object(
                orchestrator_module,
                "_fsync_directory",
                side_effect=lambda path: events.append(("directory", Path(path))),
            ),
            patch.object(
                orchestrator_module,
                "_fsync_parent_directory",
                side_effect=lambda path: events.append(("parent", Path(path))),
            ),
        ):
            orchestrator_module._make_tree_durable(root)

        self.assertLess(
            events.index(("file", child_file)),
            events.index(("directory", child)),
        )
        self.assertLess(
            events.index(("directory", child)),
            events.index(("directory", root)),
        )
        self.assertLess(
            events.index(("file", root_file)),
            events.index(("directory", root)),
        )
        self.assertEqual(events[-1], ("parent", root))

    def test_regular_file_durability_preserves_read_only_mode(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        target = self.root / "read-only.bin"
        target.write_bytes(b"durable")
        target.chmod(stat.S_IREAD)
        expected_mode = stat.S_IMODE(target.lstat().st_mode)
        try:
            orchestrator_module._fsync_regular_file(target)
            self.assertEqual(stat.S_IMODE(target.lstat().st_mode), expected_mode)
        finally:
            target.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_read_only_manual_directory_is_copied_without_mutating_source(self) -> None:
        source_file = self.botiot_source / "flows.csv"
        source_file.chmod(stat.S_IREAD)
        source_mode = stat.S_IMODE(source_file.lstat().st_mode)
        options = replace(self.options, datasets=("botiot",))
        try:
            report = run_bootstrap(options, dependencies=self.dependencies())

            self.assertEqual(report["status"], "sources_verified")
            self.assertEqual(stat.S_IMODE(source_file.lstat().st_mode), source_mode)
            copied = next((self.data_root / "raw").rglob("flows.csv"))
            self.assertEqual(stat.S_IMODE(copied.lstat().st_mode), source_mode)
        finally:
            for candidate in self.root.rglob("*"):
                if candidate.is_file():
                    candidate.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_read_only_zip_result_is_durable_and_keeps_its_mode(self) -> None:
        def readonly_zip(source, destination, **kwargs):
            result = extract_zip(source, destination, **kwargs)
            extracted = next(Path(destination).rglob("*.csv"))
            extracted.chmod(stat.S_IREAD)
            return result

        options = replace(
            self.options,
            datasets=("nbaiot",),
            botiot_source=None,
            accepted_botiot_license=False,
        )
        try:
            report = run_bootstrap(
                options,
                dependencies=self.dependencies(zip_extractor=readonly_zip),
            )

            self.assertEqual(report["status"], "sources_verified")
            extracted = next((self.data_root / "raw").rglob("*.csv"))
            self.assertFalse(stat.S_IMODE(extracted.lstat().st_mode) & stat.S_IWRITE)
        finally:
            for candidate in self.root.rglob("*"):
                if candidate.is_file():
                    candidate.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_read_only_rar_result_is_durable_and_keeps_its_mode(self) -> None:
        rar_source = self.root / "official-botiot.rar"
        rar_source.write_bytes(b"fixture handled by injected extractor")

        def readonly_rar(source, destination, **_kwargs):
            destination = Path(destination)
            destination.mkdir()
            extracted = destination / "flows.csv"
            extracted.write_text(
                "category,subcategory,saddr,stime,bytes,rate\n"
                "Normal,Normal,10.0.0.1,1.5,100,2.0\n",
                encoding="utf-8",
            )
            extracted.chmod(stat.S_IREAD)
            return ExtractionResult(
                source=str(source),
                destination=str(destination),
                extractor="injected-rar-fixture",
                files=("flows.csv",),
                total_bytes=extracted.stat().st_size,
            )

        options = replace(
            self.options,
            datasets=("botiot",),
            botiot_source=rar_source,
        )
        dependencies = replace(
            self.dependencies(),
            rar_extractor=readonly_rar,
        )
        try:
            report = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(report["status"], "sources_verified")
            extracted = next((self.data_root / "raw").rglob("flows.csv"))
            self.assertFalse(stat.S_IMODE(extracted.lstat().st_mode) & stat.S_IWRITE)
        finally:
            for candidate in self.root.rglob("*"):
                if candidate.is_file():
                    candidate.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_protocol_roots_use_durable_creation_in_stage_order(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        observed: list[Path] = []
        real_ensure = getattr(orchestrator_module, "_ensure_durable_directory", None)

        def track(path):
            path = Path(path)
            observed.append(path)
            if real_ensure is None:
                path.mkdir(parents=True, exist_ok=True)
            else:
                real_ensure(path)

        with patch.object(
            orchestrator_module,
            "_ensure_durable_directory",
            side_effect=track,
            create=True,
        ):
            report = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(report["status"], "sources_verified")
        metadata = self.data_root / ".bitguard"
        required = (
            self.data_root,
            metadata,
            self.runs_root,
            metadata / "acquired",
            metadata / "acquisition-journal",
            self.data_root / "raw",
            metadata / "extraction-journal",
            metadata / "manifests",
            metadata / "schema",
        )
        for path in required:
            self.assertIn(path, observed)
        self.assertLess(
            observed.index(metadata / "acquired"),
            observed.index(metadata / "acquisition-journal"),
        )
        self.assertLess(
            observed.index(self.data_root / "raw"),
            observed.index(metadata / "extraction-journal"),
        )

    def test_journal_and_aggregate_reports_redact_injected_downloader_urls(
        self,
    ) -> None:
        def local_download(_url, destination, **_kwargs):
            destination = Path(destination)
            payload = self.nbaiot_archive.read_bytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return DownloadResult(
                destination=str(destination),
                byte_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                resumed=False,
                restarted=False,
                reused=False,
                source_url=(
                    "https://user:password@example.invalid/a path?token=source#fragment"
                ),
                final_response_url=(
                    r"https:\other:secret@example.invalid\b path?token=final"
                ),
            )

        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "datasets": ("nbaiot",),
                "botiot_source": None,
                "accepted_botiot_license": False,
            }
        )
        report = run_bootstrap(
            options,
            dependencies=BootstrapDependencies(
                nbaiot_download_size=self.nbaiot_archive.stat().st_size,
                available_bytes=10**12,
                compute_resolver=self.dependencies().compute_resolver,
                downloader=local_download,
            ),
        )
        self.assertEqual(report["status"], "sources_verified")
        serialized = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in (
                report["reports"]["acquisition"],
                report["reports"]["acquisition_journals"]["nbaiot"],
            )
        )
        for secret in ("password", "secret", "token=source", "token=final", "fragment"):
            self.assertNotIn(secret, serialized)
        self.assertIn("<redacted-url>", serialized)

    def test_extraction_journal_reuses_first_tree_after_later_pcap_failure(
        self,
    ) -> None:
        botiot_archive = self.root / "botiot-repairable.zip"

        def write_botiot(*, include_pcap: bool) -> None:
            with zipfile.ZipFile(botiot_archive, "w") as archive:
                archive.writestr(
                    "flows.csv",
                    "category,subcategory,saddr,stime,bytes,rate\n"
                    "Normal,Normal,10.0.0.1,1.5,100,2.0\n",
                )
                if include_pcap:
                    archive.writestr("capture.pcap", b"excluded")

        write_botiot(include_pcap=True)
        options = BootstrapOptions(
            **{**self.options.__dict__, "botiot_source": botiot_archive.resolve()}
        )
        first = run_bootstrap(options, dependencies=self.dependencies())

        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["failed_stage"], "extract")
        journal = Path(first["reports"]["extraction_journals"]["nbaiot"])
        journal_payload = json.loads(journal.read_text(encoding="utf-8"))
        self.assertEqual(journal_payload["status"], "completed")
        nbaiot_tree = Path(journal_payload["final_path"])
        original_tree_digest = journal_payload["tree_sha256"]

        write_botiot(include_pcap=False)

        def selective_extract(source, destination, **kwargs):
            if Path(source).name.startswith("nbaiot-"):
                raise AssertionError("journaled N-BaIoT tree must not be re-extracted")
            return extract_zip(source, destination, **kwargs)

        restarted = BootstrapOptions(**{**options.__dict__, "restart_stage": "extract"})
        recovered = run_bootstrap(
            restarted,
            dependencies=self.dependencies(zip_extractor=selective_extract),
        )

        self.assertEqual(recovered["status"], "sources_verified")
        self.assertTrue(nbaiot_tree.is_dir())
        self.assertEqual(
            json.loads(journal.read_text(encoding="utf-8"))["tree_sha256"],
            original_tree_digest,
        )

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
        self.assertTrue(
            (self.data_root / ".bitguard" / "bootstrap-report.json").is_file()
        )

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

    def test_nested_rar_failure_retains_private_candidate_without_final_tree(
        self,
    ) -> None:
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
        self.assertIn("<redacted-url>", encoded)
        report_path = self.data_root / ".bitguard" / "bootstrap-report.json"
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), failed)
        self.assertEqual(
            list(report_path.parent.glob("bootstrap-report.json.*.tmp")), []
        )

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
            **{
                **self.options.__dict__,
                "runs_root": (self.root / "other-runs").resolve(),
            }
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

    def test_lock_path_resolution_failure_uses_comprehensive_fallback_report(
        self,
    ) -> None:
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

    def test_cleanup_scan_failure_reuses_platform_inspection_command(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        data_root = (self.root / "data with ' quote").resolve()
        options = BootstrapOptions(
            **{
                **self.options.__dict__,
                "data_root": data_root,
                "runs_root": (self.root / "runs with ' quote").resolve(),
            }
        )

        commands: dict[str, str] = {}
        for system in ("Linux", "Windows"):
            with (
                self.subTest(system=system),
                patch(
                    "bitguard_bnn.bootstrap.cleanup.platform.system",
                    return_value=system,
                ),
                patch.object(
                    orchestrator_module,
                    "scan_cleanup_debt",
                    side_effect=OSError("injected cleanup scan failure"),
                ),
                patch.object(
                    orchestrator_module,
                    "_lock_path",
                    side_effect=RuntimeError("injected early failure"),
                ),
            ):
                report = run_bootstrap(options, dependencies=self.dependencies())
            commands[system] = report["cleanup_debt"]["recovery_command"]

        self.assertIn("ls -ld --", commands["Linux"])
        self.assertIn(shlex.quote(str(data_root)), commands["Linux"])
        self.assertNotRegex(commands["Linux"], r"(^|\s)rm(\s|$)")
        self.assertIn("Get-Item -Force -LiteralPath", commands["Windows"])
        self.assertIn(str(data_root).replace("'", "''"), commands["Windows"])
        self.assertNotIn("Remove-Item", commands["Windows"])

    def test_recursive_redaction_sanitizes_urlish_keys_and_space_tails(self) -> None:
        first_key = (
            r"prefix https:\user:password@example.invalid\path with spaces"
            "?token=first#fragment"
        )
        second_key = (
            "prefix https://other:secret@example.invalid/another path "
            "with spaces?token=second"
        )
        error_url = (
            "failure https://error:password@example.invalid/a path "
            "with spaces?token=error#fragment"
        )

        with patch(
            "bitguard_bnn.bootstrap.orchestrator._lock_path",
            side_effect=RuntimeError(error_url),
        ):
            failed = run_bootstrap(
                self.options,
                raw_inputs={
                    "nested": {
                        first_key: ["first"],
                        second_key: ["second"],
                    }
                },
                dependencies=self.dependencies(),
            )

        self.assert_comprehensive_report(failed)
        encoded = json.dumps(failed)
        for secret in (
            "password",
            "secret",
            "token=first",
            "token=second",
            "token=error",
            "#fragment",
        ):
            self.assertNotIn(secret, encoded)
        nested = failed["inputs"]["original"]["nested"]
        self.assertEqual(len(nested), 2)
        self.assertEqual(
            sorted(value[0] for value in nested.values()), ["first", "second"]
        )
        self.assertEqual(
            set(nested),
            {"prefix <redacted-url>", "prefix <redacted-url>#collision-2"},
        )

    def test_redaction_fails_closed_for_every_uri_scheme_and_mapping_key(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        payload = {
            "ftp://user:ftp-secret@example.invalid/a?token=ftp#fragment": "ftp",
            "s3://access:s3-secret@bucket/key?X-Amz-Signature=signed": "s3",
            "file://user:file-secret@host/path?api_key=file-key": "file",
            r"custom:\user:custom-secret@example.invalid\path?secret=custom": "custom",
            "opaque:user:opaque-secret@example.invalid/path?secret=opaque": "opaque",
        }

        redacted = orchestrator_module._redact_report_value(payload)

        encoded = json.dumps(redacted)
        for secret in (
            "ftp-secret",
            "s3-secret",
            "file-secret",
            "custom-secret",
            "opaque-secret",
            "token=ftp",
            "X-Amz-Signature",
            "api_key",
            "secret=custom",
            "secret=opaque",
            "#fragment",
        ):
            self.assertNotIn(secret, encoded)
        self.assertEqual(
            set(redacted.values()),
            {"ftp", "s3", "file", "custom", "opaque"},
        )

    def test_copy_tree_fsync_failure_prevents_journal_and_publication(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        options = replace(self.options, datasets=("botiot",))
        original_fsync_directory = orchestrator_module._fsync_directory

        def fail_copy_staging(path):
            if path.name.startswith(".bitguard-extract-copy-"):
                raise OSError("injected copied-tree fsync failure")
            return original_fsync_directory(path)

        with patch.object(
            orchestrator_module,
            "_fsync_directory",
            side_effect=fail_copy_staging,
        ):
            failed = run_bootstrap(options, dependencies=self.dependencies())

        self.assertEqual(failed["failed_stage"], "acquire")
        metadata = self.data_root / ".bitguard"
        self.assertFalse((metadata / "acquisition-journal" / "botiot.json").exists())
        acquired = metadata / "acquired"
        self.assertFalse(any(acquired.glob("botiot-*")))

    def test_extracted_tree_fsync_failure_prevents_journal_and_publication(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        options = replace(
            self.options,
            datasets=("nbaiot",),
            botiot_source=None,
            accepted_botiot_license=False,
        )
        original_fsync_directory = orchestrator_module._fsync_directory

        def fail_extraction_staging(path):
            if path.name.startswith(".bitguard-extract-nbaiot-"):
                raise OSError("injected extracted-tree fsync failure")
            return original_fsync_directory(path)

        with patch.object(
            orchestrator_module,
            "_fsync_directory",
            side_effect=fail_extraction_staging,
        ):
            failed = run_bootstrap(options, dependencies=self.dependencies())

        self.assertEqual(failed["failed_stage"], "extract")
        metadata = self.data_root / ".bitguard"
        self.assertFalse((metadata / "extraction-journal" / "nbaiot.json").exists())
        self.assertFalse(any((self.data_root / "raw").glob("nbaiot-*")))

    def test_every_protocol_directory_is_reverified_on_fresh_and_retry_runs(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        original_ensure = orchestrator_module._ensure_durable_directory
        with patch.object(
            orchestrator_module,
            "_ensure_durable_directory",
            wraps=original_ensure,
        ) as ensure:
            first = run_bootstrap(self.options, dependencies=self.dependencies())
            retry = run_bootstrap(
                replace(self.options, restart_stage="inspect"),
                dependencies=self.dependencies(),
            )

        self.assertEqual(first["status"], "sources_verified")
        self.assertEqual(retry["status"], "sources_verified")
        verified = [call.args[0] for call in ensure.call_args_list]
        for required in (
            self.data_root,
            self.data_root / ".bitguard",
            self.data_root / ".bitguard" / "manifests",
            self.data_root / ".bitguard" / "schema",
            self.runs_root,
        ):
            self.assertGreaterEqual(verified.count(required), 2, required)

    def test_hardware_change_invalidates_environment_and_downstream_stages(
        self,
    ) -> None:
        first = run_bootstrap(self.options, dependencies=self.dependencies())
        changed = self.dependencies()
        changed = replace(
            changed,
            compute_resolver=lambda requested: {
                "requested": requested,
                "selected_profile": "cpu",
                "device": "cpu",
                "device_name": "different-cpu-profile",
            },
        )

        rerun = run_bootstrap(self.options, dependencies=changed)

        self.assertEqual(first["status"], "sources_verified")
        self.assertEqual(rerun["reused_stages"], ["preflight"])
        self.assertEqual(
            rerun["executed_stages"],
            ["environment", "acquire", "extract", "inspect"],
        )

    def test_moving_data_root_invalidates_environment_and_downstream_stages(
        self,
    ) -> None:
        first = run_bootstrap(self.options, dependencies=self.dependencies())
        moved_root = (self.root / "moved-data").resolve()
        shutil.copytree(self.data_root, moved_root)
        moved = replace(self.options, data_root=moved_root)

        rerun = run_bootstrap(moved, dependencies=self.dependencies())

        self.assertEqual(first["status"], "sources_verified")
        self.assertEqual(rerun["reused_stages"], ["preflight"])
        self.assertEqual(rerun["executed_stages"], ["environment"])
        self.assertEqual(rerun["failed_stage"], "acquire")
        for downstream in ("acquire", "extract", "inspect"):
            self.assertNotIn(downstream, rerun["reused_stages"])

    def test_environment_manifest_omits_hostname_user_and_unknown_secrets(self) -> None:
        dependencies = replace(
            self.dependencies(),
            compute_resolver=lambda requested: {
                "requested": requested,
                "selected_profile": "cpu",
                "device": "cpu",
                "hostname": "private-host-secret",
                "user": "private-user-secret",
                "custom_token": "private-token-secret",
            },
        )

        report = run_bootstrap(self.options, dependencies=dependencies)

        environment = Path(report["reports"]["environment"]).read_text(encoding="utf-8")
        payload = json.loads(environment)
        self.assertEqual(
            set(payload["runtime"]),
            {"os", "python", "torch", "cuda"},
        )
        self.assertGreaterEqual(
            set(payload["runtime"]["os"]),
            {"system", "platform", "release", "version", "machine", "processor"},
        )
        self.assertEqual(
            set(payload["runtime"]["python"]),
            {"implementation", "version"},
        )
        self.assertIn("version", payload["runtime"]["torch"])
        self.assertGreaterEqual(
            set(payload["runtime"]["cuda"]),
            {"runtime", "device", "device_name", "device_index", "profile"},
        )
        self.assertEqual(len(payload["data_root_identity"]), 64)
        self.assertNotIn(str(self.data_root), environment)
        for secret in (
            "private-host-secret",
            "private-user-secret",
            "private-token-secret",
        ):
            self.assertNotIn(secret, environment)

    def test_post_replace_report_fsync_failure_never_leaves_canonical_success(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        canonical = self.data_root / ".bitguard" / "bootstrap-report.json"
        original_fsync_parent = orchestrator_module._fsync_parent_directory

        def fail_canonical_report_fsync(path):
            if Path(path) == canonical:
                raise OSError("injected post-replace report fsync failure")
            return original_fsync_parent(path)

        with patch.object(
            orchestrator_module,
            "_fsync_parent_directory",
            side_effect=fail_canonical_report_fsync,
        ):
            failed = run_bootstrap(self.options, dependencies=self.dependencies())

        fallback = Path(failed["report_path"])
        canonical_payload = json.loads(canonical.read_text(encoding="utf-8"))
        fallback_payload = json.loads(fallback.read_text(encoding="utf-8"))
        self.assertEqual(failed, fallback_payload)
        self.assertNotEqual(fallback, canonical)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "report")
        self.assertEqual(canonical_payload["status"], "failed")
        self.assertEqual(canonical_payload["failed_stage"], "report")
        self.assertEqual(canonical_payload["error"], fallback_payload["error"])
        self.assertEqual(
            canonical_payload["report_error"],
            fallback_payload["report_error"],
        )
        self.assertIn("durability", failed["report_error"].casefold())
        state = json.loads(
            (self.data_root / ".bitguard" / "bootstrap-state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("inspect", state["stages"])

        restarted = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(restarted["status"], "sources_verified")
        self.assertEqual(
            restarted["reused_stages"],
            ["preflight", "environment", "acquire", "extract", "inspect"],
        )

    def test_stage_failure_remains_primary_when_report_persistence_also_fails(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        original_write_json = orchestrator_module._write_json

        def fail_canonical_report(path, value):
            if path.name == "bootstrap-report.json":
                raise OSError("injected canonical report failure")
            return original_write_json(path, value)

        with (
            patch.object(
                orchestrator_module,
                "_write_json",
                side_effect=fail_canonical_report,
            ),
            patch.object(
                orchestrator_module,
                "_extract_archive",
                side_effect=RuntimeError("injected extraction stage failure"),
            ),
        ):
            failed = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "extract")
        self.assertIn("injected extraction stage failure", failed["error"])
        self.assertIn("injected canonical report failure", failed["report_error"])
        self.assertIsNone(failed["lock_release_error"])
        self.assertNotEqual(Path(failed["report_path"]).name, "bootstrap-report.json")
        self.assert_comprehensive_report(failed)

    def test_stage_failure_remains_primary_when_lock_release_also_fails(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator as orchestrator_module

        original_release = BootstrapWriterLock.release

        def fail_after_release(lock):
            original_release(lock)
            raise RuntimeError("injected lock release failure")

        with (
            patch.object(BootstrapWriterLock, "release", fail_after_release),
            patch.object(
                orchestrator_module,
                "_extract_archive",
                side_effect=RuntimeError("injected extraction stage failure"),
            ),
        ):
            failed = run_bootstrap(self.options, dependencies=self.dependencies())

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_stage"], "extract")
        self.assertIn("injected extraction stage failure", failed["error"])
        self.assertIn("injected lock release failure", failed["lock_release_error"])
        self.assertIsNone(failed["report_error"])
        self.assert_comprehensive_report(failed)

    def test_successful_stages_with_lock_release_failure_never_publish_success(
        self,
    ) -> None:
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
        self.assertIn("release", failed["lock_release_error"])
        self.assertIsNone(failed["report_error"])
        self.assert_comprehensive_report(failed)
        canonical = self.data_root / ".bitguard" / "bootstrap-report.json"
        self.assertEqual(Path(failed["report_path"]), canonical)
        self.assertEqual(
            json.loads(canonical.read_text(encoding="utf-8"))["status"],
            "failed",
        )
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

    def test_cleanup_command_is_platform_specific_and_quotes_literal_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retired = root / ".bitguard-retired-space ' quote"
            retired.mkdir()
            (retired / "artifact").write_bytes(b"x")

            with patch(
                "bitguard_bnn.bootstrap.cleanup.platform.system",
                return_value="Linux",
            ):
                posix = scan_cleanup_debt((root,))
            self.assertIn("ls -ld --", posix["recovery_command"])
            self.assertIn(shlex.quote(str(retired)), posix["recovery_command"])
            self.assertNotRegex(posix["recovery_command"], r"(^|\s)rm(\s|$)")

            with patch(
                "bitguard_bnn.bootstrap.cleanup.platform.system",
                return_value="Windows",
            ):
                windows = scan_cleanup_debt((root,))
            self.assertIn("Get-Item -Force -LiteralPath", windows["recovery_command"])
            self.assertIn(str(retired).replace("'", "''"), windows["recovery_command"])
            self.assertNotIn("Remove-Item", windows["recovery_command"])

    def test_scan_never_traverses_symlinked_debt_and_records_scan_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.bin").write_bytes(b"do-not-count")
            linked = root / ".bitguard-retired-linked"
            try:
                linked.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")

            report = scan_cleanup_debt((root,))

            self.assertEqual(report["apparent_bytes"], 0)
            self.assertEqual(report["unique_bytes"], 0)
            self.assertTrue(report["scan_errors"])
            self.assertIn(str(linked), json.dumps(report["scan_errors"]))

    def test_scan_reports_interrupted_partial_and_private_json_temporaries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "archive.zip.partial"
            partial.write_bytes(b"partial")
            report_temp = root / (
                "bootstrap-report.json.0123456789abcdef0123456789abcdef.tmp"
            )
            report_temp.write_bytes(b"report")
            journal_temp = root / "nbaiot.json.abcdef0123456789abcdef0123456789.tmp"
            journal_temp.write_bytes(b"journal")
            (root / "ordinary.tmp").write_bytes(b"ignore")

            report = scan_cleanup_debt((root,))

            paths = {
                Path(item["path"]).name: item["kind"] for item in report["artifacts"]
            }
            self.assertEqual(paths[partial.name], "partial")
            self.assertEqual(paths[report_temp.name], "atomic_json_temporary")
            self.assertEqual(paths[journal_temp.name], "atomic_json_temporary")
            self.assertNotIn("ordinary.tmp", paths)
            for candidate in (partial, report_temp, journal_temp):
                self.assertTrue(candidate.exists())


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
