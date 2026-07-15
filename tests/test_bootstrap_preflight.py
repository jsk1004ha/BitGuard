from __future__ import annotations

import ctypes
import json
import os
import stat
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bitguard_bnn.bootstrap.preflight import (
    NVIDIA_SMI_COMMAND,
    NVIDIA_SMI_TIMEOUT_SECONDS,
    CPUFacts,
    DiskFacts,
    DriverInfo,
    NvidiaProbeError,
    RAMFacts,
    ResourceProbeError,
    ResourceRequest,
    TorchVerification,
    build_post_install_report,
    choose_compute,
    collect_package_versions,
    discover_cpu,
    discover_ram,
    estimate_resources,
    inspect_local_archives,
    probe_nvidia_driver,
    probe_posix_ram,
    probe_windows_ram,
    require_disk,
    require_disk_at,
    verify_torch_compute,
)


class ResourceRequestTests(unittest.TestCase):
    def test_insufficient_disk_fails_before_mutation(self):
        request = ResourceRequest(download=10, extracted=20, shards=15, temporary=5, reserve=10)
        with self.assertRaisesRegex(RuntimeError, "required=60.*available=59"):
            require_disk(request, available_bytes=59)

    def test_detected_cuda_never_silently_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA profile verification failed"):
            choose_compute("auto", driver=DriverInfo(nvidia=True), torch_cuda=False)

    def test_resource_arithmetic_includes_partial_and_evaluation_exactly(self):
        request = ResourceRequest(
            download=10,
            extracted=20,
            shards=15,
            temporary=5,
            reserve=10,
            partial=7,
            evaluation=3,
        )

        self.assertEqual(request.required_bytes, 70)
        self.assertEqual(
            request.breakdown(),
            {
                "final_downloads": 10,
                "partial_downloads": 7,
                "extracted_data": 20,
                "parquet_shards": 15,
                "evaluation_artifacts": 3,
                "temporary_workspace": 5,
                "reserve": 10,
            },
        )
        self.assertEqual(request.as_dict()["required_bytes"], 70)
        json.dumps(request.as_dict())

    def test_resource_sizes_reject_bool_negative_and_non_integer_values(self):
        invalid_values = (True, -1, 1.5, "1")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    ResourceRequest(value, 0, 0, 0, 0)

        request = ResourceRequest(0, 0, 0, 0, 0)
        for value in invalid_values:
            with self.subTest(available=value):
                with self.assertRaises((TypeError, ValueError)):
                    require_disk(request, available_bytes=value)

    def test_resource_facts_are_immutable(self):
        request = ResourceRequest(1, 2, 3, 4, 5)
        with self.assertRaises(FrozenInstanceError):
            request.download = 2  # type: ignore[misc]

    def test_disk_boundary_equal_passes_and_one_byte_short_fails(self):
        request = ResourceRequest(10, 20, 15, 5, 10)

        facts = require_disk(request, available_bytes=60)

        self.assertEqual(facts.available_bytes, 60)
        self.assertEqual(facts.required_bytes, 60)
        with self.assertRaisesRegex(RuntimeError, "required=60.*available=59"):
            require_disk(request, available_bytes=59)

    def test_disk_failure_does_not_create_or_write_any_path(self):
        request = ResourceRequest(10, 20, 15, 5, 10)
        with (
            patch("pathlib.Path.mkdir") as mkdir,
            patch("builtins.open") as open_file,
            self.assertRaises(RuntimeError),
        ):
            require_disk(request, available_bytes=0)
        mkdir.assert_not_called()
        open_file.assert_not_called()

    def test_path_disk_check_uses_free_space_without_creating_path(self):
        request = ResourceRequest(1, 2, 3, 4, 5)
        usage = Mock(return_value=SimpleNamespace(free=15))
        target = Path("a-directory-that-does-not-need-to-exist")

        with patch("pathlib.Path.mkdir") as mkdir:
            facts = require_disk_at(request, target, disk_usage_fn=usage)

        mkdir.assert_not_called()
        usage.assert_called_once()
        self.assertEqual(facts.available_bytes, 15)
        self.assertEqual(facts.path, str(target.resolve()))

    def test_huge_integer_arithmetic_never_converts_through_float(self):
        huge = 2**100 + 123
        request = ResourceRequest(huge, huge, huge, huge, huge, partial=huge, evaluation=huge)

        facts = require_disk(request, available_bytes=huge * 7)

        self.assertEqual(request.required_bytes, huge * 7)
        self.assertEqual(facts.as_dict()["available_bytes"], huge * 7)
        json.dumps(facts.as_dict())


class ArchiveInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_archive_sizes_and_partial_siblings_are_observed(self):
        first = self.root / "first.zip"
        second = self.root / "second.tar.gz"
        first.write_bytes(b"12345")
        second.write_bytes(b"1234567")
        Path(f"{first}.partial").write_bytes(b"123")

        inspection = inspect_local_archives([first, second])

        self.assertEqual(inspection.total_archive_bytes, 12)
        self.assertEqual(inspection.total_partial_bytes, 3)
        self.assertEqual(inspection.archives[0].path, str(first.resolve()))
        self.assertEqual(inspection.archives[0].partial_bytes, 3)
        self.assertIsNone(inspection.archives[1].partial_path)
        json.dumps(inspection.as_dict())

    def test_missing_archive_is_rejected_actionably(self):
        missing = self.root / "missing.zip"
        with self.assertRaisesRegex(ResourceProbeError, "Archive does not exist.*missing.zip"):
            inspect_local_archives([missing])

    def test_directory_is_not_accepted_as_an_archive(self):
        directory = self.root / "archive.zip"
        directory.mkdir()
        with self.assertRaisesRegex(ResourceProbeError, "not a regular file"):
            inspect_local_archives([directory])

    def test_non_file_partial_sibling_is_rejected(self):
        archive = self.root / "archive.zip"
        archive.write_bytes(b"archive")
        Path(f"{archive}.partial").mkdir()
        with self.assertRaisesRegex(ResourceProbeError, "partial.*not a regular file"):
            inspect_local_archives([archive])

    def test_archive_symlink_is_rejected_before_partial_accounting(self):
        target = self.root / "target.zip"
        target.write_bytes(b"archive")
        link = self.root / "link.zip"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        Path(f"{link}.partial").write_bytes(b"partial")

        with self.assertRaisesRegex(ResourceProbeError, "symlink.*not allowed"):
            inspect_local_archives([link])

    def test_partial_appearing_during_absence_probe_fails_closed(self):
        archive = (self.root / "archive.zip").resolve()
        archive.write_bytes(b"archive")
        partial = Path(f"{archive}.partial").resolve()
        partial_calls = 0

        def missing_then_present(path: Path):
            nonlocal partial_calls
            if path == archive:
                return path.stat()
            if path == partial:
                partial_calls += 1
                if partial_calls == 1:
                    raise FileNotFoundError(path)
                return SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o600,
                    st_size=4,
                    st_mtime_ns=1,
                    st_dev=1,
                    st_ino=2,
                )
            raise AssertionError(path)

        with self.assertRaisesRegex(
            ResourceProbeError, "Archive partial.*changed during inspection"
        ):
            inspect_local_archives([archive], stat_fn=missing_then_present)

    def test_partial_disappearing_during_present_probe_fails_closed(self):
        archive = (self.root / "archive.zip").resolve()
        archive.write_bytes(b"archive")
        partial = Path(f"{archive}.partial").resolve()
        partial_calls = 0

        def present_then_missing(path: Path):
            nonlocal partial_calls
            if path == archive:
                return path.stat()
            if path == partial:
                partial_calls += 1
                if partial_calls == 2:
                    raise FileNotFoundError(path)
                return SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o600,
                    st_size=4,
                    st_mtime_ns=1,
                    st_dev=1,
                    st_ino=2,
                )
            raise AssertionError(path)

        with self.assertRaisesRegex(
            ResourceProbeError, "Archive partial.*changed during inspection"
        ):
            inspect_local_archives([archive], stat_fn=present_then_missing)

    def test_stable_partial_absence_requires_two_missing_observations(self):
        archive = (self.root / "archive.zip").resolve()
        archive.write_bytes(b"archive")
        partial = Path(f"{archive}.partial").resolve()
        partial_calls = 0

        def stable_absence(path: Path):
            nonlocal partial_calls
            if path == archive:
                return path.stat()
            if path == partial:
                partial_calls += 1
                raise FileNotFoundError(path)
            raise AssertionError(path)

        inspection = inspect_local_archives([archive], stat_fn=stable_absence)

        self.assertEqual(partial_calls, 2)
        self.assertEqual(inspection.total_partial_bytes, 0)
        self.assertIsNone(inspection.archives[0].partial_path)

    def test_changing_archive_observation_fails_closed(self):
        archive = (self.root / "changing.zip").resolve()
        archive.write_bytes(b"123")
        calls = 0

        def changing_stat(path: Path):
            nonlocal calls
            if path == archive:
                calls += 1
                return SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o600,
                    st_size=3 if calls == 1 else 4,
                    st_mtime_ns=1,
                    st_dev=1,
                    st_ino=2,
                )
            raise FileNotFoundError(path)

        with self.assertRaisesRegex(ResourceProbeError, "changed during inspection"):
            inspect_local_archives([archive], stat_fn=changing_stat)

    def test_duplicate_resolved_archives_are_rejected(self):
        archive = self.root / "archive.zip"
        archive.write_bytes(b"123")
        with self.assertRaisesRegex(ResourceProbeError, "Duplicate archive path"):
            inspect_local_archives([archive, archive.parent / "." / archive.name])

    def test_caller_supplied_estimates_are_explicit_and_reported(self):
        archive = self.root / "archive.zip"
        archive.write_bytes(b"1234567890")
        Path(f"{archive}.partial").write_bytes(b"123")
        inspection = inspect_local_archives([archive])

        estimate = estimate_resources(
            inspection,
            final_download_bytes=12,
            extracted_bytes=20,
            shards_bytes=15,
            evaluation_bytes=4,
            temporary_bytes=5,
            reserve_bytes=10,
        )

        self.assertEqual(estimate.request.required_bytes, 69)
        self.assertEqual(estimate.request.partial, 3)
        self.assertEqual(
            estimate.as_dict()["estimate_sources"],
            {
                "final_downloads": "caller_supplied",
                "extracted_data": "caller_supplied",
                "parquet_shards": "caller_supplied",
                "evaluation_artifacts": "caller_supplied",
                "temporary_workspace": "caller_supplied",
                "reserve": "caller_supplied",
                "partial_downloads": "observed_local_files",
            },
        )
        json.dumps(estimate.as_dict())

    def test_download_estimate_cannot_be_smaller_than_observed_archives(self):
        archive = self.root / "archive.zip"
        archive.write_bytes(b"1234567890")
        inspection = inspect_local_archives([archive])
        with self.assertRaisesRegex(ValueError, "final_download_bytes=9.*observed=10"):
            estimate_resources(
                inspection,
                final_download_bytes=9,
                extracted_bytes=0,
                shards_bytes=0,
                evaluation_bytes=0,
                temporary_bytes=0,
                reserve_bytes=0,
            )


class PlatformResourceTests(unittest.TestCase):
    def test_cpu_count_uses_probe_and_falls_back_safely(self):
        self.assertEqual(discover_cpu(lambda: 12).logical_count, 12)
        for invalid in (None, 0, -1, True, "8"):
            with self.subTest(invalid=invalid):
                facts = discover_cpu(lambda invalid=invalid: invalid)
                self.assertEqual(facts.logical_count, 1)
                self.assertTrue(facts.used_fallback)

    def test_posix_ram_uses_page_counts_with_exact_integer_arithmetic(self):
        values = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 1_000_000,
            "SC_AVPHYS_PAGES": 250_000,
        }

        facts = probe_posix_ram(sysconf_fn=values.__getitem__)

        self.assertEqual(facts.total_bytes, 4_096_000_000)
        self.assertEqual(facts.available_bytes, 1_024_000_000)
        self.assertEqual(facts.platform, "posix")

    def test_posix_ram_rejects_malformed_or_failed_probe(self):
        invalid = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 100, "SC_AVPHYS_PAGES": 0}
        with self.assertRaisesRegex(ResourceProbeError, "SC_AVPHYS_PAGES"):
            probe_posix_ram(sysconf_fn=invalid.__getitem__)

        def failed(_name: str):
            raise OSError("probe unavailable")

        with self.assertRaisesRegex(
            ResourceProbeError, "POSIX RAM probe failed.*probe unavailable"
        ):
            probe_posix_ram(sysconf_fn=failed)

    def test_windows_ram_configures_and_calls_global_memory_status_ex(self):
        class FakeGlobalMemoryStatusEx:
            argtypes = None
            restype = None

            def __call__(self, pointer):
                status = pointer._obj
                self.length_seen = status.dwLength
                status.ullTotalPhys = 16 * 1024**3
                status.ullAvailPhys = 6 * 1024**3
                return 1

        function = FakeGlobalMemoryStatusEx()

        facts = probe_windows_ram(global_memory_status_ex=function)

        self.assertEqual(facts.total_bytes, 16 * 1024**3)
        self.assertEqual(facts.available_bytes, 6 * 1024**3)
        self.assertEqual(facts.platform, "windows")
        self.assertEqual(function.restype, ctypes.c_int)
        self.assertEqual(len(function.argtypes), 1)
        self.assertGreater(function.length_seen, 0)

    def test_windows_ram_failure_is_actionable(self):
        function = Mock(return_value=0)
        with self.assertRaisesRegex(ResourceProbeError, "GlobalMemoryStatusEx failed.*error=5"):
            probe_windows_ram(global_memory_status_ex=function, get_last_error=lambda: 5)

    def test_platform_selection_is_injected_without_changing_os_name(self):
        values = {
            "SC_PAGE_SIZE": 1024,
            "SC_PHYS_PAGES": 10,
            "SC_AVPHYS_PAGES": 5,
        }
        facts = discover_ram(platform_name="posix", sysconf_fn=values.__getitem__)
        self.assertEqual(facts.total_bytes, 10_240)

        with self.assertRaisesRegex(ResourceProbeError, "Unsupported platform"):
            discover_ram(platform_name="plan9")

    def test_resource_fact_reports_are_immutable_and_json_safe(self):
        cpu = CPUFacts(logical_count=4, used_fallback=False)
        ram = RAMFacts(total_bytes=100, available_bytes=25, platform="posix")
        disk = DiskFacts(required_bytes=10, available_bytes=20, path="C:/data")
        for facts in (cpu, ram, disk):
            json.dumps(facts.as_dict())
        with self.assertRaises(FrozenInstanceError):
            ram.total_bytes = 101  # type: ignore[misc]


class NvidiaProbeAndSelectionTests(unittest.TestCase):
    def test_nvidia_probe_runs_the_exact_command_and_parses_integer_mib(self):
        result = SimpleNamespace(
            returncode=0,
            stdout="555.42, NVIDIA RTX 4090, 24576 MiB\n",
            stderr="",
        )
        run = Mock(return_value=result)

        driver = probe_nvidia_driver(run=run)

        run.assert_called_once_with(
            list(NVIDIA_SMI_COMMAND),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
        self.assertTrue(driver.nvidia)
        self.assertEqual(driver.driver_version, "555.42")
        self.assertEqual(driver.device_name, "NVIDIA RTX 4090")
        self.assertEqual(driver.memory_bytes, 24576 * 1024**2)
        self.assertIsNone(driver.device_index)
        json.dumps(driver.as_dict())

    def test_nvidia_probe_timeout_is_bounded_and_actionable(self):
        timeout = subprocess.TimeoutExpired(list(NVIDIA_SMI_COMMAND), timeout=2)
        with self.assertRaisesRegex(NvidiaProbeError, "timed out.*2"):
            probe_nvidia_driver(run=Mock(side_effect=timeout), timeout_seconds=2)

    def test_nvidia_probe_decode_failure_is_wrapped_actionably(self):
        decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with self.assertRaisesRegex(NvidiaProbeError, "UTF-8.*decode"):
            probe_nvidia_driver(run=Mock(side_effect=decode_error))

    def test_nvidia_probe_selects_device_index_one_explicitly(self):
        result = SimpleNamespace(
            returncode=0,
            stdout="555.42, NVIDIA RTX 4090, 24576 MiB\n",
            stderr="",
        )
        run = Mock(return_value=result)

        driver = probe_nvidia_driver(run=run, device_index=1)

        run.assert_called_once_with(
            [
                "nvidia-smi",
                "--id=1",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
        self.assertEqual(driver.device_index, 1)
        self.assertEqual(driver.as_dict()["device_index"], 1)

    def test_invalid_nvidia_device_selectors_fail_before_subprocess(self):
        for index in (True, -1, 1.5):
            with self.subTest(index=index):
                run = Mock()
                with self.assertRaises((TypeError, ValueError)):
                    probe_nvidia_driver(run=run, device_index=index)  # type: ignore[arg-type]
                run.assert_not_called()

    def test_selected_nvidia_probe_still_rejects_multiple_rows(self):
        output = "555.42, GPU A, 1024 MiB\n555.42, GPU B, 2048 MiB\n"
        result = SimpleNamespace(returncode=0, stdout=output, stderr="")
        with self.assertRaisesRegex(NvidiaProbeError, "multiple GPU rows"):
            probe_nvidia_driver(run=Mock(return_value=result), device_index=1)

    def test_missing_nvidia_smi_means_no_nvidia_driver(self):
        driver = probe_nvidia_driver(run=Mock(side_effect=FileNotFoundError()))
        self.assertEqual(driver, DriverInfo(nvidia=False))

    def test_nvidia_probe_operating_system_failure_is_not_hidden(self):
        with self.assertRaisesRegex(NvidiaProbeError, "Could not execute nvidia-smi.*denied"):
            probe_nvidia_driver(run=Mock(side_effect=OSError("denied")))

    def test_nonzero_nvidia_probe_is_not_reported_as_cpu(self):
        result = SimpleNamespace(returncode=9, stdout="", stderr="driver communication failed")
        with self.assertRaisesRegex(NvidiaProbeError, "exit code 9.*driver communication failed"):
            probe_nvidia_driver(run=Mock(return_value=result))

    def test_malformed_and_invalid_memory_rows_fail_closed(self):
        outputs = (
            "not,csv\n",
            "555.42, RTX, unknown\n",
            "555.42, , 1024 MiB\n",
        )
        for output in outputs:
            with self.subTest(output=output):
                result = SimpleNamespace(returncode=0, stdout=output, stderr="")
                with self.assertRaisesRegex(NvidiaProbeError, "malformed"):
                    probe_nvidia_driver(run=Mock(return_value=result))

    def test_multiple_gpu_rows_fail_closed_instead_of_choosing_implicitly(self):
        output = "555.42, GPU A, 1024 MiB\n555.42, GPU B, 2048 MiB\n"
        result = SimpleNamespace(returncode=0, stdout=output, stderr="")
        with self.assertRaisesRegex(NvidiaProbeError, "multiple GPU rows"):
            probe_nvidia_driver(run=Mock(return_value=result))

    def test_explicit_cpu_is_preserved_even_when_gpu_exists(self):
        driver = DriverInfo(nvidia=True, device_name="GPU", memory_bytes=1024)
        self.assertEqual(
            choose_compute(
                "cpu",
                driver=driver,
                torch_cuda=True,
                torch_cuda_version="12.1",
            ),
            "cpu",
        )

    def test_auto_without_nvidia_selects_cpu(self):
        self.assertEqual(
            choose_compute("auto", driver=DriverInfo(nvidia=False), torch_cuda=False), "cpu"
        )

    def test_explicit_cuda_never_downgrades_when_driver_or_torch_is_missing(self):
        with self.assertRaisesRegex(RuntimeError, "NVIDIA driver/GPU is required"):
            choose_compute("cu118", driver=DriverInfo(nvidia=False), torch_cuda=False)
        with self.assertRaisesRegex(RuntimeError, "CUDA profile verification failed"):
            choose_compute("cu124", driver=DriverInfo(nvidia=True), torch_cuda=False)

    def test_successful_compute_selection_preserves_profiles(self):
        driver = DriverInfo(nvidia=True)
        self.assertEqual(
            choose_compute(
                "cu118",
                driver=driver,
                torch_cuda=True,
                torch_cuda_version="11.8",
            ),
            "cu118",
        )
        self.assertEqual(
            choose_compute(
                "cu124",
                driver=driver,
                torch_cuda=True,
                torch_cuda_version="12.4",
            ),
            "cu124",
        )

    def test_auto_cuda_resolves_only_supported_pinned_profiles(self):
        driver = DriverInfo(nvidia=True)
        for build_version, expected in (("11.8", "cu118"), ("12.4", "cu124")):
            with self.subTest(build_version=build_version):
                self.assertEqual(
                    choose_compute(
                        "auto",
                        driver=driver,
                        torch_cuda=True,
                        torch_cuda_version=build_version,
                    ),
                    expected,
                )

    def test_auto_cuda_rejects_missing_or_unsupported_torch_build(self):
        driver = DriverInfo(nvidia=True)
        for build_version in (None, "12.1"):
            with self.subTest(build_version=build_version):
                with self.assertRaisesRegex(
                    RuntimeError, "CUDA profile verification failed"
                ):
                    choose_compute(
                        "auto",
                        driver=driver,
                        torch_cuda=True,
                        torch_cuda_version=build_version,
                    )

    def test_explicit_cuda_rejects_profile_build_mismatch_both_directions(self):
        driver = DriverInfo(nvidia=True)
        for requested, build_version in (("cu118", "12.4"), ("cu124", "11.8")):
            with self.subTest(requested=requested, build_version=build_version):
                with self.assertRaisesRegex(
                    RuntimeError, "CUDA profile verification failed"
                ):
                    choose_compute(
                        requested,
                        driver=driver,
                        torch_cuda=True,
                        torch_cuda_version=build_version,
                    )

    def test_unknown_compute_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requested compute profile"):
            choose_compute("mps", driver=DriverInfo(nvidia=False), torch_cuda=False)


class _FakeTensor:
    def __init__(self, value: float):
        self.value = value

    def __add__(self, other: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(self.value + other.value)

    def item(self) -> float:
        return self.value


class _FakeCuda:
    def __init__(self, *, available: bool, sync_error: Exception | None = None):
        self._available = available
        self._sync_error = sync_error
        self.synchronized = False
        self.synchronized_indices: list[int] = []
        self.device_name_indices: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int) -> str:
        self.device_name_indices.append(index)
        return "Fake GPU" if index == 0 else f"Fake GPU {index}"

    def synchronize(self, index: int) -> None:
        self.synchronized_indices.append(index)
        if self._sync_error is not None:
            raise self._sync_error
        self.synchronized = True


class _FakeTorch:
    __version__ = "2.5.1"

    def __init__(
        self,
        *,
        cuda_available: bool,
        cuda_build: str | None,
        allocation_error: Exception | None = None,
        sync_error: Exception | None = None,
    ):
        self.version = SimpleNamespace(cuda=cuda_build)
        self.cuda = _FakeCuda(available=cuda_available, sync_error=sync_error)
        self.allocation_error = allocation_error
        self.allocations: list[tuple[tuple[int, ...], str]] = []

    def ones(self, shape: tuple[int, ...], *, device: str) -> _FakeTensor:
        if self.allocation_error is not None:
            raise self.allocation_error
        self.allocations.append((shape, device))
        return _FakeTensor(1.0)


class TorchVerificationTests(unittest.TestCase):
    def test_torch_is_imported_lazily_through_injected_importer(self):
        torch = _FakeTorch(cuda_available=False, cuda_build=None)
        importer = Mock(return_value=torch)

        verification = verify_torch_compute("cpu", importer=importer)

        importer.assert_called_once_with("torch")
        self.assertEqual(torch.allocations, [((1,), "cpu")])
        self.assertEqual(verification.device_name, "CPU")
        self.assertEqual(verification.torch_version, "2.5.1")
        self.assertIsNone(verification.torch_cuda_version)

    def test_cuda_verification_allocates_checks_kernel_and_synchronizes(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.4")

        verification = verify_torch_compute("cu124", torch_module=torch)

        self.assertEqual(torch.allocations, [((1,), "cuda:0")])
        self.assertTrue(torch.cuda.synchronized)
        self.assertEqual(torch.cuda.synchronized_indices, [0])
        self.assertEqual(verification.selected_profile, "cu124")
        self.assertEqual(verification.device, "cuda:0")
        self.assertEqual(verification.device_name, "Fake GPU")
        self.assertEqual(verification.device_index, 0)

    def test_cuda_verification_uses_selected_device_index_consistently(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.4")
        driver = DriverInfo(nvidia=True, cuda_profile="cu124", device_index=1)

        verification = verify_torch_compute(
            "cu124",
            torch_module=torch,
            device_index=1,
            driver=driver,
        )

        self.assertEqual(torch.allocations, [((1,), "cuda:1")])
        self.assertEqual(torch.cuda.device_name_indices, [1])
        self.assertEqual(torch.cuda.synchronized_indices, [1])
        self.assertEqual(verification.device, "cuda:1")
        self.assertEqual(verification.device_index, 1)
        self.assertEqual(verification.device_name, "Fake GPU 1")
        self.assertEqual(verification.as_dict()["device_index"], 1)

    def test_torch_verification_rejects_driver_device_index_mismatch(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.4")
        driver = DriverInfo(nvidia=True, cuda_profile="cu124", device_index=1)
        with self.assertRaisesRegex(RuntimeError, "device index.*mismatch"):
            verify_torch_compute(
                "cu124",
                torch_module=torch,
                device_index=0,
                driver=driver,
            )
        self.assertEqual(torch.allocations, [])

    def test_torch_verification_rejects_invalid_device_indexes(self):
        for index in (True, -1, 1.5):
            with self.subTest(index=index):
                torch = _FakeTorch(cuda_available=True, cuda_build="12.4")
                with self.assertRaises((TypeError, ValueError)):
                    verify_torch_compute(
                        "cu124",
                        torch_module=torch,
                        device_index=index,  # type: ignore[arg-type]
                    )
                self.assertEqual(torch.allocations, [])

    def test_cuda_unavailable_never_falls_back_during_torch_verification(self):
        torch = _FakeTorch(cuda_available=False, cuda_build="12.4")
        with self.assertRaisesRegex(RuntimeError, "CUDA profile verification failed"):
            verify_torch_compute("cu124", torch_module=torch)
        self.assertEqual(torch.allocations, [])

    def test_direct_cuda_verification_rejects_profile_build_mismatch(self):
        for selected, build_version in (("cu118", "12.4"), ("cu124", "11.8")):
            with self.subTest(selected=selected, build_version=build_version):
                torch = _FakeTorch(cuda_available=True, cuda_build=build_version)
                with self.assertRaisesRegex(
                    RuntimeError, "CUDA profile verification failed"
                ):
                    verify_torch_compute(selected, torch_module=torch)
                self.assertEqual(torch.allocations, [])

    def test_generic_cuda_profile_is_rejected_as_ambiguous(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.4")
        with self.assertRaisesRegex(ValueError, "selected compute profile"):
            verify_torch_compute("cuda", torch_module=torch)

    def test_cpu_verification_ignores_unsupported_cuda_build(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.1")

        verification = verify_torch_compute("cpu", torch_module=torch)

        self.assertEqual(verification.selected_profile, "cpu")
        self.assertEqual(torch.allocations, [((1,), "cpu")])
        self.assertFalse(torch.cuda.synchronized)
        self.assertEqual(torch.cuda.synchronized_indices, [])

    def test_import_allocation_and_sync_failures_are_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "Torch import failed.*not installed"):
            verify_torch_compute("cpu", importer=Mock(side_effect=ImportError("not installed")))

        broken_allocation = _FakeTorch(
            cuda_available=False,
            cuda_build=None,
            allocation_error=RuntimeError("allocator failed"),
        )
        with self.assertRaisesRegex(RuntimeError, "tensor allocation.*allocator failed"):
            verify_torch_compute("cpu", torch_module=broken_allocation)

        broken_sync = _FakeTorch(
            cuda_available=True,
            cuda_build="12.4",
            sync_error=RuntimeError("sync failed"),
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA synchronization.*sync failed"):
            verify_torch_compute("cu124", torch_module=broken_sync)
        self.assertEqual(broken_sync.cuda.synchronized_indices, [0])

    def test_failed_tensor_operation_is_actionable(self):
        class WrongTensor(_FakeTensor):
            def __add__(self, other: "_FakeTensor") -> "_FakeTensor":
                return _FakeTensor(3.0)

        torch = _FakeTorch(cuda_available=False, cuda_build=None)
        torch.ones = lambda shape, *, device: WrongTensor(1.0)  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "tensor operation.*expected=2"):
            verify_torch_compute("cpu", torch_module=torch)

    def test_package_versions_use_metadata_without_importing_packages(self):
        versions = {"torch": "2.5.1", "pyarrow": "18.0.0"}
        version_fn = Mock(side_effect=versions.__getitem__)

        result = collect_package_versions(("torch", "pyarrow"), version_fn=version_fn)

        self.assertEqual(result, versions)
        self.assertEqual(version_fn.call_count, 2)

    def test_metadata_failure_is_actionable(self):
        with self.assertRaisesRegex(ResourceProbeError, "package version.*torch.*metadata broken"):
            collect_package_versions(
                ("torch",), version_fn=Mock(side_effect=RuntimeError("metadata broken"))
            )

    def test_post_install_report_records_compute_resources_and_packages(self):
        torch = _FakeTorch(cuda_available=True, cuda_build="12.4")
        verification = verify_torch_compute("cu124", torch_module=torch)
        report = build_post_install_report(
            verification=verification,
            driver=DriverInfo(
                nvidia=True,
                driver_version="555.42",
                device_name="Fake GPU",
                memory_bytes=24 * 1024**3,
                cuda_profile="cu124",
                device_index=0,
            ),
            cpu=CPUFacts(logical_count=16, used_fallback=False),
            ram=RAMFacts(
                total_bytes=64 * 1024**3,
                available_bytes=48 * 1024**3,
                platform="windows",
            ),
            disk=DiskFacts(required_bytes=100, available_bytes=200, path="C:/data"),
            package_names=("torch", "pyarrow"),
            version_fn={"torch": "2.5.1", "pyarrow": "18.0.0"}.__getitem__,
        )

        serialized = report.as_dict()
        self.assertEqual(serialized["compute"]["selected_profile"], "cu124")
        self.assertEqual(serialized["compute"]["device"], "cuda:0")
        self.assertEqual(serialized["compute"]["device_name"], "Fake GPU")
        self.assertEqual(serialized["compute"]["torch_cuda_version"], "12.4")
        self.assertEqual(serialized["compute"]["device_index"], 0)
        self.assertEqual(serialized["compute"]["driver"]["device_index"], 0)
        self.assertEqual(serialized["resources"]["cpu"]["logical_count"], 16)
        self.assertEqual(serialized["resources"]["ram"]["available_bytes"], 48 * 1024**3)
        self.assertEqual(serialized["resources"]["disk"]["required_bytes"], 100)
        self.assertEqual(serialized["installed_packages"]["pyarrow"], "18.0.0")
        json.dumps(serialized)

    def test_post_install_report_rejects_contradictory_cuda_facts(self):
        verification = verify_torch_compute(
            "cu124",
            torch_module=_FakeTorch(cuda_available=True, cuda_build="12.4"),
        )
        cpu = CPUFacts(logical_count=4, used_fallback=False)
        ram = RAMFacts(total_bytes=100, available_bytes=25, platform="posix")
        disk = DiskFacts(required_bytes=10, available_bytes=20, path="C:/data")
        contradictory_drivers = (
            DriverInfo(nvidia=False),
            DriverInfo(nvidia=True, cuda_profile="cu118", device_index=0),
            DriverInfo(nvidia=True, cuda_profile="cu124", device_index=1),
        )

        for driver in contradictory_drivers:
            with self.subTest(driver=driver):
                with self.assertRaisesRegex(ResourceProbeError, "contradictory"):
                    build_post_install_report(
                        verification=verification,
                        driver=driver,
                        cpu=cpu,
                        ram=ram,
                        disk=disk,
                        package_names=(),
                    )

    def test_post_install_report_rejects_cpu_verification_labeled_cuda(self):
        verification = TorchVerification(
            selected_profile="cpu",
            device="cuda:0",
            device_name="CPU",
            torch_version="2.5.1",
            torch_cuda_version="12.4",
        )
        with self.assertRaisesRegex(ResourceProbeError, "contradictory"):
            build_post_install_report(
                verification=verification,
                driver=DriverInfo(nvidia=True, device_index=0),
                cpu=CPUFacts(logical_count=4, used_fallback=False),
                ram=RAMFacts(total_bytes=100, available_bytes=25, platform="posix"),
                disk=DiskFacts(required_bytes=10, available_bytes=20),
                package_names=(),
            )

    def test_cpu_runtime_report_allows_cuda_capable_installed_wheel(self):
        verification = verify_torch_compute(
            "cpu",
            torch_module=_FakeTorch(cuda_available=True, cuda_build="12.4"),
        )
        report = build_post_install_report(
            verification=verification,
            driver=DriverInfo(nvidia=True, cuda_profile="cu124", device_index=1),
            cpu=CPUFacts(logical_count=4, used_fallback=False),
            ram=RAMFacts(total_bytes=100, available_bytes=25, platform="posix"),
            disk=DiskFacts(required_bytes=10, available_bytes=20),
            package_names=(),
        )

        self.assertEqual(report.as_dict()["compute"]["selected_profile"], "cpu")
        self.assertEqual(report.as_dict()["compute"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
