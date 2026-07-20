import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import bootstrap


class BootstrapEntryTest(unittest.TestCase):
    def test_build_command_forwards_automatic_full_download(self):
        command = bootstrap.build_package_command(
            Path(".venv"),
            ["--full", "--accept-botiot-academic-license"],
        )

        self.assertEqual(
            command[-3:],
            ["bootstrap", "--full", "--accept-botiot-academic-license"],
        )
        self.assertNotIn("--botiot-source", command)

    def test_build_command_forwards_full_source_and_license(self):
        command = bootstrap.build_package_command(
            Path(".venv"),
            [
                "--full",
                "--botiot-source",
                "input.zip",
                "--accept-botiot-academic-license",
            ],
        )
        self.assertEqual(
            command[-5:],
            [
                "bootstrap",
                "--full",
                "--botiot-source",
                "input.zip",
                "--accept-botiot-academic-license",
            ],
        )

    def test_python_outside_supported_range_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.10 through 3.12"):
            bootstrap.validate_python_version((3, 13, 0))

    def test_supported_real_virtual_environment_passes(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            stdout='{"version": [3, 12, 1], "is_venv": true}\n',
            stderr="",
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result) as run:
            bootstrap.validate_virtual_environment(Path(".venv"))

        probe = run.call_args.args[0]
        self.assertEqual(probe[:2], [str(bootstrap.venv_python(Path(".venv"))), "-c"])
        self.assertIn("sys.prefix != sys.base_prefix", probe[2])

    def test_unsupported_virtual_environment_python_is_rejected(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            stdout='{"version": [3, 13, 0], "is_venv": true}\n',
            stderr="",
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Python 3.10 through 3.12"):
                bootstrap.validate_virtual_environment(Path(".venv"))

    def test_non_virtual_environment_interpreter_is_rejected(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            stdout='{"version": [3, 12, 1], "is_venv": false}\n',
            stderr="",
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "not a virtual environment"):
                bootstrap.validate_virtual_environment(Path(".venv"))

    def test_auto_compute_uses_cpu_when_nvidia_smi_is_absent(self):
        with patch("scripts.bootstrap.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(bootstrap._detect_torch_profile(), "cpu")

    def test_auto_compute_rejects_broken_nvidia_smi_executable(self):
        with patch(
            "scripts.bootstrap.subprocess.run", side_effect=OSError("probe failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA detection failed"):
                bootstrap._detect_torch_profile()

    def test_auto_compute_rejects_failed_nvidia_smi_probe(self):
        result = subprocess.CompletedProcess(
            [], 9, stdout="", stderr="driver unavailable"
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "refusing to downgrade to CPU"):
                bootstrap._detect_torch_profile()

    def test_auto_compute_rejects_malformed_nvidia_smi_output(self):
        result = subprocess.CompletedProcess(
            [], 0, stdout="NVIDIA-SMI without CUDA", stderr=""
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "did not report a CUDA version"):
                bootstrap._detect_torch_profile()

    def test_auto_compute_maps_supported_cuda_versions_to_torch_profiles(self):
        cases = (
            ("11.8", "cu118"),
            ("12.3", "cu118"),
            ("12.4", "cu124"),
            ("12.7", "cu124"),
            ("12.8", "cu128"),
            ("12.9", "cu128"),
        )
        for cuda_version, expected_profile in cases:
            with self.subTest(cuda_version=cuda_version):
                result = subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=f"NVIDIA-SMI 570.00    CUDA Version: {cuda_version}\n",
                    stderr="",
                )
                with patch("scripts.bootstrap.subprocess.run", return_value=result):
                    self.assertEqual(
                        bootstrap._detect_torch_profile(), expected_profile
                    )

    def test_auto_compute_rejects_cuda_below_cu118_threshold(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            stdout="NVIDIA-SMI 470.00    CUDA Version: 11.7\n",
            stderr="",
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(
                RuntimeError, "below the supported cu118 profile"
            ):
                bootstrap._detect_torch_profile()

    def test_cuda_verification_runs_allocation_kernel_and_synchronization(self):
        result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("scripts.bootstrap.subprocess.run", return_value=result) as run:
            bootstrap._verify_torch_profile(Path(".venv"), "cu124")

        verification = run.call_args.args[0][2]
        self.assertIn("torch.ones", verification)
        self.assertIn("device='cuda'", verification)
        self.assertIn("probe + 1", verification)
        self.assertIn("torch.cuda.synchronize()", verification)

    def test_cu128_verification_requires_cuda_12_8(self):
        result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("scripts.bootstrap.subprocess.run", return_value=result) as run:
            bootstrap._verify_torch_profile(Path(".venv"), "cu128")

        verification = run.call_args.args[0][2]
        self.assertIn("expected = '12.8'", verification)

    def test_cuda_verification_propagates_probe_failure(self):
        result = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="kernel launch failed"
        )
        with patch("scripts.bootstrap.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
                bootstrap._verify_torch_profile(Path(".venv"), "cu118")

    def _exercise_main(
        self,
        *,
        environment_exists: bool = True,
        package_exit_status: int = 0,
        arguments: list[str] | None = None,
        detected_profile: str = "cu124",
    ) -> tuple[int, list[tuple[str, object]]]:
        events: list[tuple[str, object]] = []

        def record_run(command, **_kwargs):
            events.append(("run", command))
            return subprocess.CompletedProcess(command, 0)

        def record_validate(environment):
            events.append(("validate", environment))

        def record_verify(_environment, profile):
            events.append(("verify", profile))

        def record_handoff(command, **_kwargs):
            events.append(("handoff", command))
            return package_exit_status

        with (
            patch("scripts.bootstrap.validate_python_version"),
            patch(
                "scripts.bootstrap._detect_torch_profile",
                return_value=detected_profile,
            ),
            patch.object(Path, "exists", return_value=environment_exists),
            patch(
                "scripts.bootstrap.validate_virtual_environment",
                side_effect=record_validate,
            ),
            patch("scripts.bootstrap._verify_torch_profile", side_effect=record_verify),
            patch("scripts.bootstrap.subprocess.run", side_effect=record_run),
            patch("scripts.bootstrap.subprocess.call", side_effect=record_handoff),
        ):
            status = bootstrap.main(
                ["--compute", "cpu", "--full"] if arguments is None else arguments
            )
        return status, events

    def test_main_validates_reused_environment_and_installs_in_contract_order(self):
        _status, events = self._exercise_main()
        self.assertEqual(
            [kind for kind, _value in events],
            ["validate", "run", "verify", "run", "run", "handoff"],
        )

        install_commands = [value for kind, value in events if kind == "run"]
        self.assertTrue(str(install_commands[0][-1]).endswith("torch-cpu.txt"))
        self.assertEqual(install_commands[1][-1], "--no-deps")
        self.assertEqual(install_commands[1][-3], "--editable")
        self.assertTrue(str(install_commands[2][-1]).endswith("full-base.txt"))
        self.assertEqual(
            events[-1][1][-4:], ["bootstrap", "--compute", "cpu", "--full"]
        )
        self.assertEqual(events[-1][1].count("--compute"), 1)

    def test_auto_compute_handoff_uses_the_resolved_installed_profile_once(self):
        _status, events = self._exercise_main(
            arguments=["--full"], detected_profile="cu124"
        )
        handoff = events[-1][1]
        self.assertEqual(handoff[-4:], ["bootstrap", "--compute", "cu124", "--full"])
        self.assertEqual(handoff.count("--compute"), 1)

    def test_explicit_cu128_uses_matching_lock_and_handoff(self):
        _status, events = self._exercise_main(
            arguments=["--compute", "cu128", "--full"]
        )

        install_commands = [value for kind, value in events if kind == "run"]
        self.assertTrue(str(install_commands[0][-1]).endswith("torch-cu128.txt"))
        self.assertEqual(
            events[-1][1][-4:], ["bootstrap", "--compute", "cu128", "--full"]
        )

    def test_cu128_lock_pins_the_official_torch_wheel(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        lock = repository / "requirements" / "locks" / "torch-cu128.txt"

        self.assertEqual(
            lock.read_text(encoding="utf-8"),
            "--index-url https://download.pytorch.org/whl/cu128\n" "torch==2.11.0\n",
        )

    def test_platform_wrappers_preserve_cli_arguments_for_python_handoff(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        powershell = (repository / "bootstrap.ps1").read_text(encoding="utf-8")
        shell = (repository / "bootstrap.sh").read_text(encoding="utf-8")
        self.assertIn("$scriptPath @args", powershell)
        self.assertIn('"$SCRIPT_DIR/scripts/bootstrap.py" "$@"', shell)

    def test_powershell_wrapper_uses_py_launcher_when_only_python_310_is_registered(
        self,
    ):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is required to exercise bootstrap.ps1")

        repository = Path(bootstrap.__file__).resolve().parents[1]
        wrapper = repository / "bootstrap.ps1"
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = Path(temporary_directory) / "python-arguments.txt"
            script = f"""
function py {{
    if ($args.Count -ge 2 -and $args[1] -eq '-c') {{
        if ($args[0] -eq '-3.10') {{
            $global:LASTEXITCODE = 0
        }} else {{
            $global:LASTEXITCODE = 1
        }}
        return
    }}
    [System.IO.File]::WriteAllLines($env:BITGUARD_CAPTURE, [string[]]$args)
    $global:LASTEXITCODE = 0
}}
$env:Path = ''
& '{wrapper}' --compute cpu --full
"""
            result = subprocess.run(
                [powershell, "-NoProfile", "-Command", script],
                capture_output=True,
                check=False,
                encoding="utf-8",
                env={**os.environ, "BITGUARD_CAPTURE": str(capture)},
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                [
                    "-3.10",
                    str(repository / "scripts" / "bootstrap.py"),
                    "--compute",
                    "cpu",
                    "--full",
                ],
            )

    def test_main_validates_new_environment_before_installing(self):
        _status, events = self._exercise_main(environment_exists=False)
        self.assertEqual([kind for kind, _value in events[:2]], ["run", "validate"])
        self.assertEqual(events[0][1][1:3], ["-m", "venv"])

    def test_main_propagates_package_exit_status(self):
        status, _events = self._exercise_main(package_exit_status=23)
        self.assertEqual(status, 23)


if __name__ == "__main__":
    unittest.main()
