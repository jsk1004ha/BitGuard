import json
import os
import shutil
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import bootstrap


class BootstrapEntryTest(unittest.TestCase):
    def test_failed_package_bootstrap_is_retried_with_the_exact_same_command(self):
        command = ["python", "-m", "bitguard_bnn", "bootstrap", "--full"]
        calls: list[tuple[list[str], Path]] = []

        def invoke(candidate, *, cwd):
            calls.append((list(candidate), cwd))
            return 1 if len(calls) == 1 else 0

        delays: list[float] = []
        status = bootstrap.run_package_with_retries(
            command,
            cwd=Path("repository"),
            attempts=3,
            invoke=invoke,
            sleeper=delays.append,
            stream=StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            calls,
            [(command, Path("repository")), (command, Path("repository"))],
        )
        self.assertEqual(delays, [2.0])

    def test_usage_error_is_not_retried(self):
        calls: list[list[str]] = []

        def invoke(command, *, cwd):
            del cwd
            calls.append(list(command))
            return 2

        status = bootstrap.run_package_with_retries(
            ["python", "-m", "bitguard_bnn", "bootstrap"],
            cwd=Path("repository"),
            attempts=3,
            invoke=invoke,
            sleeper=lambda _seconds: self.fail("usage errors must not sleep"),
            stream=StringIO(),
        )

        self.assertEqual(status, 2)
        self.assertEqual(len(calls), 1)

    def test_bootstrap_retry_stops_at_the_attempt_limit(self):
        attempts: list[int] = []
        delays: list[float] = []

        status = bootstrap.run_package_with_retries(
            ["python", "-m", "bitguard_bnn", "bootstrap"],
            cwd=Path("repository"),
            attempts=3,
            invoke=lambda _command, *, cwd: attempts.append(len(str(cwd))) or 1,
            sleeper=delays.append,
            stream=StringIO(),
        )

        self.assertEqual(status, 1)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(delays, [2.0, 5.0])

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
        self.assertIn("--no-build-isolation", install_commands[1])
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

    def test_windows_start_launcher_is_elevated_resumable_and_repo_relative(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        batch = (repository / "start.bat").read_text(encoding="utf-8")
        launcher = (repository / "scripts" / "start.ps1").read_text(encoding="utf-8")

        self.assertIn(r"%~dp0scripts\start.ps1", batch)
        self.assertIn("%*", batch)
        self.assertIn('set "BITGUARD_EXIT_CODE=%ERRORLEVEL%"', batch)
        self.assertIn("exit /b %BITGUARD_EXIT_CODE%", batch)
        self.assertIn("-Verb RunAs", launcher)
        self.assertIn("CmdletBinding(PositionalBinding = $false)", launcher)
        self.assertIn("-EncodedCommand", launcher)
        self.assertIn("EncodedArguments", launcher)
        self.assertNotIn("GetTempPath", launcher)
        self.assertNotIn("ArgumentFile", launcher)
        self.assertIn("--full", launcher)
        self.assertIn("--compute", launcher)
        self.assertIn("cu128", launcher)
        self.assertIn("--accept-botiot-academic-license", launcher)
        self.assertIn("BitGuardData", launcher)
        self.assertIn("BitGuardRuns", launcher)
        self.assertIn("Get-BitGuardOptionValue", launcher)
        self.assertIn("$displayDataRoot", launcher)
        self.assertIn("$displayRunsRoot", launcher)
        self.assertNotIn("--restart-stage", launcher)

    def test_windows_start_launcher_keeps_the_elevated_failure_window_open(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        batch = (repository / "start.bat").read_text(encoding="utf-8")
        launcher = (repository / "scripts" / "start.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("-PauseOnError", batch)
        self.assertNotIn("pause >nul", batch.casefold())
        self.assertIn("[switch]$PauseOnError", launcher)
        self.assertIn("-PauseOnError", launcher)
        self.assertIn("function Wait-BitGuardFailureWindow", launcher)
        self.assertIn("Read-Host", launcher)

    @unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell is required")
    def test_windows_start_launcher_summarizes_the_failure_report(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        launcher = repository / "scripts" / "start.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "bootstrap-report.json"
            log = root / "bootstrap.log"
            report.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failed_stage": "inspect",
                        "last_completed_stage": "extract",
                        "error": "SchemaInspectionError: malformed fixture",
                        "recovery_command": "rerun the original command",
                        "report_path": str(report),
                    }
                ),
                encoding="utf-8",
            )
            probe = r"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:BITGUARD_START_LAUNCHER, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) { exit 91 }
$function = $ast.FindAll(
    {
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Get-BitGuardFailureSummary'
    },
    $true
) | Select-Object -First 1
if ($null -eq $function) { exit 92 }
$body = @'
Get-BitGuardFailureSummary `
    -ExitCode 1 `
    -ReportPath $env:BITGUARD_TEST_REPORT `
    -LogPath $env:BITGUARD_TEST_LOG
'@
$summaryProbe = [ScriptBlock]::Create("$($function.Extent.Text)`n$body")
& $summaryProbe
"""
            environment = os.environ.copy()
            environment["BITGUARD_START_LAUNCHER"] = str(launcher)
            environment["BITGUARD_TEST_REPORT"] = str(report)
            environment["BITGUARD_TEST_LOG"] = str(log)
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", probe],
                text=True,
                capture_output=True,
                env=environment,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("inspect", result.stdout)
        self.assertIn("extract", result.stdout)
        self.assertIn("SchemaInspectionError: malformed fixture", result.stdout)
        self.assertIn("rerun the original command", result.stdout)
        self.assertIn(str(report), result.stdout)
        self.assertIn(str(log), result.stdout)

    @unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell is required")
    def test_windows_start_launcher_appends_failure_summary_after_child_transcript(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        launcher = repository / "scripts" / "start.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            user_profile = Path(temporary)
            log_root = user_profile / "BitGuardLogs"
            log_root.mkdir()
            log = log_root / "bootstrap-20260807-120000.log"
            log.write_text("child transcript ended\n", encoding="utf-8")
            report = user_profile / "bootstrap-report.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failed_stage": "inspect",
                        "last_completed_stage": "extract",
                        "error": "SchemaInspectionError: malformed fixture",
                        "recovery_command": "rerun the original command",
                        "report_path": str(report),
                    }
                ),
                encoding="utf-8",
            )
            probe = r"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:BITGUARD_START_LAUNCHER, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) { exit 91 }
$wanted = @('Get-BitGuardFailureSummary', 'Write-BitGuardFailureSummary')
$functions = $ast.FindAll(
    {
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $wanted -contains $node.Name
    },
    $true
) | ForEach-Object { $_.Extent.Text }
if ($functions.Count -ne $wanted.Count) { exit 92 }
$body = @'
Write-BitGuardFailureSummary `
    -ExitCode 7 `
    -ReportPath $env:BITGUARD_TEST_REPORT `
    -LogPath $env:BITGUARD_TEST_LOG
'@
$summaryProbe = [ScriptBlock]::Create("$($functions -join "`n")`n$body")
& $summaryProbe
"""
            environment = os.environ.copy()
            environment["USERPROFILE"] = str(user_profile)
            environment["BITGUARD_START_LAUNCHER"] = str(launcher)
            environment["BITGUARD_TEST_REPORT"] = str(report)
            environment["BITGUARD_TEST_LOG"] = str(log)
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", probe],
                text=True,
                capture_output=True,
                env=environment,
            )
            logged = log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(logged.index("child transcript ended"), logged.index("exit code 7"))
        self.assertIn("Failed stage: inspect", logged)
        self.assertIn("SchemaInspectionError: malformed fixture", logged)
        self.assertIn("Recovery: rerun the original command", logged)
        self.assertIn(f"Report: {report}", logged)
        self.assertIn(f"Log:    {log}", logged)

    @unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell is required")
    def test_windows_start_launcher_captures_child_exit_and_returns_to_parent(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        launcher = repository / "scripts" / "start.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failing_bootstrap = root / "failing-bootstrap.ps1"
            failing_bootstrap.write_text(
                "param([Parameter(ValueFromRemainingArguments = $true)]"
                "[string[]]$Forwarded)\n"
                '[Console]::Error.WriteLine("synthetic bootstrap failure")\n'
                '[Console]::Error.WriteLine("ARGS=" + ($Forwarded -join "|"))\n'
                "exit 7\n",
                encoding="utf-8",
            )
            log = root / "bootstrap.log"
            probe = r"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:BITGUARD_START_LAUNCHER, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) { exit 91 }
$function = $ast.FindAll(
    {
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq 'Invoke-BitGuardBootstrapSession'
    },
    $true
) | Select-Object -First 1
if ($null -eq $function) { exit 92 }
$body = @'
$exitCode = 0
$arguments = [Collections.Generic.List[string]]::new()
@('--compute', 'cpu', '--data-root', 'C:\A B\Data') |
    ForEach-Object { $arguments.Add($_) } | Out-Null
Invoke-BitGuardBootstrapSession `
    -BootstrapScript $env:BITGUARD_TEST_BOOTSTRAP `
    -EffectiveArguments $arguments `
    -LogPath $env:BITGUARD_TEST_LOG `
    -ExitCode ([ref]$exitCode)
[pscustomobject]@{
    ExitCode = $exitCode
    ReachedAfterCatch = $true
} | ConvertTo-Json -Compress
'@
$sessionProbe = [ScriptBlock]::Create("$($function.Extent.Text)`n$body")
& $sessionProbe
"""
            environment = os.environ.copy()
            environment["BITGUARD_START_LAUNCHER"] = str(launcher)
            environment["BITGUARD_TEST_BOOTSTRAP"] = str(failing_bootstrap)
            environment["BITGUARD_TEST_LOG"] = str(log)
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", probe],
                text=True,
                capture_output=True,
                env=environment,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["ExitCode"], 7)
        self.assertTrue(payload["ReachedAfterCatch"])
        self.assertIn("synthetic bootstrap failure", result.stderr)
        self.assertIn("ARGS=--compute|cpu|--data-root|C:\\A B\\Data", result.stderr)

    @unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell is required")
    def test_windows_start_launcher_preserves_dash_prefixed_user_arguments(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        launcher = repository / "scripts" / "start.ps1"
        probe = r"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:BITGUARD_START_LAUNCHER, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) { exit 91 }
$attributes = ($ast.ParamBlock.Attributes | ForEach-Object { $_.Extent.Text }) -join "`n"
$parameters = $ast.ParamBlock.Extent.Text
$body = @'
[pscustomobject]@{
    Encoded = $EncodedArguments
    Elevated = [bool]$Elevated
    Remaining = @($BootstrapArguments)
} | ConvertTo-Json -Compress
'@
$bindingProbe = [ScriptBlock]::Create("$attributes`n$parameters`n$body")
& $bindingProbe --compute cpu --data-root 'C:\Users\A B\BitGuardData'
"""
        environment = os.environ.copy()
        environment["BITGUARD_START_LAUNCHER"] = str(launcher)
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-Command",
                probe,
            ],
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["Encoded"], "")
        self.assertFalse(payload["Elevated"])
        self.assertEqual(
            payload["Remaining"],
            ["--compute", "cpu", "--data-root", r"C:\Users\A B\BitGuardData"],
        )

    @unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell is required")
    def test_windows_start_launcher_reports_overridden_roots(self):
        repository = Path(bootstrap.__file__).resolve().parents[1]
        launcher = repository / "scripts" / "start.ps1"
        probe = r"""
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:BITGUARD_START_LAUNCHER, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) { exit 91 }
$wanted = @('Get-BitGuardOptionValue', 'Resolve-BitGuardDisplayPath')
$functions = $ast.FindAll(
    {
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
            $wanted -contains $node.Name
    },
    $true
) | ForEach-Object { $_.Extent.Text }
$body = @'
$arguments = @(
    '--data-root', 'ignored-data',
    '--runs-root=ignored-runs',
    '--data-root', 'C:\Users\A B\Data',
    '--runs-root=relative-runs'
)
$data = Get-BitGuardOptionValue $arguments '--data-root'
$runs = Get-BitGuardOptionValue $arguments '--runs-root'
[pscustomobject]@{
    Data = Resolve-BitGuardDisplayPath $data 'C:\repository'
    Runs = Resolve-BitGuardDisplayPath $runs 'C:\repository'
    Tilde = Resolve-BitGuardDisplayPath '~\BitGuardData' 'C:\repository'
} | ConvertTo-Json -Compress
'@
$rootProbe = [ScriptBlock]::Create("$($functions -join "`n")`n$body")
& $rootProbe
"""
        environment = os.environ.copy()
        environment["BITGUARD_START_LAUNCHER"] = str(launcher)
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", probe],
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["Data"], r"C:\Users\A B\Data")
        self.assertEqual(payload["Runs"], r"C:\repository\relative-runs")
        self.assertEqual(payload["Tilde"], str(Path.home() / "BitGuardData"))

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
