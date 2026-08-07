[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$EncodedArguments,
    [switch]$Elevated,
    [switch]$PauseOnError,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BootstrapArguments
)

$ErrorActionPreference = "Stop"

function Test-BitGuardAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-BitGuardOption {
    param(
        [string[]]$Arguments,
        [string[]]$Names
    )

    foreach ($argument in $Arguments) {
        foreach ($name in $Names) {
            if ($argument -eq $name -or $argument.StartsWith("$name=", [StringComparison]::Ordinal)) {
                return $true
            }
        }
    }
    return $false
}

function Get-BitGuardOptionValue {
    param(
        [string[]]$Arguments,
        [string]$Name
    )

    $value = $null
    for ($index = 0; $index -lt $Arguments.Count; $index++) {
        $argument = $Arguments[$index]
        if ($argument.StartsWith("$Name=", [StringComparison]::Ordinal)) {
            $value = $argument.Substring($Name.Length + 1)
        }
        elseif ($argument -eq $Name -and $index + 1 -lt $Arguments.Count) {
            $value = $Arguments[$index + 1]
        }
    }
    return $value
}

function Resolve-BitGuardDisplayPath {
    param(
        [string]$Path,
        [string]$BasePath
    )

    if ($Path -eq "~") {
        $Path = $env:USERPROFILE
    }
    elseif ($Path.StartsWith("~\", [StringComparison]::Ordinal) -or
        $Path.StartsWith("~/", [StringComparison]::Ordinal)) {
        $Path = Join-Path $env:USERPROFILE $Path.Substring(2)
    }
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Get-BitGuardFailureSummary {
    param(
        [int]$ExitCode,
        [string]$ReportPath,
        [string]$LogPath
    )

    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add("BitGuard setup failed with exit code $ExitCode.") | Out-Null
    $resolvedReportPath = $ReportPath
    try {
        if (Test-Path -LiteralPath $ReportPath -PathType Leaf) {
            $report = Get-Content -Raw -LiteralPath $ReportPath | ConvertFrom-Json
            if ($report.report_path) {
                $resolvedReportPath = [string]$report.report_path
            }
            if ($report.failed_stage) {
                $lines.Add("Failed stage: $($report.failed_stage)") | Out-Null
            }
            if ($report.last_completed_stage) {
                $lines.Add(
                    "Last completed stage: $($report.last_completed_stage)"
                ) | Out-Null
            }
            if ($report.error) {
                $lines.Add("Error: $($report.error)") | Out-Null
            }
            if ($report.recovery_command) {
                $lines.Add("Recovery: $($report.recovery_command)") | Out-Null
            }
        }
        else {
            $lines.Add("Failure report was not found at the expected path.") | Out-Null
        }
    }
    catch {
        $lines.Add("Failure report could not be read: $($_.Exception.Message)") | Out-Null
    }
    $lines.Add("Report: $resolvedReportPath") | Out-Null
    $lines.Add("Log:    $LogPath") | Out-Null
    return $lines
}

function Wait-BitGuardFailureWindow {
    param([switch]$Enabled)

    if (-not $Enabled -or $env:BITGUARD_NO_PAUSE -eq "1") {
        return
    }
    try {
        [void](Read-Host "Press Enter after reviewing the error to close this window")
    }
    catch {
        Write-Warning "Could not wait for input: $($_.Exception.Message)"
    }
}

function Invoke-BitGuardBootstrapSession {
    param(
        [string]$BootstrapScript,
        [string[]]$EffectiveArguments,
        [string]$LogPath,
        [ref]$ExitCode
    )

    $ExitCode.Value = 1
    $transcriptStarted = $false
    try {
        Start-Transcript -LiteralPath $LogPath -Append | Out-Null
        $transcriptStarted = $true
    }
    catch {
        [Console]::Error.WriteLine(
            "BitGuard could not start its transcript: $($_.Exception.Message)"
        )
    }
    try {
        $powershellPath = (Get-Process -Id $PID).Path
        & $powershellPath `
            -NoLogo `
            -NoProfile `
            -ExecutionPolicy Bypass `
            -File $BootstrapScript `
            @EffectiveArguments
        $ExitCode.Value = [int]$LASTEXITCODE
    }
    catch {
        [Console]::Error.WriteLine("BitGuard launcher error: $($_.Exception.Message)")
        $ExitCode.Value = 1
    }
    finally {
        if ($transcriptStarted) {
            try {
                Stop-Transcript | Out-Null
            }
            catch {
                [Console]::Error.WriteLine(
                    "BitGuard could not stop its transcript: $($_.Exception.Message)"
                )
            }
        }
    }
}

if (-not (Test-BitGuardAdministrator)) {
    if ($Elevated) {
        throw "Administrator elevation did not take effect."
    }

    $argumentPayload = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes(
            (ConvertTo-Json -InputObject @($BootstrapArguments) -Compress)
        )
    )
    $quotedScript = $PSCommandPath.Replace("'", "''")
    $pauseOnErrorArgument = if ($PauseOnError) { " -PauseOnError" } else { "" }
    $elevatedCommand = (
        "& '{0}' -Elevated{1} -EncodedArguments '{2}'; exit `$LASTEXITCODE" -f
        $quotedScript,
        $pauseOnErrorArgument,
        $argumentPayload
    )
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($elevatedCommand)
    )
    $powershellPath = (Get-Process -Id $PID).Path
    try {
        $process = Start-Process `
            -FilePath $powershellPath `
            -ArgumentList @(
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", $encodedCommand
            ) `
            -Verb RunAs `
            -Wait `
            -PassThru
    }
    catch {
        Write-Host "Administrator launch failed: $($_.Exception.Message)" -ForegroundColor Red
        Wait-BitGuardFailureWindow -Enabled:$PauseOnError
        exit 1
    }
    exit $process.ExitCode
}

if ($EncodedArguments) {
    $decodedJson = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($EncodedArguments)
    )
    $decodedArguments = $decodedJson | ConvertFrom-Json
    $BootstrapArguments = @($decodedArguments | ForEach-Object { [string]$_ })
}

$repository = Split-Path -Parent $PSScriptRoot
$bootstrapScript = Join-Path $repository "bootstrap.ps1"
$defaultDataRoot = Join-Path $env:USERPROFILE "BitGuardData"
$defaultRunsRoot = Join-Path $env:USERPROFILE "BitGuardRuns"
$requestedDataRoot = Get-BitGuardOptionValue $BootstrapArguments "--data-root"
$requestedRunsRoot = Get-BitGuardOptionValue $BootstrapArguments "--runs-root"
$dataRoot = if ($null -eq $requestedDataRoot) { $defaultDataRoot } else { $requestedDataRoot }
$runsRoot = if ($null -eq $requestedRunsRoot) { $defaultRunsRoot } else { $requestedRunsRoot }
$displayDataRoot = Resolve-BitGuardDisplayPath $dataRoot $repository
$displayRunsRoot = Resolve-BitGuardDisplayPath $runsRoot $repository
$logRoot = Join-Path $env:USERPROFILE "BitGuardLogs"
[IO.Directory]::CreateDirectory($logRoot) | Out-Null
$logPath = Join-Path $logRoot ("bootstrap-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

$effectiveArguments = [Collections.Generic.List[string]]::new()
if (-not (Test-BitGuardOption $BootstrapArguments @("--full", "--dataset"))) {
    $effectiveArguments.Add("--full")
}
if (-not (Test-BitGuardOption $BootstrapArguments @("--compute"))) {
    $effectiveArguments.Add("--compute")
    $effectiveArguments.Add("cu128")
}
if (-not (Test-BitGuardOption $BootstrapArguments @("--accept-botiot-academic-license"))) {
    $effectiveArguments.Add("--accept-botiot-academic-license")
}
if (-not (Test-BitGuardOption $BootstrapArguments @("--data-root"))) {
    $effectiveArguments.Add("--data-root")
    $effectiveArguments.Add($defaultDataRoot)
}
if (-not (Test-BitGuardOption $BootstrapArguments @("--runs-root"))) {
    $effectiveArguments.Add("--runs-root")
    $effectiveArguments.Add($defaultRunsRoot)
}
foreach ($argument in $BootstrapArguments) {
    $effectiveArguments.Add($argument)
}

Write-Host "BitGuard automatic setup and training"
Write-Host "Data: $displayDataRoot"
Write-Host "Runs: $displayRunsRoot"
Write-Host "Log:  $logPath"
Write-Host "Running with Administrator privileges. Existing verified stages and checkpoints will be reused."

$exitCode = 1
Invoke-BitGuardBootstrapSession `
    -BootstrapScript $bootstrapScript `
    -EffectiveArguments $effectiveArguments `
    -LogPath $logPath `
    -ExitCode ([ref]$exitCode)

if ($exitCode -ne 0) {
    $reportPath = Join-Path $displayDataRoot ".bitguard\bootstrap-report.json"
    Write-Host ""
    Get-BitGuardFailureSummary `
        -ExitCode $exitCode `
        -ReportPath $reportPath `
        -LogPath $logPath |
        ForEach-Object { Write-Host $_ -ForegroundColor Red }
    Write-Host "Run start.bat again after correcting the reported cause; completed work will be reused."
    Wait-BitGuardFailureWindow -Enabled:$PauseOnError
}
else {
    Write-Host ""
    Write-Host "BitGuard setup and training completed." -ForegroundColor Green
    Write-Host "Runs: $displayRunsRoot"
}

exit $exitCode
