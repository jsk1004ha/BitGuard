[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$EncodedArguments,
    [switch]$Elevated,
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
    $elevatedCommand = (
        "& '{0}' -Elevated -EncodedArguments '{1}'; exit `$LASTEXITCODE" -f
        $quotedScript,
        $argumentPayload
    )
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($elevatedCommand)
    )
    $powershellPath = (Get-Process -Id $PID).Path
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

$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $logPath -Append | Out-Null
    $transcriptStarted = $true
    & $bootstrapScript @effectiveArguments
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

if ($exitCode -ne 0) {
    $reportPath = Join-Path $displayDataRoot ".bitguard\bootstrap-report.json"
    Write-Host ""
    Write-Host "Automatic attempts ended with exit code $exitCode." -ForegroundColor Red
    Write-Host "Report: $reportPath"
    Write-Host "Log:    $logPath"
    Write-Host "Run start.bat again after correcting the reported cause; completed work will be reused."
}
else {
    Write-Host ""
    Write-Host "BitGuard setup and training completed." -ForegroundColor Green
    Write-Host "Runs: $displayRunsRoot"
}

exit $exitCode
