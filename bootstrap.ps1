$scriptPath = Join-Path $PSScriptRoot "scripts/bootstrap.py"
$pythonCommand = $null
$pythonArguments = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("-3.12", "-3.11", "-3.10")) {
        & py $version -c "import sys" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = "py"
            $pythonArguments = @($version)
            break
        }
    }
}

if ($null -eq $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $pythonCommand = "python"
}

if ($null -eq $pythonCommand) {
    Write-Error "Python 3.10 through 3.12 is required."
    exit 1
}

& $pythonCommand @pythonArguments $scriptPath @args
exit $LASTEXITCODE
