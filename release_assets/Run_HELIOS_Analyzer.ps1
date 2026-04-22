param(
    [switch]$SkipInstall,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Resolve-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python 3.10+ was not found in PATH."
}

$pythonCommand = Resolve-Python
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    if ($pythonCommand.Length -gt 1) {
        & $pythonCommand[0] @($pythonCommand[1..($pythonCommand.Length - 1)]) -m venv .venv
    }
    else {
        & $pythonCommand[0] -m venv .venv
    }
}

if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e ".[desktop]"
}

if ($NoLaunch) {
    Write-Host "Environment ready at $venvPython"
    exit 0
}

& $venvPython -m helios_app
