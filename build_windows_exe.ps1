$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $($Command -join ' ')"
    }
}

function Test-WorkingPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath
    )

    if (-not (Test-Path $PythonPath)) {
        return $false
    }

    & $PythonPath --version *> $null
    return $LASTEXITCODE -eq 0
}

$venvPath = Join-Path $repoRoot ".venv"
$python = Join-Path $venvPath "Scripts\\python.exe"
if (-not (Test-WorkingPython $python)) {
    if (Test-Path $venvPath) {
        Remove-Item $venvPath -Recurse -Force
    }
    Invoke-CheckedCommand @("python", "-m", "venv", $venvPath)
}

Invoke-CheckedCommand @($python, "-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedCommand @($python, "-m", "pip", "install", "pyinstaller", "openai", "langdetect")

Invoke-CheckedCommand @(
    $python,
    "-m",
    "PyInstaller",
    "--noconsole",
    "--onefile",
    "--name",
    "CucumberDestroyer_TranslatorGUI",
    "--icon",
    "cucumber.ico",
    "--add-data",
    "cucumber.png;.",
    "--add-data",
    "cucumber.ico;.",
    "--collect-data",
    "langdetect",
    "translator_gui.py"
)

Invoke-CheckedCommand @(
    $python,
    "-m",
    "PyInstaller",
    "--noconsole",
    "--onefile",
    "--name",
    "CucumberDestroyer_PotTranslatorGUI",
    "--icon",
    "cucumber.ico",
    "--add-data",
    "cucumber.png;.",
    "--add-data",
    "cucumber.ico;.",
    "pot_translator_gui.py"
)

Write-Host "Built dist\\CucumberDestroyer_TranslatorGUI.exe"
Write-Host "Built dist\\CucumberDestroyer_PotTranslatorGUI.exe"
