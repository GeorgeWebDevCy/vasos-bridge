$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

$python = Join-Path $venvPath "Scripts\\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install pyinstaller openai langdetect

& $python -m PyInstaller `
    --noconsole `
    --onefile `
    --name TranslatorGUI `
    --collect-data langdetect `
    translator_gui.py

Write-Host "Built dist\\TranslatorGUI.exe"
