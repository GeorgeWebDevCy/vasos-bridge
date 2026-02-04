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
    --name "CucumberDestroyer_TranslatorGUI" `
    --icon "cucumber.ico" `
    --add-data "cucumber.png;." `
    --collect-data langdetect `
    translator_gui.py

& $python -m PyInstaller `
    --noconsole `
    --onefile `
    --name "CucumberDestroyer_PotTranslatorGUI" `
    --icon "cucumber.ico" `
    --add-data "cucumber.png;." `
    pot_translator_gui.py

Write-Host "Built dist\\CucumberDestroyer_TranslatorGUI.exe"
Write-Host "Built dist\\CucumberDestroyer_PotTranslatorGUI.exe"
