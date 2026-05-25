#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if ! python3 - <<'PY'
import tkinter
PY
then
    echo "Python Tk support is required to build the Linux GUI applications." >&2
    echo "On Ubuntu/Debian, install it with: sudo apt install python3-tk" >&2
    exit 1
fi

venv_path="$repo_root/.venv-linux"
python="$venv_path/bin/python"
if [[ ! -x "$python" ]]; then
    python3 -m venv "$venv_path"
fi

"$python" -m pip install --upgrade pip
"$python" -m pip install pyinstaller openai anthropic langdetect

"$python" -m PyInstaller \
    --noconfirm \
    --clean \
    --workpath build-linux \
    --noconsole \
    --onefile \
    --name CucumberDestroyer_TranslatorGUI-linux-x86_64 \
    --add-data "cucumber.png:." \
    --add-data "cucumber.ico:." \
    --collect-data langdetect \
    --hidden-import openai \
    --hidden-import anthropic \
    translator_gui.py

"$python" -m PyInstaller \
    --noconfirm \
    --clean \
    --workpath build-linux \
    --noconsole \
    --onefile \
    --name CucumberDestroyer_PotTranslatorGUI-linux-x86_64 \
    --add-data "cucumber.png:." \
    --add-data "cucumber.ico:." \
    --hidden-import openai \
    --hidden-import anthropic \
    pot_translator_gui.py

echo "Built dist/CucumberDestroyer_TranslatorGUI-linux-x86_64"
echo "Built dist/CucumberDestroyer_PotTranslatorGUI-linux-x86_64"
