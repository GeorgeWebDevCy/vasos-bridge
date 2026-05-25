# Bridge Project translation archive

This repository stores XLIFF 1.2 exports for the Bridge Project website. Each file represents a single translation job sourced from English content and localized into one of six languages (Arabic, Czech, German, Greek, Polish, or Ukrainian).

## File inventory

*   Files are named `The Bridge Project-translation-job-<id>.xliff`, with sequential job IDs from 385 through 2084 (no gaps), for a total of 1,700 files.
*   Original files are typically stored in the `originals/` directory.

### Target-language coverage

| Language | Count |
| --- | --- |
| Arabic (`ar`) | 283 |
| Czech (`cs`) | 283 |
| German (`de`) | 283 |
| Greek (`el`) | 283 |
| Polish (`pl`) | 284 |
| Ukrainian (`uk`) | 284 |

## Setup & Requirements

To use the translation and management tools, you need Python 3.9+ and the following dependencies:

```bash
pip install openai anthropic langdetect
```

### Providers And API Keys

The translators support `openai`, `anthropic` (Claude), and `ollama`.

| Provider | Where to obtain access | Key variable |
| --- | --- | --- |
| OpenAI | Create an API key in the [OpenAI API dashboard](https://platform.openai.com/api-keys). | `OPENAI_API_KEY` |
| Anthropic Claude | Create an API key in [Anthropic Console](https://console.anthropic.com/settings/keys). | `ANTHROPIC_API_KEY` |
| Ollama local | Install and run [Ollama](https://ollama.com/download); no API key is needed for local models. | None |
| Ollama Cloud direct API | Create a key in [Ollama API key settings](https://ollama.com/settings/keys). | `OLLAMA_API_KEY` |

ChatGPT subscriptions and Claude.ai paid plans are separate from their developer APIs. A ChatGPT subscription does not provide OpenAI API usage, and a Claude subscription does not provide Anthropic API usage; these applications need an API key with API billing configured for those providers. The apps do not sign into ChatGPT or Claude through a browser.

For scripts or Python GUIs, create `.env` in this repository root. It is already ignored by git. Add only the provider values you use:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

For direct Ollama Cloud API calls, put this in `.env`:

```env
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
```

For an Ollama server running locally, do not set `OLLAMA_HOST`, or set:

```env
OLLAMA_HOST=http://localhost:11434
```

Ollama also supports running cloud models through the local Ollama service. Run `ollama signin`, then select the `ollama` provider with a cloud model such as `gpt-oss:120b-cloud`; the app continues to call your local endpoint while Ollama handles cloud authentication, so an `OLLAMA_API_KEY` is not required in this application for this route.

For packaged Windows `.exe` or Linux standalone builds, put the same `.env` beside the application file you launch. Never commit or share `.env` files.

### Testing Connections

In either desktop application, select a provider and press **Load Models** beside the model dropdown. This reads available models using the configured API key or Ollama endpoint. Select a model, then press **Test Connection** in the same options area; the result is displayed and written to the app log. The dropdown remains editable for custom or newly installed model names.

The command-line connection tester sends the same kind of very small model request:

```bash
python test_provider_connection.py --provider openai --model gpt-4.1
python test_provider_connection.py --provider anthropic --model claude-sonnet-4-20250514
python test_provider_connection.py --provider ollama --model llama3.2
```

For Ollama Cloud through the local signed-in app:

```bash
ollama signin
python test_provider_connection.py --provider ollama --model gpt-oss:120b-cloud
```

For direct Ollama Cloud API access, set `OLLAMA_HOST` and `OLLAMA_API_KEY` in `.env` as above, then run:

```bash
python test_provider_connection.py --provider ollama --model gpt-oss:120b
```

Any hosted-provider test may incur a small API usage charge.

## Tooling Overview

The repository contains several utility scripts:

*   **`xliff_tools.py`**: The core utility for cataloging, validating, and generating XLIFF files.
*   **`translator_gui.py`**: A desktop GUI for translating files using OpenAI, Claude, or Ollama models.
*   **`translate_one.py`**: A command-line tool for translating single or multiple files via an AI provider.
*   **`prepare_wpml_import.py`**: A script to organize translated files for re-import into WPML.
 *   **`pot_translator_gui.py`**: Tkinter GUI for translating `.pot` templates via selectable AI providers with live progress, compile controls, plural overrides, and dry-run support.
 *   **`translate_pot.py`**: Translate gettext `.pot` templates into per-language `.po` files and optionally compile `.mo` binaries.

### Translating gettext templates

#### CLI

To generate `.po`/`.mo` bundles from a gettext `.pot` template, run:

```bash
python translate_pot.py \
  --input wp-xliff-translator/includes/plugin-update-checker/languages/plugin-update-checker.pot \
  --languages ar,cs,de \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --compile
```

You can provide a fallback context for entries that lack `msgctxt` by using `--default-context`, exporting `GPT_TRANSLATOR_CONTEXT`, or adding `default_context = "..."` under `[tool.gpt-po-translator]` in `pyproject.toml`. The CLI option overrides the environment/configured values. Translations are also tagged with `#. AI-generated` by default; pass `--no-ai-comment` when you want to skip that comment.

The CLI writes `po/<locale>/<domain>-<locale>.po` and, when `--compile` is used, companion `.mo` files. It resolves the proper locale suffix for each language (e.g., translating `masterstudy-lms-learning-management-system.pot` to Greek produces `masterstudy-lms-learning-management-system-el.po`), selects a provider with `--provider openai|anthropic|ollama`, throttles requests via `--rpm`, and preserves comments/context while translating placeholders. Use `--max-entries` to limit how many strings are translated per language, `--plural-forms` to override locale plural rules (`lang=expr`), and `--dry-run` to emit header-only `.po` files without calling an AI provider.

#### GUI

Alternatively, run `python pot_translator_gui.py` to launch a desktop interface. Select one or more `.pot` files, enter a comma-separated list of target languages, choose an output directory (defaults to `po`), optionally enable `.mo` compilation and dry runs, and set plural-form overrides in `lang=expr` form. The GUI also exposes the provider, a loadable model dropdown, connection testing, requests-per-minute throttle, and per-language entry limits before starting a batch. Log output and progress appear in real time, and generated `.po`/`.mo` bundles follow the `po/<locale>/<domain>-<locale>.(po|mo)` pattern.

If you need to strip the AI-generated “References” paragraph from existing bundles without retranslating, use the **Clean AI reference fragments** section: point it at a `.po` file or directory, optionally recompile the `.mo`, and press the button to rewrite every entry that still contains the stray notes.

## Workflows

### 1. Catalog and Inventory

Use `xliff_tools.py` to confirm coverage and detect anomalies:

```bash
python xliff_tools.py catalog . --expect ar=283 cs=283 de=283 el=283 pl=284 uk=284
```

### 2. Validation

Run metadata and QA checks across the archive:

```bash
python xliff_tools.py validate . --lang-from auto --require-prolog
```

The validator enforces `source-language="en"`, checks `target-language` conventions, reports duplicate IDs, and flags empty/unchanged targets.

### 3. Generating XLIFFs (from Source)

Ingest English source files and emit XLIFFs for all locales:

```bash
python xliff_tools.py generate content/en build/xliff --copy-source --segment-by paragraph --job-start 2085
```

### 4. Translating Content (AI-Powered)

You can translate XLIFF files using OpenAI, Anthropic Claude, or local/cloud Ollama models via GUI or CLI.

**Using the GUI:**
Run `python translator_gui.py` to open the interface.
*   Select files to translate.
*   Choose a provider, press **Load Models**, and pick an available model.
*   Press **Test Connection** before translating to validate that model and provider.
*   Set options (e.g., skip pre-filled, overwrite).
*   Monitor progress and logs in real-time.
*   Outputs are saved to `translated/` with `-<lang>-translated.xliff` names unless overwrite is enabled.

**Windows executable (GUI):**
Build a standalone `.exe` using PyInstaller:
```powershell
.\build_windows_exe.ps1
```
The script creates/uses `.venv` in the repo. The outputs are `dist\CucumberDestroyer_TranslatorGUI.exe` and `dist\CucumberDestroyer_PotTranslatorGUI.exe`. Place a `.env` file next to the exe for hosted-provider keys, or run Ollama locally before choosing its provider.

**Linux executable (GUI):**
On Ubuntu/Debian, install Tk support once, then build standalone executables:
```bash
sudo apt install python3-tk python3-venv
chmod +x build_linux.sh
./build_linux.sh
```
The script creates/uses `.venv-linux` in the repo. The outputs are `dist/CucumberDestroyer_TranslatorGUI-linux-x86_64` and `dist/CucumberDestroyer_PotTranslatorGUI-linux-x86_64`. Put `.env` beside the executable for hosted-provider keys, or run Ollama locally before choosing its provider.

**Using the CLI:**
Translate a single file (or batch via scripts):
```bash
python translate_one.py --input originals/MyJob.xliff --languages ar,cs,de --provider openai --model gpt-4o
python translate_one.py --input originals/MyJob.xliff --languages el --provider anthropic --model claude-sonnet-4-20250514
python translate_one.py --input originals/MyJob.xliff --languages de --provider ollama --model llama3.2
```
Outputs are saved to `translated/` (flat) with `-<lang>-translated.xliff` suffixes unless you set `--output` or `--overwrite`.
Key features:
*   Preserves HTML tags and attributes.
*   Rate limiting support.
*   Language detection to skip already-translated segments.

### 5. Preparing for Import

After translation, organize the files for WPML import. This script matches translated files back to their originals and creates a flat import directory:

```bash
python prepare_wpml_import.py --originals originals --translated translated --output wpml-import
```

## XLIFF structure & Rules

*   **Format**: XLIFF 1.2 with `wpml` namespace extensions.
*   **Metadata**: `<file>` elements capture source/target languages and original paths.
*   **Structure**: `<trans-unit>` entries contain paired `<source>` and `<target>` segments.
*   **Normalization**:
    *   Preserve XML prologs.
    *   Keep `source-language` fixed to `en`.
    *   Retain `<trans-unit>` IDs.
    *   Maintain inline placeholders/tags as-is.
