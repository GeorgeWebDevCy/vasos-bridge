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
pip install openai langdetect
```

### Environment Variables

For AI translation features, create a `.env` file in the repo root with your OpenAI API key. For packaged Windows `.exe` builds, place the same `.env` next to the executable:

```env
OPENAI_API_KEY=sk-...
```

## Tooling Overview

The repository contains several utility scripts:

*   **`xliff_tools.py`**: The core utility for cataloging, validating, and generating XLIFF files.
*   **`translator_gui.py`**: A desktop GUI for translating files using OpenAI models (GPT-4, etc.).
*   **`translate_one.py`**: A command-line tool for translating single or multiple files via OpenAI.
*   **`prepare_wpml_import.py`**: A script to organize translated files for re-import into WPML.
 *   **`pot_translator_gui.py`**: Tkinter GUI for translating `.pot` templates via OpenAI with live progress, compile controls, plural overrides, and dry-run support.
 *   **`translate_pot.py`**: Translate gettext `.pot` templates into per-language `.po` files and optionally compile `.mo` binaries.

### Translating gettext templates

#### CLI

To generate `.po`/`.mo` bundles from a gettext `.pot` template, run:

```bash
python translate_pot.py \
  --input wp-xliff-translator/includes/plugin-update-checker/languages/plugin-update-checker.pot \
  --languages ar,cs,de \
  --compile
```

You can provide a fallback context for entries that lack `msgctxt` by using `--default-context`, exporting `GPT_TRANSLATOR_CONTEXT`, or adding `default_context = "..."` under `[tool.gpt-po-translator]` in `pyproject.toml`. The CLI option overrides the environment/configured values. Translations are also tagged with `#. AI-generated` by default; pass `--no-ai-comment` when you want to skip that comment.

The CLI writes `po/<locale>/<domain>-<locale>.po` and, when `--compile` is used, companion `.mo` files. It resolves the proper locale suffix for each language (e.g., translating `masterstudy-lms-learning-management-system.pot` to Greek produces `masterstudy-lms-learning-management-system-el_GR.po`), uses the same `OPENAI_API_KEY` as the other tools, throttles requests via `--rpm`, and preserves comments/context while translating placeholders. Use `--max-entries` to limit how many strings are translated per language, `--plural-forms` to override locale plural rules (`lang=expr`), and `--dry-run` to emit header-only `.po` files without calling OpenAI.

#### GUI

Alternatively, run `python pot_translator_gui.py` to launch a desktop interface. Select one or more `.pot` files, enter a comma-separated list of target languages, choose an output directory (defaults to `po`), optionally enable `.mo` compilation and dry runs, and set plural-form overrides in `lang=expr` form. The GUI also exposes the translation model, requests-per-minute throttle, and per-language entry limits before kicking off the OpenAI batch. Log output and progress appear in real time, and generated `.po`/`.mo` bundles follow the `po/<locale>/<domain>-<locale>.(po|mo)` pattern so the resolved locale (for example, `el_GR`) appears in every filename.

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

You can translate XLIFF files using OpenAI's models via GUI or CLI.

**Using the GUI:**
Run `python translator_gui.py` to open the interface.
*   Select files to translate.
*   Choose target languages.
*   Set options (e.g., skip pre-filled, overwrite).
*   Monitor progress and logs in real-time.
*   Outputs are saved to `translated/` with `-<lang>-translated.xliff` names unless overwrite is enabled.

**Windows executable (GUI):**
Build a standalone `.exe` using PyInstaller:
```powershell
.\build_windows_exe.ps1
```
The script creates/uses `.venv` in the repo. The outputs are `dist\CucumberDestroyer_TranslatorGUI.exe` and `dist\CucumberDestroyer_PotTranslatorGUI.exe`. Place a `.env` file next to the exe you run (or set `OPENAI_API_KEY`) before launching it.

**Using the CLI:**
Translate a single file (or batch via scripts):
```bash
python translate_one.py --input originals/MyJob.xliff --languages ar,cs,de --model gpt-4o
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
