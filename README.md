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

For AI translation features, create a `.env` file in the root directory with your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

## Tooling Overview

The repository contains several utility scripts:

*   **`xliff_tools.py`**: The core utility for cataloging, validating, and generating XLIFF files.
*   **`translator_gui.py`**: A desktop GUI for translating files using OpenAI models (GPT-4, etc.).
*   **`translate_one.py`**: A command-line tool for translating single or multiple files via OpenAI.
*   **`prepare_wpml_import.py`**: A script to organize translated files for re-import into WPML.

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
