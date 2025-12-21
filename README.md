# Bridge Project translation archive

This repository stores XLIFF 1.2 exports for the Bridge Project website. Each file represents a single translation job sourced from English content and localized into one of six languages (Arabic, Czech, German, Greek, Polish, or Ukrainian).【F:The Bridge Project-translation-job-1000.xliff†L1-L2】

## File inventory

* Files are named `The Bridge Project-translation-job-<id>.xliff`, with sequential job IDs from 385 through 2084 (no gaps), for a total of 1,700 files.【F:The Bridge Project-translation-job-385.xliff†L1-L2】【F:The Bridge Project-translation-job-2084.xliff†L1-L2】
* Target-language coverage:

  | Language | Count |
  | --- | --- |
  | Arabic (`ar`) | 283 |
  | Czech (`cs`) | 283 |
  | German (`de`) | 283 |
  | Greek (`el`) | 283 |
  | Polish (`pl`) | 284 |
  | Ukrainian (`uk`) | 284 |

## XLIFF structure

* Every file begins with an XML prolog and a `<file>` element that captures source and target languages, site metadata (domain, sender identity), and WPML-specific attributes like word counts.【F:The Bridge Project-translation-job-1000.xliff†L1-L2】
* The `<header>` section includes processing phases plus a `<reference>` that links back to the originating Bridge Project page, preserving traceability to the source content.【F:The Bridge Project-translation-job-385.xliff†L1-L2】
* Text content appears in `<trans-unit>` entries under `<body>`, where each unit stores paired `<source>` and `<target>` segments for titles, bodies, or taxonomy labels.【F:The Bridge Project-translation-job-1000.xliff†L2-L5】

## Working with the archive

* Filter by language with globbing (for example, `*target-language="cs"*`) or by job ID using the numeric suffix in the filename.
* Because every job ID is present in the 385–2084 range, you can map site content to translation files directly via that identifier.

## Verified inventory and counts

Use the helper script to confirm coverage and detect anomalies:

```bash
python scripts/xliff_tools.py catalog . --expect ar=283 cs=283 de=283 el=283 pl=284 uk=284
```

The current archive matches expectations (ar/cs/de/el: 283 each; pl/uk: 284 each), with 1,700 files overall.【F:scripts/xliff_tools.py†L93-L127】

## Source-to-target normalization rules

* Preserve the XML prolog; files missing it will fail validation when `--require-prolog` is set.
* Keep `source-language` fixed to `en` and ensure `target-language` reflects the locale implied by file naming (either a `-<lang>.xliff` suffix or a parent directory named after the language code).
* Retain `<trans-unit>` IDs; duplicates are reported as errors, and IDs are derived from the source path plus an ordinal in generated files to keep them stable across languages.【F:scripts/xliff_tools.py†L129-L197】
* Maintain inline placeholders/tags as-is within segments; unchanged or empty targets are flagged for QA review rather than silently accepted.

## Validation workflow

Run metadata and QA checks across the archive:

```bash
python scripts/xliff_tools.py validate . --lang-from auto --require-prolog
```

The validator enforces `source-language="en"`, checks `target-language` against naming conventions, reports duplicate `<trans-unit>` IDs, flags empty or unchanged targets, and warns when Arabic targets lack RTL characters.【F:scripts/xliff_tools.py†L129-L197】

## Translation pipeline (English ➜ per-language XLIFF)

`scripts/xliff_tools.py generate` ingests English source files and emits XLIFFs for all six locales in one batch. Example:

```bash
python scripts/xliff_tools.py generate content/en build/xliff --copy-source --segment-by paragraph --job-start 2085
```

Defaults:

* Languages: `ar, cs, de, el, pl, uk` (override with `--languages`).
* Source file extensions: `.txt, .md, .html` (override with `--extensions`).
* Segmentation: paragraphs split by blank lines (switch to line-based with `--segment-by line`).
* Targets can start empty, or prefilled with source text when `--copy-source` is set to aid CAT/TM tools.【F:scripts/xliff_tools.py†L199-L287】

Generated files are placed under `build/xliff/<lang>/` using the pattern `<source-stem>-job-<id>.xliff`, keeping `<trans-unit>` IDs aligned across locales and embedding `original` paths for traceability.【F:scripts/xliff_tools.py†L171-L232】

## QA and schema checks

* **Schema/structure:** XML parsing plus presence of `<file>` and `<trans-unit>` elements.
* **Segment quality:** Empty or unchanged targets are surfaced as warnings for follow-up.
* **Typography:** RTL hints—Arabic targets with no Arabic script characters are warned to encourage explicit RTL handling.
