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
