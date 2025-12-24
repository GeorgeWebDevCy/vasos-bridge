#!/usr/bin/env python3
"""
Build a WPML import folder by matching originals to translated outputs.

For each XLIFF in originals/, this script finds a translated file under
translated/ with the same WPML job id and target-language, then copies it
into wpml-import/ (flat, no subfolders).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _nsmap(root: ET.Element) -> Dict[str, str]:
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        return {"x": uri}
    return {}


def _read_metadata(path: Path) -> Tuple[str, str]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = _nsmap(root)
    file_el = root.find("x:file", ns) if ns else root.find("file")
    if file_el is None:
        raise ValueError("Missing <file> element")
    original_id = file_el.attrib.get("original", "").strip()
    target_lang = file_el.attrib.get("target-language", "").strip()
    if not original_id or not target_lang:
        raise ValueError("Missing original or target-language attribute")
    return original_id, target_lang


def _iter_xliff_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.xliff")


def _build_translated_index(
    translated_root: Path,
    exclude_root: Optional[Path],
) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[Path, str]]]:
    index: Dict[Tuple[str, str], Path] = {}
    warnings: List[Tuple[Path, str]] = []

    for path in _iter_xliff_files(translated_root):
        if exclude_root and path.is_relative_to(exclude_root):
            continue
        try:
            original_id, target_lang = _read_metadata(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append((path, f"skip: {exc}"))
            continue
        key = (original_id, target_lang)
        if key in index:
            warnings.append((path, f"duplicate for job {original_id} lang {target_lang}"))
            continue
        index[key] = path

    return index, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a wpml-import folder from originals/ and translated/ outputs.",
    )
    parser.add_argument(
        "--originals",
        default="originals",
        help="Folder containing original WPML XLIFF jobs (default: originals).",
    )
    parser.add_argument(
        "--translated",
        default="translated",
        help="Folder containing translated XLIFF outputs (default: translated).",
    )
    parser.add_argument(
        "--output",
        default="wpml-import",
        help="Folder to write import-ready XLIFFs (default: wpml-import).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copy when the output file already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any originals are missing translations.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    originals_root = Path(args.originals).resolve()
    translated_root = Path(args.translated).resolve()
    output_root = Path(args.output).resolve()

    if not originals_root.exists():
        print(f"Originals folder not found: {originals_root}", file=sys.stderr)
        return 1
    if not translated_root.exists():
        print(f"Translated folder not found: {translated_root}", file=sys.stderr)
        return 1

    index, warnings = _build_translated_index(translated_root, output_root)
    for path, message in warnings:
        print(f"[warn] {path}: {message}", file=sys.stderr)

    output_root.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing = 0
    skipped = 0

    for original_path in _iter_xliff_files(originals_root):
        try:
            original_id, target_lang = _read_metadata(original_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {original_path}: skip: {exc}", file=sys.stderr)
            continue
        key = (original_id, target_lang)
        translated_path = index.get(key)
        if translated_path is None:
            print(f"[missing] {original_path}: no translated match for {target_lang}", file=sys.stderr)
            missing += 1
            continue
        dest_path = output_root / translated_path.name
        if dest_path.exists() and args.skip_existing:
            skipped += 1
            continue
        shutil.copy2(translated_path, dest_path)
        matched += 1

    print(f"Prepared {matched} file(s) in {output_root}.")
    if skipped:
        print(f"Skipped {skipped} existing file(s).")
    if missing:
        print(f"Missing {missing} translation(s).", file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
