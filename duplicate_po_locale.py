#!/usr/bin/env python3
"""
Duplicate an existing gettext .po file to another locale and optionally compile a .mo.

This utility reuses the parsing/compilation helpers from translate_pot.py so the duplicated
bundle matches the same serialization rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from translate_pot import (
    PoEntry,
    _build_catalog,
    _metadata_to_text,
    _parse_metadata_lines,
    _parse_po,
    _plural_setting,
    _parse_plural_overrides,
    _update_metadata,
    _write_mo,
    _write_po,
)


def collect_po_paths(path_arg: str) -> List[Path]:
    path = Path(path_arg)
    if path.is_dir():
        return sorted(path.glob("*.po"))
    if path.is_file():
        return [path]
    return []


def _normalize_locale_input(value: str) -> str:
    cleaned = value.replace("-", "_").strip()
    if not cleaned:
        return ""
    parts = [part for part in cleaned.split("_") if part]
    base = parts[0].lower()
    if len(parts) > 1:
        region = "_".join(part.upper() for part in parts[1:])
        return f"{base}_{region}"
    return base


def _metadata_language(metadata: List[Tuple[str, str]]) -> str:
    for key, value in metadata:
        if key.lower() == "language":
            return _normalize_locale_input(value)
    return ""


def _domain_for_target(stem: str, existing_locale: str) -> str:
    if existing_locale:
        suffix = f"-{existing_locale}"
        if stem.endswith(suffix):
            domain = stem[: -len(suffix)]
            return domain or stem
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return stem


@dataclass
class DuplicateResult:
    source: Path
    target_locale: str
    po_path: Path
    mo_path: Optional[Path]


def _duplicate_po(
    source: Path,
    target_locale: str,
    output_root: Path,
    compile_mo: bool,
    plural_overrides: Dict[str, str],
) -> Tuple[Path, Optional[Path]]:
    locale = _normalize_locale_input(target_locale)
    if not locale:
        raise ValueError(f"Invalid locale: {target_locale}")

    header, entries = _parse_po(source)
    if header is None:
        header = PoEntry()

    metadata = _parse_metadata_lines(header.msgstr or "")
    existing_locale = _metadata_language(metadata)
    domain = _domain_for_target(source.stem, existing_locale)
    if not domain:
        domain = locale

    _update_metadata(metadata, "Language", locale)
    plural_expr, _ = _plural_setting(locale, plural_overrides)
    _update_metadata(metadata, "Plural-Forms", plural_expr)
    header.msgstr = _metadata_to_text(metadata)

    dest_root = Path(output_root)
    dest_dir = dest_root / locale
    dest_stem = f"{domain}-{locale}"
    dest_path = dest_dir / f"{dest_stem}.po"
    _write_po(dest_path, header, entries, metadata)

    mo_path: Optional[Path] = None
    if compile_mo:
        catalog = _build_catalog(entries, header)
        mo_path = dest_path.with_suffix(".mo")
        _write_mo(catalog, mo_path)

    return dest_path, mo_path


def _parse_targets(values: Optional[str]) -> List[str]:
    if not values:
        return []
    return [token.strip() for token in values.split(",") if token.strip()]


def duplicate_po_locales(
    sources: Sequence[Path],
    targets: Sequence[str],
    output_root: Path,
    compile_mo: bool,
    plural_overrides: Dict[str, str],
) -> List[DuplicateResult]:
    normalized_targets: List[str] = []
    for target in targets:
        locale = _normalize_locale_input(target)
        if not locale:
            raise ValueError(f"Invalid target locale: {target}")
        normalized_targets.append(locale)

    results: List[DuplicateResult] = []
    for source in sources:
        for locale in normalized_targets:
            po_path, mo_path = _duplicate_po(
                source=source,
                target_locale=locale,
                output_root=output_root,
                compile_mo=compile_mo,
                plural_overrides=plural_overrides,
            )
            results.append(
                DuplicateResult(source=source, target_locale=locale, po_path=po_path, mo_path=mo_path)
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duplicate an existing .po bundle to another locale and optionally compile the .mo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python duplicate_po_locale.py --source po/el_GR --targets el --compile\n"
            "  python duplicate_po_locale.py -s po/el_GR/masterstudy-lms-learning-management-system-pro-el_GR.po "
            "--targets el,el_GR"
        ),
    )
    parser.add_argument("--source", "-s", required=True, help="Path to a .po file or a directory of .po files.")
    parser.add_argument(
        "--targets",
        "-t",
        required=True,
        help="Comma-separated list of target locales (e.g., el,el_GR).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="po",
        help="Root directory where duplicated locales are written (default: po).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile a .mo file alongside each duplicated .po.",
    )
    parser.add_argument(
        "--plural-forms",
        action="append",
        help="Override Plural-Forms for a locale (lang=expression).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    sources = collect_po_paths(args.source)
    if not sources:
        print(f"No .po sources found at {args.source}", file=sys.stderr)
        return 1

    targets = _parse_targets(args.targets)
    if not targets:
        print("No target locales provided.", file=sys.stderr)
        return 1

    plural_overrides = _parse_plural_overrides(args.plural_forms)
    output_root = Path(args.output_dir)

    missing = [src for src in sources if not src.exists()]
    if missing:
        print(f"Source not found: {missing[0]}", file=sys.stderr)
        return 1

    try:
        results = duplicate_po_locales(
            sources=sources,
            targets=targets,
            output_root=output_root,
            compile_mo=args.compile,
            plural_overrides=plural_overrides,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for res in results:
        print(f"[{res.target_locale}] Wrote {res.po_path}")
        if args.compile and res.mo_path:
            print(f"[{res.target_locale}] Compiled {res.mo_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
