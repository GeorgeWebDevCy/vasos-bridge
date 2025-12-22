#!/usr/bin/env python3
"""
XLIFF cataloging, validation, and generation utilities for the Bridge Project archive.

Subcommands:
* catalog  - summarize target-language counts and optionally check against expectations.
* validate - enforce XLIFF metadata rules and run QA checks on trans-units.
* generate - create per-language XLIFF files from English source documents.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SUPPORTED_LANGUAGES = ("ar", "cs", "de", "el", "pl", "uk")
DEFAULT_EXPECTED_COUNTS = {
    "ar": 283,
    "cs": 283,
    "de": 283,
    "el": 283,
    "pl": 284,
    "uk": 284,
}
XLING_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"
WPML_NAMESPACE = "https://cdn.wpml.org/xliff/custom-attributes.xsd"
XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/"


def _register_xliff_namespaces() -> None:
    ET.register_namespace("", XLING_NAMESPACE)
    ET.register_namespace("wpml", WPML_NAMESPACE)


_register_xliff_namespaces()


class ValidationIssue:
    def __init__(self, path: pathlib.Path, level: str, message: str):
        self.path = path
        self.level = level
        self.message = message

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.path}: {self.message}"


def _nsmap(root: ET.Element) -> Dict[str, str]:
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        return {"x": uri}
    return {}


def _iter_xliff_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    yield from root.rglob("*.xliff")


def _infer_expected_lang(path: pathlib.Path, strategy: str) -> Optional[str]:
    if strategy == "filename":
        match = re.search(r"-([a-z]{2})\\.xliff$", path.name)
        if match:
            return match.group(1)
    if strategy == "parent":
        candidate = path.parent.name
        if len(candidate) == 2:
            return candidate
    if strategy == "auto":
        for mode in ("filename", "parent"):
            lang = _infer_expected_lang(path, mode)
            if lang:
                return lang
        return None
    if strategy == "none":
        return None
    raise ValueError(f"Unknown lang inference strategy '{strategy}'")


def _read_file_metadata(path: pathlib.Path) -> Tuple[str, str, List[ET.Element], Dict[str, int]]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = _nsmap(root)
    file_el = root.find("x:file", ns) if ns else root.find("file")
    if file_el is None:
        raise ValueError("Missing <file> element")

    source_lang = file_el.attrib.get("source-language", "")
    target_lang = file_el.attrib.get("target-language", "")

    trans_units = list(file_el.findall(".//x:trans-unit", ns) if ns else file_el.findall(".//trans-unit"))
    id_counts: Dict[str, int] = defaultdict(int)
    for tu in trans_units:
        if "id" in tu.attrib:
            id_counts[tu.attrib["id"]] += 1
    return source_lang, target_lang, trans_units, id_counts


def _segment_source(text: str, mode: str) -> List[str]:
    if mode == "paragraph":
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if mode == "line":
        return [line.strip() for line in text.splitlines() if line.strip()]
    raise ValueError(f"Unknown segmentation mode '{mode}'")


def _build_trans_unit_id(rel_path: pathlib.Path, idx: int) -> str:
    normalized = rel_path.as_posix()
    return f"{normalized}#{idx + 1}"


def _write_tree(tree: ET.ElementTree, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def _emit_xliff(
    original_path: pathlib.Path,
    rel_path: pathlib.Path,
    segments: Sequence[str],
    target_language: str,
    output_path: pathlib.Path,
    copy_source: bool,
) -> None:
    xliff_el = ET.Element(f"{{{XLING_NAMESPACE}}}xliff", {"version": "1.2"})
    file_attrs = {
        "source-language": "en",
        "target-language": target_language,
        "datatype": "plaintext",
        "original": rel_path.as_posix(),
    }
    file_el = ET.SubElement(xliff_el, "file", file_attrs)
    body_el = ET.SubElement(file_el, "body")

    for idx, segment in enumerate(segments):
        tu_el = ET.SubElement(body_el, "trans-unit", {"id": _build_trans_unit_id(rel_path, idx)})
        source_el = ET.SubElement(tu_el, "source")
        source_el.text = segment
        target_el = ET.SubElement(tu_el, "target")
        if copy_source:
            target_el.text = segment
        else:
            target_el.text = ""

    tree = ET.ElementTree(xliff_el)
    _write_tree(tree, output_path)


def _normalize_wpml_prefixes(tree: ET.ElementTree) -> bool:
    """Return True when the tree was modified to use the wpml prefix."""

    root = tree.getroot()
    changed = False
    xmlns_prefix = f"{{{XMLNS_NAMESPACE}}}"

    for key in list(root.attrib):
        if not key.startswith(xmlns_prefix):
            continue
        prefix = key[len(xmlns_prefix) :]
        uri = root.attrib[key]
        if uri == WPML_NAMESPACE and prefix != "wpml":
            del root.attrib[key]
            changed = True

    if root.attrib.get(f"{xmlns_prefix}wpml") != WPML_NAMESPACE:
        root.attrib[f"{xmlns_prefix}wpml"] = WPML_NAMESPACE
        changed = True

    def _normalize_element(el: ET.Element) -> None:
        nonlocal changed
        if el.tag.startswith("{"):
            uri, local = el.tag[1:].split("}", 1)
            if uri == WPML_NAMESPACE:
                normalized_tag = f"{{{WPML_NAMESPACE}}}{local}"
                if normalized_tag != el.tag:
                    el.tag = normalized_tag
                    changed = True

        new_attrib: Dict[str, str] = {}
        for attr_key, attr_val in el.attrib.items():
            if attr_key.startswith("{"):
                uri, local = attr_key[1:].split("}", 1)
                if uri == WPML_NAMESPACE:
                    normalized_key = f"{{{WPML_NAMESPACE}}}{local}"
                    if normalized_key != attr_key:
                        changed = True
                    new_attrib[normalized_key] = attr_val
                else:
                    new_attrib[attr_key] = attr_val
            else:
                new_attrib[attr_key] = attr_val

        if new_attrib != el.attrib:
            el.attrib.clear()
            el.attrib.update(new_attrib)

        for child in el:
            _normalize_element(child)

    _normalize_element(root)
    return changed


def cmd_catalog(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root)
    files = sorted(_iter_xliff_files(root))
    counts = Counter()
    for path in files:
        source_lang, target_lang, _, _ = _read_file_metadata(path)
        if source_lang != "en":
            print(f"[WARN] {path} has unexpected source-language '{source_lang}'", file=sys.stderr)
        counts[target_lang] += 1

    print("Target-language totals:")
    for lang in sorted(counts):
        print(f"  {lang}: {counts[lang]}")
    if args.expect:
        expected = dict(DEFAULT_EXPECTED_COUNTS)
        expected.update(args.expect)
        mismatches = []
        for lang, exp_count in expected.items():
            actual = counts.get(lang, 0)
            if actual != exp_count:
                mismatches.append((lang, actual, exp_count))
        if mismatches:
            print("Count mismatches found:")
            for lang, actual, exp in mismatches:
                print(f"  {lang}: found {actual}, expected {exp}")
            return 1
        print("Counts match expected targets.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root)
    files = sorted(_iter_xliff_files(root))
    issues: List[ValidationIssue] = []

    for path in files:
        if args.require_prolog:
            with path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
            if not first_line.startswith("<?xml"):
                issues.append(ValidationIssue(path, "error", "Missing XML prolog"))

        try:
            source_lang, target_lang, trans_units, id_counts = _read_file_metadata(path)
        except Exception as exc:  # noqa: BLE001
            issues.append(ValidationIssue(path, "error", f"Parse failure: {exc}"))
            continue

        if source_lang != "en":
            issues.append(ValidationIssue(path, "error", f"source-language is '{source_lang}', expected 'en'"))

        inferred_lang = _infer_expected_lang(path, args.lang_from)
        if inferred_lang and target_lang != inferred_lang:
            issues.append(
                ValidationIssue(
                    path,
                    "error",
                    f"target-language '{target_lang}' does not match naming convention '{inferred_lang}'",
                )
            )

        if not trans_units:
            issues.append(ValidationIssue(path, "error", "No <trans-unit> elements present"))

        duplicate_ids = [tu_id for tu_id, count in id_counts.items() if count > 1]
        if duplicate_ids:
            issues.append(ValidationIssue(path, "error", f"Duplicate trans-unit ids: {', '.join(duplicate_ids)}"))

        for tu in trans_units:
            tu_id = tu.attrib.get("id", "<missing>")
            source_el = tu.find("source")
            target_el = tu.find("target")
            source_text = (source_el.text or "").strip() if source_el is not None else ""
            target_text = (target_el.text or "").strip() if target_el is not None else ""

            if not source_text:
                issues.append(ValidationIssue(path, "error", f"trans-unit {tu_id} has empty source"))
            if target_el is None:
                issues.append(ValidationIssue(path, "error", f"trans-unit {tu_id} missing target element"))
                continue
            if not target_text:
                issues.append(ValidationIssue(path, "warning", f"trans-unit {tu_id} has empty target"))
            if target_text == source_text:
                issues.append(ValidationIssue(path, "warning", f"trans-unit {tu_id} target matches source"))

            if target_lang == "ar" and target_text:
                if not re.search(r"[\\u0600-\\u06FF]", target_text):
                    issues.append(
                        ValidationIssue(path, "warning", f"Arabic target in trans-unit {tu_id} lacks RTL characters")
                    )

    errors = [iss for iss in issues if iss.level == "error"]
    warnings = [iss for iss in issues if iss.level == "warning"]

    for issue in issues:
        print(issue, file=sys.stderr if issue.level == "error" else sys.stdout)

    print(f"Validation completed with {len(errors)} error(s) and {len(warnings)} warning(s).")
    return 1 if errors else 0


def cmd_generate(args: argparse.Namespace) -> int:
    source_dir = pathlib.Path(args.source_dir)
    output_dir = pathlib.Path(args.output_dir)
    languages = tuple(args.languages.split(",")) if args.languages else SUPPORTED_LANGUAGES

    allowed_suffixes = tuple(f".{ext}" for ext in args.extensions.split(","))
    english_files = sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_suffixes
    )
    if not english_files:
        print(f"No source files found in {source_dir} with extensions {allowed_suffixes}", file=sys.stderr)
        return 1

    job_counter = itertools.count(args.job_start)
    for path in english_files:
        rel_path = path.relative_to(source_dir)
        segments = _segment_source(path.read_text(encoding="utf-8"), args.segment_by)
        if not segments:
            print(f"[SKIP] {rel_path} is empty after segmentation")
            continue
        for lang in languages:
            job_id = next(job_counter)
            output_path = output_dir / lang / f"{path.stem}-job-{job_id}.xliff"
            _emit_xliff(path, rel_path, segments, lang, output_path, args.copy_source)
    print(f"Generated XLIFF files in {output_dir} for languages: {', '.join(languages)}")
    return 0


def cmd_fix_prefixes(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root)
    files = sorted(_iter_xliff_files(root))
    modified: List[pathlib.Path] = []

    for path in files:
        tree = ET.parse(path)
        if _normalize_wpml_prefixes(tree):
            _write_tree(tree, path)
            modified.append(path)

    if modified:
        print("Updated WPML prefixes in:")
        for path in modified:
            print(f"  {path}")
    else:
        print("No files required WPML prefix fixes.")
    return 0


def _parse_expectations(expect_args: Sequence[str]) -> Dict[str, int]:
    expectations: Dict[str, int] = {}
    for pair in expect_args:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Invalid expectation '{pair}', expected <lang>=<count>")
        lang, value = pair.split("=", 1)
        expectations[lang] = int(value)
    return expectations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XLIFF cataloging, validation, and generation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/xliff_tools.py catalog .
              python scripts/xliff_tools.py validate . --lang-from auto --require-prolog
              python scripts/xliff_tools.py generate content/en build/xliff --copy-source
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="Summarize target-language counts")
    catalog_parser.add_argument("root", help="Directory to scan for XLIFF files")
    catalog_parser.add_argument(
        "--expect",
        type=_parse_expectations,
        nargs="*",
        help="Expectation overrides as <lang>=<count> (defaults are built-in)",
    )
    catalog_parser.set_defaults(func=cmd_catalog)

    validate_parser = subparsers.add_parser("validate", help="Validate XLIFF metadata and QA checks")
    validate_parser.add_argument("root", help="Directory to scan for XLIFF files")
    validate_parser.add_argument(
        "--lang-from",
        choices=["auto", "filename", "parent", "none"],
        default="auto",
        help="How to infer expected target language from file names or directories",
    )
    validate_parser.add_argument(
        "--require-prolog",
        action="store_true",
        help="Fail when the XML prolog is missing (normalization rule)",
    )
    validate_parser.set_defaults(func=cmd_validate)

    generate_parser = subparsers.add_parser("generate", help="Generate per-language XLIFF files from English content")
    generate_parser.add_argument("source_dir", help="Directory containing English source content")
    generate_parser.add_argument("output_dir", help="Destination directory for generated XLIFF files")
    generate_parser.add_argument(
        "--languages",
        help="Comma-separated target languages (default: ar,cs,de,el,pl,uk)",
    )
    generate_parser.add_argument(
        "--extensions",
        default="txt,md,html",
        help="Comma-separated list of source file extensions to ingest",
    )
    generate_parser.add_argument(
        "--segment-by",
        choices=["paragraph", "line"],
        default="paragraph",
        help="Segmentation strategy for turning source into trans-units",
    )
    generate_parser.add_argument(
        "--job-start",
        type=int,
        default=1,
        help="Starting job id counter used in generated file names",
    )
    generate_parser.add_argument(
        "--copy-source",
        action="store_true",
        help="Populate targets with source text to aid translation memory tools",
    )
    generate_parser.set_defaults(func=cmd_generate)

    fix_prefixes_parser = subparsers.add_parser(
        "fix-wpml-prefixes",
        help="Normalize WPML namespace prefixes to 'wpml' across XLIFF files",
    )
    fix_prefixes_parser.add_argument("root", help="Directory to scan for XLIFF files")
    fix_prefixes_parser.set_defaults(func=cmd_fix_prefixes)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
