#!/usr/bin/env python3
"""
Translate one or more XLIFF files using OpenAI and write translated copies.

Defaults to test jobs 771-776 in originals/ when --input is not provided.
By default, outputs are written under translated/ (flat).
Use --languages to translate the same input into multiple languages.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


XLING_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"
WPML_NAMESPACE = "https://cdn.wpml.org/xliff/custom-attributes.xsd"
DEFAULT_LANGUAGES = ("ar", "cs", "de", "el", "pl", "uk")
DEFAULT_TEST_JOB_IDS = (771, 772, 773, 774, 775, 776)


def _register_xliff_namespaces() -> None:
    ET.register_namespace("", XLING_NAMESPACE)
    ET.register_namespace("wpml", WPML_NAMESPACE)


_register_xliff_namespaces()


def _nsmap(root: ET.Element) -> Dict[str, str]:
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0][1:]
        return {"x": uri}
    return {}


def _iter_trans_units(file_el: ET.Element, ns: Dict[str, str]) -> Iterable[ET.Element]:
    return file_el.findall(".//x:trans-unit", ns) if ns else file_el.findall(".//trans-unit")


def _ensure_target(tu_el: ET.Element, ns: Dict[str, str]) -> ET.Element:
    target_el = tu_el.find("x:target", ns) if ns else tu_el.find("target")
    if target_el is None:
        if ns and "x" in ns:
            target_el = ET.SubElement(tu_el, f"{{{ns['x']}}}target")
        else:
            target_el = ET.SubElement(tu_el, "target")
    return target_el


def _load_xliff(path: Path) -> Tuple[ET.ElementTree, ET.Element, Dict[str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = _nsmap(root)
    file_el = root.find("x:file", ns) if ns else root.find("file")
    if file_el is None:
        raise ValueError("Missing <file> element")
    return tree, file_el, ns


def _load_dotenv_key(var_name: str, dotenv_path: Path) -> Optional[str]:
    if not dotenv_path.exists():
        return None
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == var_name:
            return value.strip().strip('"').strip("'")
    return None


def _write_xliff(tree: ET.ElementTree, output_path: Path) -> None:
    _register_xliff_namespaces()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def _translate_text(client: "OpenAI", model: str, text: str, target_lang: str) -> str:
    system_prompt = (
        "You are a translation engine. Translate the user text to the target language while preserving all HTML tags, "
        "attributes, placeholders, and URLs. Return only the translated text. Do not add explanations or quotes."
    )
    user_prompt = f"Target language: {target_lang}\n\nText:\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _pick_default_inputs() -> List[Path]:
    originals_dir = Path("originals")
    preferred = [
        originals_dir / f"The Bridge Project-translation-job-{job_id}.xliff"
        for job_id in DEFAULT_TEST_JOB_IDS
    ]
    existing = [path for path in preferred if path.exists()]
    if existing:
        return existing
    xliffs = sorted(originals_dir.glob("*.xliff"))
    if not xliffs:
        raise FileNotFoundError(f"No .xliff files found under {originals_dir}")
    return [xliffs[0]]


def _default_output_path(
    input_path: Path,
    target_lang: str,
    output_root: Optional[Path] = None,
) -> Path:
    output_root = output_root or (Path.cwd() / "translated")
    lang_suffix = target_lang or "unknown"
    output_name = f"{input_path.stem}-{lang_suffix}-translated{input_path.suffix}"
    return output_root / output_name


def _parse_languages(lang_arg: str) -> List[str]:
    normalized = lang_arg.strip().lower()
    if normalized in ("all", "default"):
        return list(DEFAULT_LANGUAGES)
    parts = [part.strip().lower() for part in lang_arg.split(",") if part.strip()]
    if not parts:
        raise ValueError("No languages specified.")
    seen = set()
    ordered: List[str] = []
    for lang in parts:
        if lang in seen:
            continue
        seen.add(lang)
        ordered.append(lang)
    return ordered


def _ensure_api_key() -> None:
    # Prefer the repo-local .env to avoid stale global keys.
    dotenv_val = _load_dotenv_key("OPENAI_API_KEY", Path.cwd() / ".env")
    if dotenv_val:
        os.environ["OPENAI_API_KEY"] = dotenv_val
        return
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError("Missing OPENAI_API_KEY (set it in your environment or .env).")


def translate_one(
    input_path: Path,
    output_path: Optional[Path],
    target_lang: str,
    model: str,
    rpm: int,
    max_units: int,
    overwrite: bool,
    output_root: Optional[Path] = None,
) -> Tuple[int, int, Path]:
    tree, file_el, ns = _load_xliff(input_path)
    if target_lang:
        file_el.attrib["target-language"] = target_lang
    else:
        target_lang = file_el.attrib.get("target-language", "").strip()
        if not target_lang:
            raise ValueError("No target-language found in file; provide --lang.")
    if output_path is None:
        output_path = _default_output_path(input_path, target_lang, output_root)

    client = OpenAI()
    min_delay = 60.0 / float(max(1, rpm))
    last_call = 0.0
    translated_count = 0
    changed_count = 0

    for tu in _iter_trans_units(file_el, ns):
        if max_units and translated_count >= max_units:
            break
        source_el = tu.find("x:source", ns) if ns else tu.find("source")
        target_el = _ensure_target(tu, ns)
        source_text = (source_el.text or "").strip() if source_el is not None else ""
        if not source_text:
            continue

        now = time.time()
        wait_for = min_delay - (now - last_call)
        if wait_for > 0:
            time.sleep(wait_for)
        call_started = time.time()
        translated = _translate_text(client, model, source_text, target_lang)
        last_call = call_started

        target_el.text = translated
        translated_count += 1
        if translated.strip() and translated.strip() != source_text:
            changed_count += 1

    if translated_count == 0:
        raise RuntimeError("No trans-units were translated (empty sources or limit too low).")

    if overwrite:
        output_path = input_path

    _write_xliff(tree, output_path)
    return translated_count, changed_count, output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate one or more XLIFF files using OpenAI.")
    parser.add_argument(
        "--input",
        "-i",
        help=(
            "Path to a .xliff file (defaults to jobs 771-776 in originals/ when present, "
            "otherwise the first file in originals/)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (defaults to translated/<input-stem>-<lang>-translated.xliff).",
    )
    parser.add_argument("--lang", help="Override the target language in the XLIFF file.")
    parser.add_argument(
        "--languages",
        help=(
            "Comma-separated languages to translate in one run (e.g., ar,cs,de,el,pl,uk). "
            "Use 'all' for the default language set."
        ),
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use.")
    parser.add_argument("--rpm", type=int, default=120, help="Requests per minute throttle.")
    parser.add_argument(
        "--max-units",
        type=int,
        default=0,
        help="Limit the number of trans-units to translate (0 = all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of writing under translated/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if OpenAI is None:
        print("Missing dependency: install openai (pip install openai).", file=sys.stderr)
        return 1

    try:
        _ensure_api_key()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    input_paths = [Path(args.input)] if args.input else _pick_default_inputs()
    missing_inputs = [path for path in input_paths if not path.exists()]
    if missing_inputs:
        print(f"Input file not found: {missing_inputs[0]}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None
    languages: List[str] = []
    if args.languages:
        try:
            languages = _parse_languages(args.languages)
        except ValueError as exc:
            print(f"Invalid --languages value: {exc}", file=sys.stderr)
            return 1
    if args.lang and languages:
        print("Use only one of --lang or --languages.", file=sys.stderr)
        return 1
    if args.overwrite and languages:
        print("Cannot use --overwrite when translating multiple languages.", file=sys.stderr)
        return 1
    output_root: Optional[Path] = None
    if languages and output_path is not None:
        if output_path.suffix.lower() == ".xliff":
            print("When using --languages, --output must be a directory path.", file=sys.stderr)
            return 1
        output_root = output_path
        output_path = None
    if not languages and len(input_paths) > 1 and output_path is not None:
        if output_path.suffix.lower() == ".xliff":
            print("When translating multiple inputs, --output must be a directory path.", file=sys.stderr)
            return 1
        output_root = output_path
        output_path = None

    if languages:
        for input_path in input_paths:
            for lang in languages:
                try:
                    translated_count, changed_count, out_path = translate_one(
                        input_path=input_path,
                        output_path=None,
                        target_lang=lang,
                        model=args.model,
                        rpm=args.rpm,
                        max_units=args.max_units,
                        overwrite=False,
                        output_root=output_root,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[{lang}] Translation failed for {input_path}: {exc}", file=sys.stderr)
                    return 1
                print(f"[{lang}] Translated {translated_count} trans-unit(s) in {input_path}.")
                if changed_count == 0:
                    print(
                        f"[{lang}] Warning: no targets changed compared to the source text.",
                        file=sys.stderr,
                    )
                print(f"[{lang}] Saved output to {out_path}.")
        return 0

    for input_path in input_paths:
        try:
            translated_count, changed_count, resolved_output_path = translate_one(
                input_path=input_path,
                output_path=output_path,
                target_lang=args.lang or "",
                model=args.model,
                rpm=args.rpm,
                max_units=args.max_units,
                overwrite=args.overwrite,
                output_root=output_root,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Translation failed for {input_path}: {exc}", file=sys.stderr)
            return 1

        print(f"Translated {translated_count} trans-unit(s) in {input_path}.")
        if changed_count == 0:
            print("Warning: no targets changed compared to the source text.", file=sys.stderr)
        print(f"Saved output to {resolved_output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
