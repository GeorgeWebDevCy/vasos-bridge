#!/usr/bin/env python3
"""
Translate gettext .pot templates via OpenAI, write per-language .po files, and optionally compile .mo files.

This script keeps the new feature isolated from the existing XLIFF tooling so the working software is unaffected.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime
import json
import os
import re
import struct
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None

DEFAULT_LANGUAGES = ("ar", "cs", "de", "el", "pl", "uk")
DEFAULT_OUTPUT_ROOT = Path("po")

SINGULAR_SYSTEM_PROMPT = (
    "You are a translation engine. Translate the user text into the requested target language while preserving "
    "placeholders (e.g. %s, %d, %1$s), format tokens, and punctuation. Return only the translated text without "
    "explanations."
)
PLURAL_SYSTEM_PROMPT = (
    "You are a translation engine. Translate the provided singular/plural pair into the target language while "
    "preserving placeholders and format tokens. Respond with a JSON object containing the key \"forms\", where "
    "the value is a list of translations ordered from plural index 0 upwards."
)

PLURAL_FORMS_map = {
    "ar": "nplurals=6; plural=(n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : (n%100>=3 && n%100<=10) ? 3 : "
    "(n%100>=11 && n%100<=99) ? 4 : 5);",
    "cs": "nplurals=3; plural=(n==1 ? 0 : (n>=2 && n<=4) ? 1 : 2);",
    "de": "nplurals=2; plural=(n != 1);",
    "el": "nplurals=2; plural=(n != 1);",
    "pl": "nplurals=3; plural=(n==1 ? 0 : (n%10>=2 && n%10<=4 && (n%100<12 || n%100>14)) ? 1 : 2);",
    "uk": "nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : "
    "(n%10>=2 && n%10<=4 && (n%100<12 || n%100>14) ? 1 : 2));",
}

PLURAL_COUNTS = {lang: int(re.search(r"nplurals\s*=\s*(\d+)", expr).group(1)) for lang, expr in PLURAL_FORMS_map.items()}
DEFAULT_PLURAL_FORMS = "nplurals=2; plural=(n != 1);"
DEFAULT_NPLURALS = 2
LANGUAGE_LOCALE_DEFAULTS: Dict[str, str] = {
    "ar": "ar",
    "cs": "cs_CZ",
    "de": "de_DE",
    "el": "el_GR",
    "pl": "pl_PL",
    "uk": "uk",
}


@dataclass
class PoEntry:
    msgctxt: Optional[str] = None
    msgid: str = ""
    msgid_plural: Optional[str] = None
    msgstr: str = ""
    msgstr_plural: Dict[int, str] = field(default_factory=dict)
    raw_comments: List[str] = field(default_factory=list)
    translator_comments: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    previous: List[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    language: str
    locale: str
    pot_path: Path
    po_path: Path
    mo_path: Optional[Path]
    translated_count: int
    changed_count: int


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


def _ensure_api_key() -> None:
    dotenv_val = _load_dotenv_key("OPENAI_API_KEY", Path.cwd() / ".env")
    if dotenv_val:
        os.environ["OPENAI_API_KEY"] = dotenv_val
        return
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError("Missing OPENAI_API_KEY (set it in your environment or .env).")


def _quote_po_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_po_literal(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if not token.startswith('"'):
        return ""
    try:
        return ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return ""


def _parse_po(path: Path) -> Tuple[Optional[PoEntry], List[PoEntry]]:
    entries: List[PoEntry] = []
    current = PoEntry()
    last_field: Optional[Tuple[str, Optional[int]]] = None

    def _flush() -> None:
        nonlocal current, last_field
        if current.msgid or current.msgstr or current.raw_comments:
            entries.append(current)
            current = PoEntry()
        last_field = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                _flush()
                continue
            if line.startswith("#"):
                current.raw_comments.append(line)
                marker = line[1:2]
                body = line[2:].lstrip()
                if marker == ":":
                    current.references.append(body)
                elif marker == ".":
                    current.translator_comments.append(body)
                elif marker == ",":
                    current.flags.append(body)
                elif marker == "|":
                    current.previous.append(body)
                last_field = None
                continue
            if stripped.startswith("msgctxt"):
                value = stripped[len("msgctxt"):].strip()
                current.msgctxt = _parse_po_literal(value)
                last_field = ("msgctxt", None)
                continue
            if stripped.startswith("msgid_plural"):
                value = stripped[len("msgid_plural"):].strip()
                current.msgid_plural = _parse_po_literal(value)
                last_field = ("msgid_plural", None)
                continue
            if stripped.startswith("msgid"):
                value = stripped[len("msgid"):].strip()
                current.msgid = _parse_po_literal(value)
                last_field = ("msgid", None)
                continue
            if match := re.match(r"msgstr\[(\d+)\]", stripped):
                index = int(match.group(1))
                remainder = stripped[match.end():].strip()
                chunk = _parse_po_literal(remainder)
                current.msgstr_plural[index] = chunk
                last_field = ("msgstr_plural", index)
                continue
            if stripped.startswith("msgstr"):
                value = stripped[len("msgstr"):].strip()
                current.msgstr = _parse_po_literal(value)
                last_field = ("msgstr", None)
                continue
            if stripped.startswith('"') and last_field:
                chunk = _parse_po_literal(stripped)
                field, idx = last_field
                if field == "msgctxt":
                    current.msgctxt = (current.msgctxt or "") + chunk
                elif field == "msgid":
                    current.msgid += chunk
                elif field == "msgid_plural":
                    current.msgid_plural = (current.msgid_plural or "") + chunk
                elif field == "msgstr":
                    current.msgstr += chunk
                elif field == "msgstr_plural" and idx is not None:
                    current.msgstr_plural[idx] = (current.msgstr_plural.get(idx, "")) + chunk
                continue
    _flush()

    header: Optional[PoEntry] = None
    if entries and entries[0].msgid == "":
        header = entries.pop(0)
    return header, entries


def _serialize_header_lines(metadata: List[Tuple[str, str]]) -> List[str]:
    return [f"{key}: {value}" for key, value in metadata]


def _metadata_to_text(metadata: List[Tuple[str, str]]) -> str:
    if not metadata:
        return ""
    return "\n".join(f"{key}: {value}" for key, value in metadata) + "\n"


def _parse_metadata_lines(text: str) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            lines.append((key.strip(), value.strip()))
        else:
            lines.append((stripped, ""))
    return lines


def _update_metadata(metadata: List[Tuple[str, str]], key: str, value: str) -> None:
    lowered = key.lower()
    for idx, (existing_key, _) in enumerate(metadata):
        if existing_key.lower() == lowered:
            metadata[idx] = (existing_key, value)
            return
    metadata.append((key, value))


def _standardize_language(lang: str) -> str:
    return lang.lower()


def _resolve_locale(lang: str) -> str:
    cleaned = lang.replace("-", "_").strip()
    if not cleaned:
        return cleaned
    parts = [part for part in cleaned.split("_") if part]
    if not parts:
        return ""
    base = parts[0].lower()
    if len(parts) > 1:
        region = "_".join(part.upper() for part in parts[1:])
        return f"{base}_{region}"
    return LANGUAGE_LOCALE_DEFAULTS.get(base, base)


def _plural_setting(lang: str, overrides: Dict[str, str]) -> Tuple[str, int]:
    normalized = _standardize_language(lang)
    candidates = [normalized]
    base = normalized.split("_")[0]
    if base not in candidates:
        candidates.append(base)
    for candidate in candidates:
        if candidate in overrides:
            expr = overrides[candidate]
            match = re.search(r"nplurals\s*=\s*(\d+)", expr)
            return expr, int(match.group(1)) if match else DEFAULT_NPLURALS
    for candidate in candidates:
        if candidate in PLURAL_FORMS_map:
            return PLURAL_FORMS_map[candidate], PLURAL_COUNTS[candidate]
    return DEFAULT_PLURAL_FORMS, DEFAULT_NPLURALS


class OpenAITranslator:
    def __init__(self, model: str, rpm: int):
        self.model = model
        self.client = OpenAI()
        self.min_delay = 60.0 / max(1, rpm)
        self.last_call = 0.0

    def _throttle(self) -> None:
        now = time.time()
        wait_for = self.min_delay - (now - self.last_call)
        if wait_for > 0:
            time.sleep(wait_for)
        self.last_call = time.time()

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        self._throttle()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def translate_singular(
        self,
        entry: PoEntry,
        target_lang: str,
    ) -> str:
        components = [
            f"Target language: {target_lang}",
            "Text:",
            entry.msgid,
        ]
        if entry.msgctxt:
            components.extend(["Context:", entry.msgctxt])
        if entry.references:
            components.extend(["References:", "; ".join(entry.references)])
        if entry.translator_comments:
            components.extend(["Notes:", " ".join(entry.translator_comments)])
        user_prompt = "\n\n".join(components)
        return self._call(SINGULAR_SYSTEM_PROMPT, user_prompt)

    def translate_plural(
        self,
        entry: PoEntry,
        target_lang: str,
        nplurals: int,
    ) -> List[str]:
        components = [
            f"Target language: {target_lang}",
            "Singular text:",
            entry.msgid,
            "Plural text:",
            entry.msgid_plural or "",
            f"Expect {nplurals} plural form(s).",
        ]
        if entry.msgctxt:
            components.extend(["Context:", entry.msgctxt])
        if entry.references:
            components.extend(["References:", "; ".join(entry.references)])
        if entry.translator_comments:
            components.extend(["Notes:", " ".join(entry.translator_comments)])
        user_prompt = "\n\n".join(components)
        payload = self._call(PLURAL_SYSTEM_PROMPT, user_prompt)
        return _parse_plural_response(payload, nplurals)


def _parse_plural_response(payload: str, nplurals: int) -> List[str]:
    forms: List[str] = []
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        decoded = {}
    if isinstance(decoded, dict):
        raw_forms = decoded.get("forms") or decoded.get("translations") or []
        if isinstance(raw_forms, list):
            for item in raw_forms:
                text = str(item).strip()
                if text:
                    forms.append(text)
    if not forms:
        for line in payload.splitlines():
            value = line.strip()
            if not value:
                continue
            if ":" in value:
                pref, rest = value.split(":", 1)
                if pref.lower() in ("form0", "singular", "plural"):
                    forms.append(rest.strip())
                    continue
            forms.append(value)
    if not forms:
        forms = [""]
    while len(forms) < nplurals:
        forms.append(forms[-1])
    return forms[:nplurals]


def _translate_entries(
    entries: List[PoEntry],
    translator: Optional[OpenAITranslator],
    target_lang: str,
    nplurals: int,
    max_entries: int,
) -> Tuple[int, int]:
    translated = 0
    changed = 0
    if translator is None:
        return translated, changed
    for entry in entries:
        if not entry.msgid:
            continue
        if max_entries and translated >= max_entries:
            break
        if entry.msgid_plural:
            forms = translator.translate_plural(entry, target_lang, nplurals)
            for idx, text in enumerate(forms):
                entry.msgstr_plural[idx] = text
            entry.msgstr = forms[0] if forms else ""
            changed += sum(1 for text in forms if text.strip())
        else:
            translation = translator.translate_singular(entry, target_lang)
            entry.msgstr = translation
            if translation.strip():
                changed += 1
        translated += 1
    return translated, changed


def _build_catalog(entries: List[PoEntry], header: PoEntry) -> Dict[str, str]:
    catalog: Dict[str, str] = {}
    catalog[""] = header.msgstr
    for entry in entries:
        if not entry.msgid:
            continue
        if entry.msgid_plural:
            forms = [entry.msgstr_plural.get(idx, "") for idx in sorted(entry.msgstr_plural)]
            catalog[f"{entry.msgid}\0{entry.msgid_plural}"] = "\0".join(forms)
        else:
            catalog[entry.msgid] = entry.msgstr
    return catalog


def _write_mo(catalog: Dict[str, str], path: Path) -> None:
    entries = sorted(catalog.items())
    n = len(entries)
    original_table_offset = 7 * 4
    translation_table_offset = original_table_offset + n * 8
    string_table_offset = translation_table_offset + n * 8

    originals = []
    translations = []
    original_data = b""
    for msgid, _ in entries:
        encoded = msgid.encode("utf-8")
        originals.append((len(encoded), string_table_offset + len(original_data)))
        original_data += encoded + b"\0"

    translation_data = b""
    for _, msgstr in entries:
        encoded = msgstr.encode("utf-8")
        translations.append((len(encoded), string_table_offset + len(original_data) + len(translation_data)))
        translation_data += encoded + b"\0"

    header = struct.pack(
        "<Iiiiiii",
        0x950412de,
        0,
        n,
        original_table_offset,
        translation_table_offset,
        0,
        0,
    )

    with path.open("wb") as handle:
        handle.write(header)
        for length, offset in originals:
            handle.write(struct.pack("<ii", length, offset))
        for length, offset in translations:
            handle.write(struct.pack("<ii", length, offset))
        handle.write(original_data)
        handle.write(translation_data)


def translate_pot_template(
    pot_path: Path,
    languages: Sequence[str],
    translator: Optional[OpenAITranslator],
    output_root: Path,
    compile_mo: bool,
    max_entries: int,
    plural_overrides: Dict[str, str],
) -> List[TranslationResult]:
    header, entries = _parse_po(pot_path)
    if header is None:
        header = PoEntry()
    if not entries:
        return []
    results: List[TranslationResult] = []
    output_root = Path(output_root)
    domain = pot_path.stem

    for lang in languages:
        po_entries = copy.deepcopy(entries)
        po_header = copy.deepcopy(header)
        locale = _resolve_locale(lang)
        plural_expr, nplurals = _plural_setting(locale, plural_overrides)
        translated, changed = _translate_entries(po_entries, translator, locale, nplurals, max_entries)
        metadata = _parse_metadata_lines(po_header.msgstr or "")
        _update_metadata(metadata, "Language", locale)
        _update_metadata(
            metadata,
            "PO-Revision-Date",
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M%z"),
        )
        _update_metadata(metadata, "Plural-Forms", plural_expr)
        po_header.msgstr = _metadata_to_text(metadata)
        lang_dir = output_root / locale
        po_path = lang_dir / f"{domain}-{locale}.po"
        _write_po(po_path, po_header, po_entries, metadata)
        mo_path: Optional[Path] = None
        if compile_mo:
            catalog = _build_catalog(po_entries, po_header)
            mo_path = po_path.with_suffix(".mo")
            _write_mo(catalog, mo_path)
        results.append(
            TranslationResult(
                language=lang,
                locale=locale,
                pot_path=pot_path,
                po_path=po_path,
                mo_path=mo_path,
                translated_count=translated,
                changed_count=changed,
            )
        )
    return results


def _write_po(path: Path, header: PoEntry, entries: List[PoEntry], metadata: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for comment in header.raw_comments:
            handle.write(f"{comment}\n")
        handle.write('msgid ""\n')
        handle.write('msgstr ""\n')
        for line in _serialize_header_lines(metadata):
            handle.write(_quote_po_string(f"{line}\n") + "\n")
        handle.write("\n")
        for entry in entries:
            for comment in entry.raw_comments:
                handle.write(f"{comment}\n")
            if entry.msgctxt:
                handle.write(f"msgctxt {_quote_po_string(entry.msgctxt)}\n")
            handle.write(f"msgid {_quote_po_string(entry.msgid)}\n")
            if entry.msgid_plural:
                handle.write(f"msgid_plural {_quote_po_string(entry.msgid_plural)}\n")
                for idx in sorted(entry.msgstr_plural):
                    handle.write(f"msgstr[{idx}] {_quote_po_string(entry.msgstr_plural[idx])}\n")
            else:
                handle.write(f"msgstr {_quote_po_string(entry.msgstr)}\n")
            handle.write("\n")


def _collect_pot_files(path_arg: str) -> List[Path]:
    path = Path(path_arg)
    if path.is_dir():
        return sorted(path.glob("*.pot"))
    return [path]


def _parse_languages(arg: Optional[str]) -> List[str]:
    if not arg:
        return list(DEFAULT_LANGUAGES)
    normalized = arg.strip().lower()
    if normalized in ("all", "default"):
        return list(DEFAULT_LANGUAGES)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    seen = set()
    ordered: List[str] = []
    for lang in parts:
        if lang in seen:
            continue
        seen.add(lang)
        ordered.append(lang)
    if not ordered:
        raise ValueError("No languages provided.")
    return ordered


def _parse_plural_overrides(values: Optional[Sequence[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError("Invalid --plural-forms value, expected lang=expr")
        lang, expr = item.split("=", 1)
        overrides[lang.strip().lower()] = expr.strip()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate .pot templates into .po/.mo deliveries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python translate_pot.py -i wp-xliff-translator/includes/plugin-update-checker/languages/plugin-update-checker.pot \\
                --languages ar,cs --compile
              python translate_pot.py --input templates --languages default --dry-run
            """
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a .pot template file or directory of .pot templates.",
    )
    parser.add_argument(
        "--languages",
        help="Comma-separated list of target locales (default: ar,cs,de,el,pl,uk).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where <lang> subdirectories are created for .po/.mo files.",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use.")
    parser.add_argument("--rpm", type=int, default=120, help="Requests per minute throttle.")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Limit the number of strings translated per language (0 = all).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the generated .po files into .mo files.",
    )
    parser.add_argument(
        "--plural-forms",
        action="append",
        help="Override Plural-Forms for a locale in the form lang=expression.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse templates and write header-only .po files without calling OpenAI.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit the number of .pot files processed (0 = all).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run and OpenAI is None:
        print("Missing dependency: install openai (pip install openai).", file=sys.stderr)
        return 1

    if not args.dry_run:
        try:
            _ensure_api_key()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    pot_files = _collect_pot_files(args.input)
    if args.max_files:
        pot_files = pot_files[: args.max_files]
    missing = [path for path in pot_files if not path.exists()]
    if missing:
        print(f"Input file not found: {missing[0]}", file=sys.stderr)
        return 1
    if not pot_files:
        print(f"No .pot files found under {args.input}", file=sys.stderr)
        return 1

    languages = _parse_languages(args.languages)
    plural_overrides = _parse_plural_overrides(args.plural_forms)
    translator = None if args.dry_run else OpenAITranslator(args.model, args.rpm)
    output_root = Path(args.output_dir)

    for pot_path in pot_files:
        try:
            results = translate_pot_template(
                pot_path=pot_path,
                languages=languages,
                translator=translator,
                output_root=output_root,
                compile_mo=args.compile,
                max_entries=args.max_entries,
                plural_overrides=plural_overrides,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Translation failed for {pot_path}: {exc}", file=sys.stderr)
            return 1
        if not results:
            print(f"[SKIP] {pot_path.name} has no translatable entries.")
            continue
        for res in results:
            print(
                f"[{res.language}] Translated {res.translated_count} strings "
                f"({res.changed_count} changed) for {pot_path.name}, saved {res.po_path} "
                f"(locale {res.locale})."
            )
            if args.compile and res.mo_path:
                print(f"[{res.language}] Compiled {res.mo_path} (locale {res.locale}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
