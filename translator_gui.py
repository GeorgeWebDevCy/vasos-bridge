#!/usr/bin/env python3
"""
Simple Tkinter GUI to translate XLIFF 1.2 files using OpenAI chat models.

Features:
* Select one or more XLIFF files.
* Uses the <file> target-language attribute from each XLIFF file.
* Optionally skip targets that already have content.
* Writes translated copies under translated/ with a "-<lang>-translated.xliff" suffix.

Requirements:
* Python 3.9+
* openai >= 1.0.0 (install with `pip install openai`)
* Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic detection
except Exception:  # noqa: BLE001
    detect = None
    LangDetectException = Exception


XLING_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"
WPML_NAMESPACE = "https://cdn.wpml.org/xliff/custom-attributes.xsd"


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


def _translate_text(client: "OpenAI", model: str, text: str, target_lang: str) -> Tuple[str, str, str]:
    # Keep prompt tight to reduce latency and preserve markup.
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
    translated = response.choices[0].message.content.strip()
    return translated, system_prompt, user_prompt


class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("XLIFF Translator (OpenAI)")
        self.selected_files: List[Path] = []
        self.progress_total = 0
        self.progress_done = 0

        self.model_var = tk.StringVar(value="gpt-4.1")
        self.skip_prefilled_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.rpm_var = tk.IntVar(value=120)  # requests per minute throttle
        self.langdetect_available = detect is not None
        self._langdetect_warned = False
        self.output_root = Path.cwd() / "translated"

        self._build_ui()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # File selection
        ttk.Label(frm, text="Selected XLIFF files:").grid(row=0, column=0, sticky="w")
        self.files_list = tk.Listbox(frm, height=6, width=80)
        self.files_list.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(2, 6))
        frm.rowconfigure(1, weight=1)
        ttk.Button(frm, text="Choose files...", command=self.choose_files).grid(row=0, column=2, sticky="e")
        ttk.Button(frm, text="Load all repo .xliff", command=self.load_repo_files).grid(row=0, column=1, sticky="e", padx=(0, 6))

        ttk.Label(
            frm,
            text="Target language: read from each XLIFF file's target-language attribute.",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=6)

        # Options
        options_frame = ttk.Frame(frm)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, sticky="w")
        ttk.Entry(options_frame, textvariable=self.model_var, width=20).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Checkbutton(
            options_frame,
            text="Skip already filled targets",
            variable=self.skip_prefilled_var,
        ).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(
            options_frame,
            text="Overwrite originals (otherwise saved under translated/)",
            variable=self.overwrite_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(options_frame, text="Max requests/min:").grid(row=1, column=2, sticky="e", padx=(0, 4))
        ttk.Entry(options_frame, textvariable=self.rpm_var, width=6).grid(row=1, column=3, sticky="w")

        # Buttons
        self.translate_btn = ttk.Button(frm, text="Translate", command=self.start_translation)
        self.translate_btn.grid(row=4, column=0, sticky="w", pady=6)

        # Progress
        progress_frame = ttk.Frame(frm)
        progress_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=(0, 6))
        progress_frame.columnconfigure(1, weight=1)
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0% (0/0)")
        self.progress_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.progress_bar = ttk.Progressbar(progress_frame, maximum=1, value=0)
        self.progress_bar.grid(row=0, column=1, sticky="we")

        # Log
        ttk.Label(frm, text="Log:").grid(row=6, column=0, sticky="w")
        self.log_text = tk.Text(frm, height=12, width=100, state="disabled")
        self.log_text.grid(row=7, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(7, weight=2)

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def choose_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select XLIFF files",
            filetypes=[("XLIFF files", "*.xliff"), ("All files", "*.*")],
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self.files_list.delete(0, "end")
            for f in self.selected_files:
                self.files_list.insert("end", str(f))

    def load_repo_files(self) -> None:
        repo_root = Path.cwd()
        import_root = repo_root / "wpml-import"
        xliffs = sorted(
            p
            for p in repo_root.rglob("*.xliff")
            if not p.is_relative_to(self.output_root) and not p.is_relative_to(import_root)
        )
        if not xliffs:
            messagebox.showinfo("No files", f"No .xliff files found under {repo_root}")
            return
        self.selected_files = xliffs
        self.files_list.delete(0, "end")
        for f in self.selected_files:
            self.files_list.insert("end", str(f))
        messagebox.showinfo("Loaded", f"Loaded {len(xliffs)} XLIFF files from {repo_root}")

    def start_translation(self) -> None:
        if not self.selected_files:
            messagebox.showerror("No files", "Please choose at least one XLIFF file.")
            return
        if OpenAI is None:
            messagebox.showerror("Missing dependency", "Please install the openai package (`pip install openai`).")
            return
        if self.skip_prefilled_var.get() and not self.langdetect_available and not self._langdetect_warned:
            messagebox.showwarning(
                "Language detection unavailable",
                "Skipping prefilled targets without verifying language because langdetect is not installed "
                "(pip install langdetect).",
            )
            self._langdetect_warned = True
        # Prefer .env in the current working directory to avoid stale global keys.
        env_path = Path.cwd() / ".env"
        dotenv_val = _load_dotenv_key("OPENAI_API_KEY", env_path)
        if dotenv_val:
            existing = os.getenv("OPENAI_API_KEY")
            if existing and existing != dotenv_val:
                self.log("Using OPENAI_API_KEY from .env (overriding existing environment value).")
            os.environ["OPENAI_API_KEY"] = dotenv_val
        if not os.getenv("OPENAI_API_KEY"):
            messagebox.showerror("Missing OPENAI_API_KEY", "Set OPENAI_API_KEY in your environment or .env file.")
            return

        model = self.model_var.get().strip() or "gpt-4o-mini"
        skip_prefilled = self.skip_prefilled_var.get()
        overwrite = self.overwrite_var.get()
        rpm = max(1, self.rpm_var.get() or 1)
        if not overwrite:
            self.log(f"Outputs will be saved under {self.output_root}")

        total_units = self._count_translation_units(self.selected_files, skip_prefilled)
        self._reset_progress(total_units)
        if total_units == 0:
            self.translate_btn.config(state="normal")
            messagebox.showinfo(
                "Nothing to do",
                "No trans-units require translation (missing target-language, empty sources, or already filled).",
            )
            return

        self.translate_btn.config(state="disabled")
        thread = threading.Thread(
            target=self._run_translation,
            args=(self.selected_files, model, skip_prefilled, overwrite, rpm),
            daemon=True,
        )
        thread.start()

    def _run_translation(
        self,
        files: List[Path],
        model: str,
        skip_prefilled: bool,
        overwrite: bool,
        rpm: int,
    ) -> None:
        try:
            client = OpenAI()
        except Exception as exc:  # noqa: BLE001
            self._on_done(error=f"Failed to initialize OpenAI client: {exc}")
            return

        min_delay = 60.0 / float(rpm)
        last_call = 0.0

        for path in files:
            try:
                _, file_el, _ = _load_xliff(path)
                target_lang = file_el.attrib.get("target-language", "").strip()
                if not target_lang:
                    self.log(f"  [skip-file] {path.name}: missing target-language attribute")
                    continue
                self.log(f"Processing {path.name} for target language: {target_lang}")
                try:
                    tree, file_el, ns = _load_xliff(path)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[ERROR] {path}: {exc}")
                    continue
                file_el.attrib["target-language"] = target_lang
                pending = self._count_units_for_file(file_el, ns, target_lang, skip_prefilled)
                if pending == 0:
                    self.log(f"  [skip-file] {path.name}: nothing to do for {target_lang}")
                    continue
                last_call, translated_any = self._translate_file(
                    client,
                    model,
                    tree,
                    file_el,
                    ns,
                    path,
                    target_lang,
                    skip_prefilled,
                    overwrite,
                    min_delay,
                    last_call,
                )
                if not translated_any:
                    self.log(f"  [skip-file] {path.name}: no translations performed for {target_lang}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[ERROR] {path}: {exc}")

        self._on_done()

    def _build_output_name(self, input_path: Path, target_lang: str) -> str:
        lang_suffix = target_lang or "unknown"
        return f"{input_path.stem}-{lang_suffix}-translated{input_path.suffix}"

    def _get_output_path(self, input_path: Path, target_lang: str, overwrite: bool) -> Path:
        if overwrite:
            return input_path
        output_name = self._build_output_name(input_path, target_lang)
        return self.output_root / output_name

    def _translate_file(
        self,
        client: "OpenAI",
        model: str,
        tree: ET.ElementTree,
        file_el: ET.Element,
        ns: Dict[str, str],
        path: Path,
        target_lang: str,
        skip_prefilled: bool,
        overwrite: bool,
        min_delay: float,
        last_call: float,
    ) -> Tuple[float, bool]:
        translated_any = False
        file_el.attrib["target-language"] = target_lang
        for tu in _iter_trans_units(file_el, ns):
            tu_id = tu.attrib.get("id", "<no-id>")
            source_el = tu.find("x:source", ns) if ns else tu.find("source")
            target_el = _ensure_target(tu, ns)
            source_text = (source_el.text or "").strip() if source_el is not None else ""
            target_text = (target_el.text or "").strip()

            if not source_text:
                self.log(f"  [skip] {tu_id}: empty source")
                continue
            if skip_prefilled and self._should_skip_prefilled(target_text, target_lang):
                self._log_detected_language(target_text, target_lang, tu_id, note="skip")
                self.log(f"  [skip] {tu_id}: target already filled and matches {target_lang}")
                continue

            try:
                # Rate limit
                now = time.time()
                wait_for = min_delay - (now - last_call)
                if wait_for > 0:
                    time.sleep(wait_for)
                call_started = time.time()
                translated, system_prompt, user_prompt = _translate_text(
                    client, model, source_text, target_lang
                )
                last_call = call_started
            except Exception as exc:  # noqa: BLE001
                self.log(f"  [error] {tu_id}: translation failed: {exc}")
                self._increment_progress()
                continue

            target_el.text = translated
            self.log(f"  [prompt] {tu_id} -> {target_lang}")
            self.log(f"    system: {system_prompt}")
            self.log(f"    user: {user_prompt}")
            self.log(f"    response: {translated}")
            self.log(f"  [ok] {tu_id}")
            self._warn_if_lang_mismatch(translated, target_lang, tu_id)
            self._increment_progress()
            translated_any = True

        if translated_any:
            output_path = self._get_output_path(path, target_lang, overwrite)
            _write_xliff(tree, output_path)
            self.log(f"Saved: {output_path}")
        return last_call, translated_any

    def _on_done(self, error: Optional[str] = None) -> None:
        if error:
            messagebox.showerror("Error", error)
        else:
            messagebox.showinfo("Done", "Translation finished.")
        self.translate_btn.config(state="normal")
        if not error and self.progress_total:
            self._set_progress_complete()
        else:
            self._update_progress_label()

    def _count_translation_units(self, files: List[Path], skip_prefilled: bool) -> int:
        total = 0
        for path in files:
            try:
                tree, file_el, ns = _load_xliff(path)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[count-error] {path}: {exc}")
                continue
            target_lang = file_el.attrib.get("target-language", "").strip()
            if not target_lang:
                self.log(f"[count-skip] {path.name}: missing target-language attribute")
                continue
            for tu in _iter_trans_units(file_el, ns):
                source_el = tu.find("x:source", ns) if ns else tu.find("source")
                target_el = _ensure_target(tu, ns)
                source_text = (source_el.text or "").strip() if source_el is not None else ""
                target_text = (target_el.text or "").strip()
                if not source_text:
                    continue
                if skip_prefilled and self._should_skip_prefilled(target_text, target_lang):
                    continue
                total += 1
        return total

    def _count_units_for_file(self, file_el: ET.Element, ns: Dict[str, str], target_lang: str, skip_prefilled: bool) -> int:
        pending = 0
        for tu in _iter_trans_units(file_el, ns):
            source_el = tu.find("x:source", ns) if ns else tu.find("source")
            target_el = _ensure_target(tu, ns)
            source_text = (source_el.text or "").strip() if source_el is not None else ""
            target_text = (target_el.text or "").strip()
            if not source_text:
                continue
            if skip_prefilled and self._should_skip_prefilled(target_text, target_lang):
                continue
            pending += 1
        return pending

    def _detect_language(self, text: str) -> Optional[str]:
        if not self.langdetect_available or not text.strip():
            return None
        try:
            return detect(text)
        except LangDetectException:
            return None

    def _should_skip_prefilled(self, target_text: str, expected_lang: str) -> bool:
        if not target_text:
            return False
        detected = self._detect_language(target_text)
        if detected is None:
            return True  # no detector; fallback to legacy behavior
        return detected == expected_lang

    def _warn_if_lang_mismatch(self, text: str, expected_lang: str, tu_id: str) -> None:
        detected = self._detect_language(text)
        if detected:
            self._log_detected_language(text, expected_lang, tu_id, detected=detected)
            if detected != expected_lang:
                self.log(f"  [warn] {tu_id}: detected {detected}, expected {expected_lang}")

    def _log_detected_language(
        self, text: str, expected_lang: str, tu_id: str, note: str = "", detected: Optional[str] = None
    ) -> None:
        if not text or not self.langdetect_available:
            return
        detected_lang = detected or self._detect_language(text)
        if detected_lang:
            suffix = f" ({note})" if note else ""
            self.log(f"  [lang] {tu_id}: detected {detected_lang}, expected {expected_lang}{suffix}")

    def _reset_progress(self, total: int) -> None:
        self.progress_total = total
        self.progress_done = 0
        self.progress_bar.configure(maximum=max(total, 1), value=0)
        self._update_progress_label()

    def _increment_progress(self, step: int = 1) -> None:
        self.progress_done = min(self.progress_done + step, self.progress_total or self.progress_done + step)
        self.progress_bar.configure(value=self.progress_done)
        self._update_progress_label()

    def _set_progress_complete(self) -> None:
        self.progress_done = self.progress_total
        self.progress_bar.configure(value=self.progress_total)
        self._update_progress_label()

    def _update_progress_label(self) -> None:
        percent = int((self.progress_done / self.progress_total) * 100) if self.progress_total else 0
        self.progress_label.configure(text=f"Progress: {percent}% ({self.progress_done}/{self.progress_total})")
        self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
