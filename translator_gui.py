#!/usr/bin/env python3
"""
Simple Tkinter GUI to translate XLIFF 1.2 files using OpenAI chat models.

Features:
* Select one or more XLIFF files.
* Choose target languages (defaults to ar, cs, de, el, pl, uk).
* Optionally skip targets that already have content.
* Writes translated copies alongside the originals with a "-translated.xliff" suffix.

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


SUPPORTED_LANGUAGES = ("ar", "cs", "de", "el", "pl", "uk")
XLING_NAMESPACE = "urn:oasis:names:tc:xliff:document:1.2"


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def _translate_text(client: "OpenAI", model: str, text: str, target_lang: str) -> str:
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
    return response.choices[0].message.content.strip()


class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("XLIFF Translator (OpenAI)")
        self.selected_files: List[Path] = []

        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.skip_prefilled_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.rpm_var = tk.IntVar(value=120)  # requests per minute throttle
        self.lang_vars: Dict[str, tk.BooleanVar] = {lang: tk.BooleanVar(value=True) for lang in SUPPORTED_LANGUAGES}

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

        # Language selection
        lang_frame = ttk.LabelFrame(frm, text="Target languages")
        lang_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=6)
        for idx, lang in enumerate(SUPPORTED_LANGUAGES):
            ttk.Checkbutton(lang_frame, text=lang, variable=self.lang_vars[lang]).grid(row=0, column=idx, padx=4, pady=2)

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
            text="Overwrite originals (otherwise -translated.xliff)",
            variable=self.overwrite_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(options_frame, text="Max requests/min:").grid(row=1, column=2, sticky="e", padx=(0, 4))
        ttk.Entry(options_frame, textvariable=self.rpm_var, width=6).grid(row=1, column=3, sticky="w")

        # Buttons
        self.translate_btn = ttk.Button(frm, text="Translate", command=self.start_translation)
        self.translate_btn.grid(row=4, column=0, sticky="w", pady=6)

        # Log
        ttk.Label(frm, text="Log:").grid(row=5, column=0, sticky="w")
        self.log_text = tk.Text(frm, height=12, width=100, state="disabled")
        self.log_text.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(6, weight=2)

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
        xliffs = sorted(repo_root.rglob("*.xliff"))
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
        chosen_langs = [lang for lang, var in self.lang_vars.items() if var.get()]
        if not chosen_langs:
            messagebox.showerror("No languages", "Select at least one target language.")
            return
        if OpenAI is None:
            messagebox.showerror("Missing dependency", "Please install the openai package (`pip install openai`).")
            return
        # Allow reading from .env in the current working directory.
        env_path = Path.cwd() / ".env"
        if not os.getenv("OPENAI_API_KEY"):
            dotenv_val = _load_dotenv_key("OPENAI_API_KEY", env_path)
            if dotenv_val:
                os.environ["OPENAI_API_KEY"] = dotenv_val
        if not os.getenv("OPENAI_API_KEY"):
            messagebox.showerror("Missing OPENAI_API_KEY", "Set OPENAI_API_KEY in your environment or .env file.")
            return

        model = self.model_var.get().strip() or "gpt-4o-mini"
        skip_prefilled = self.skip_prefilled_var.get()
        overwrite = self.overwrite_var.get()
        rpm = max(1, self.rpm_var.get() or 1)

        self.translate_btn.config(state="disabled")
        thread = threading.Thread(
            target=self._run_translation,
            args=(self.selected_files, chosen_langs, model, skip_prefilled, overwrite, rpm),
            daemon=True,
        )
        thread.start()

    def _run_translation(
        self,
        files: List[Path],
        langs: List[str],
        model: str,
        skip_prefilled: bool,
        overwrite: bool,
        rpm: int,
    ) -> float:
        try:
            client = OpenAI()
        except Exception as exc:  # noqa: BLE001
            self._on_done(error=f"Failed to initialize OpenAI client: {exc}")
            return

        min_delay = 60.0 / float(rpm)
        last_call = 0.0

        for path in files:
            try:
                tree, file_el, ns = _load_xliff(path)
                target_language = file_el.attrib.get("target-language", "")
                if target_language and target_language not in langs:
                    # Respect the file's target language if present.
                    target_langs = [target_language]
                else:
                    target_langs = langs
                self.log(f"Processing {path.name} for languages: {', '.join(target_langs)}")
                for target_lang in target_langs:
                    last_call = self._translate_file(
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
            except Exception as exc:  # noqa: BLE001
                self.log(f"[ERROR] {path}: {exc}")

        self._on_done()

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
    ) -> None:
        for tu in _iter_trans_units(file_el, ns):
            tu_id = tu.attrib.get("id", "<no-id>")
            source_el = tu.find("x:source", ns) if ns else tu.find("source")
            target_el = _ensure_target(tu, ns)
            source_text = (source_el.text or "").strip() if source_el is not None else ""
            target_text = (target_el.text or "").strip()

            if not source_text:
                self.log(f"  [skip] {tu_id}: empty source")
                continue
            if skip_prefilled and target_text:
                self.log(f"  [skip] {tu_id}: target already filled")
                continue

            try:
                # Rate limit
                now = time.time()
                wait_for = min_delay - (now - last_call)
                if wait_for > 0:
                    time.sleep(wait_for)
                call_started = time.time()
                translated = _translate_text(client, model, source_text, target_lang)
                last_call = call_started
            except Exception as exc:  # noqa: BLE001
                self.log(f"  [error] {tu_id}: translation failed: {exc}")
                continue

            target_el.text = translated
            self.log(f"  [ok] {tu_id}")

        output_path = path if overwrite else path.with_name(f"{path.stem}-translated.xliff")
        _write_xliff(tree, output_path)
        self.log(f"Saved: {output_path}")
        return last_call

    def _on_done(self, error: Optional[str] = None) -> None:
        if error:
            messagebox.showerror("Error", error)
        else:
            messagebox.showinfo("Done", "Translation finished.")
        self.translate_btn.config(state="normal")


def main() -> None:
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
