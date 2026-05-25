#!/usr/bin/env python3
"""
Tkinter GUI for translating gettext .pot templates via the translate_pot CLI helpers.

This keeps gettext workflows isolated from the XLIFF UI while reusing the same AI translation core.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Iterable, List, Optional

import translate_pot
import duplicate_po_locale
from llm_provider import ChatClient, DEFAULT_MODELS, SUPPORTED_PROVIDERS, default_model, prepare_provider
from translate_pot import (
    AITranslator,
    DEFAULT_LANGUAGES,
    clean_po_references,
    resolve_default_context,
    translate_pot_template,
)

from windows_taskbar import TaskbarController, TBPFLAG


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class PotTranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cucumber Destroyer - gettext Translator")
        self.selected_files: List[Path] = []
        self.app_root = _app_root()

        # Taskbar integration
        self.taskbar = TaskbarController()
        self.taskbar.set_app_id("Cucumber.Destroyer.PotTranslator.1")

        try:
            # Prefer .ico for Windows taskbar quality
            ico_path = self.app_root / "cucumber.ico"
            if ico_path.exists():
                self.root.iconbitmap(str(ico_path))
            else:
                self.icon_img = tk.PhotoImage(file=self.app_root / "cucumber.png")
                self.root.iconphoto(True, self.icon_img)
        except Exception:
            pass
        self.languages_var = tk.StringVar(value=",".join(DEFAULT_LANGUAGES))
        self.provider_var = tk.StringVar(value="openai")
        self.model_var = tk.StringVar(value=default_model("openai"))
        self.rpm_var = tk.IntVar(value=120)
        self.max_entries_var = tk.StringVar(value="0")
        self.output_dir_var = tk.StringVar(value="po")
        self.compile_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.plural_var = tk.StringVar(value="")
        self.default_context_var = tk.StringVar(value="")
        self.ai_comment_var = tk.BooleanVar(value=True)
        self.cleanup_source_var = tk.StringVar(value="po")
        self.cleanup_compile_var = tk.BooleanVar(value=True)
        self.duplicate_source_var = tk.StringVar(value="")
        self.duplicate_targets_var = tk.StringVar(value="")
        self.duplicate_output_var = tk.StringVar(value="po")
        self.duplicate_compile_var = tk.BooleanVar(value=True)
        self.duplicate_plural_var = tk.StringVar(value="")
        self.progress_total = 0
        self.progress_done = 0
        self._build_ui()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Selected .pot files:").grid(row=0, column=0, sticky="w")
        self.files_list = tk.Listbox(frm, height=6, width=90)
        self.files_list.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(2, 6))
        frm.rowconfigure(1, weight=1)
        ttk.Button(frm, text="Add files...", command=self.choose_files).grid(row=0, column=2, sticky="e")
        ttk.Button(frm, text="Add directory...", command=self.add_directory).grid(row=0, column=3, sticky="e", padx=(6, 0))
        ttk.Button(frm, text="Clear selection", command=self.clear_selection).grid(row=0, column=1, sticky="e", padx=(0, 6))

        ttk.Label(frm, text="Languages (comma-separated):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.languages_var, width=40).grid(row=2, column=1, columnspan=2, sticky="w")

        options = ttk.Frame(frm)
        options.grid(row=3, column=0, columnspan=3, sticky="we", pady=6)
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Output directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.output_dir_var, width=30).grid(row=0, column=1, sticky="w")
        ttk.Button(options, text="Browse", command=self.choose_output_dir).grid(row=0, column=2, sticky="e")
        ttk.Label(options, text="Provider:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        provider_box = ttk.Combobox(
            options,
            textvariable=self.provider_var,
            values=SUPPORTED_PROVIDERS,
            state="readonly",
            width=14,
        )
        provider_box.grid(row=1, column=1, sticky="w", pady=(6, 0))
        provider_box.bind("<<ComboboxSelected>>", self._provider_changed)
        ttk.Label(options, text="Model:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.model_box = ttk.Combobox(options, textvariable=self.model_var, width=28)
        self.model_box.grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(options, text="RPM:").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Entry(options, textvariable=self.rpm_var, width=6).grid(row=2, column=3, sticky="w", pady=(6, 0))
        self.load_models_btn = ttk.Button(options, text="Load Models", command=self.load_models)
        self.load_models_btn.grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        self.test_connection_btn = ttk.Button(options, text="Test Connection", command=self.test_connection)
        self.test_connection_btn.grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Label(options, text="Max entries (0=all):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(options, textvariable=self.max_entries_var, width=10).grid(row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(options, text="Compile .mo", variable=self.compile_var).grid(row=3, column=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(options, text="Dry run (no AI call)", variable=self.dry_run_var).grid(row=3, column=3, sticky="w", pady=(6, 0))
        ttk.Label(options, text="Default context (optional):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(options, textvariable=self.default_context_var, width=40).grid(row=4, column=1, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Checkbutton(
            options,
            text="Tag translations (#. AI-generated)",
            variable=self.ai_comment_var,
        ).grid(row=5, column=0, columnspan=4, sticky="w")
        ttk.Label(
            options,
            text="Plural overrides (lang=expr, comma-separated):",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Entry(options, textvariable=self.plural_var, width=60).grid(row=7, column=0, columnspan=4, sticky="we")

        cleanup_frame = ttk.LabelFrame(frm, text="Clean AI reference fragments", padding=8)
        cleanup_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=8)
        cleanup_frame.columnconfigure(0, weight=1)
        cleanup_frame.columnconfigure(1, weight=1)
        cleanup_frame.columnconfigure(2, weight=0)
        cleanup_frame.columnconfigure(3, weight=0)

        ttk.Label(cleanup_frame, text=".po file or directory:").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Entry(cleanup_frame, textvariable=self.cleanup_source_var, width=60).grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 4))
        ttk.Button(cleanup_frame, text="File...", command=self.choose_cleanup_file).grid(row=1, column=2, padx=(6, 0), sticky="w")
        ttk.Button(cleanup_frame, text="Directory...", command=self.choose_cleanup_directory).grid(row=1, column=3, padx=(6, 0), sticky="w")
        ttk.Checkbutton(cleanup_frame, text="Recompile .mo", variable=self.cleanup_compile_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.cleanup_btn = ttk.Button(cleanup_frame, text="Clean .po bundles", command=self.start_cleanup)
        self.cleanup_btn.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        dup_frame = ttk.LabelFrame(frm, text="Duplicate existing .po bundles", padding=8)
        dup_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=8)
        dup_frame.columnconfigure(0, weight=1)
        dup_frame.columnconfigure(1, weight=1)
        dup_frame.columnconfigure(2, weight=0)
        dup_frame.columnconfigure(3, weight=0)

        ttk.Label(dup_frame, text="Source file or directory:").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Entry(dup_frame, textvariable=self.duplicate_source_var, width=60).grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 4))
        ttk.Button(dup_frame, text="File...", command=self.choose_duplicate_file).grid(row=1, column=2, padx=(6, 0), sticky="w")
        ttk.Button(dup_frame, text="Directory...", command=self.choose_duplicate_directory).grid(row=1, column=3, padx=(6, 0), sticky="w")
        ttk.Label(dup_frame, text="Target locales (comma-separated):").grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Entry(dup_frame, textvariable=self.duplicate_targets_var, width=40).grid(row=3, column=0, columnspan=4, sticky="we")
        ttk.Label(dup_frame, text="Output directory:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(dup_frame, textvariable=self.duplicate_output_var, width=32).grid(row=4, column=1, columnspan=2, sticky="we", padx=(6, 0))
        ttk.Button(dup_frame, text="Browse", command=self.choose_duplicate_output_dir).grid(row=4, column=3, sticky="e", padx=(6, 0))
        ttk.Checkbutton(dup_frame, text="Compile .mo", variable=self.duplicate_compile_var).grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Label(dup_frame, text="Plural overrides (lang=expr, comma-separated):").grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Entry(dup_frame, textvariable=self.duplicate_plural_var, width=60).grid(row=7, column=0, columnspan=4, sticky="we")
        self.duplicate_btn = ttk.Button(
            dup_frame,
            text="Duplicate .po bundles",
            command=self.start_duplication,
        )
        self.duplicate_btn.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.translate_btn = ttk.Button(frm, text="Translate templates", command=self.start_translation)
        self.translate_btn.grid(row=6, column=0, sticky="w", pady=8)

        progress_frame = ttk.Frame(frm)
        progress_frame.grid(row=7, column=0, columnspan=3, sticky="we")
        progress_frame.columnconfigure(1, weight=1)
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0% (0/0)")
        self.progress_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.progress_bar = ttk.Progressbar(progress_frame, maximum=1, value=0)
        self.progress_bar.grid(row=0, column=1, sticky="we")

        ttk.Label(frm, text="Log:").grid(row=8, column=0, sticky="w")
        self.log_text = tk.Text(frm, height=12, width=100, state="disabled")
        self.log_text.grid(row=9, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(9, weight=2)

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _provider_changed(self, _event: object = None) -> None:
        if self.model_var.get().strip() in DEFAULT_MODELS.values():
            self.model_var.set(default_model(self.provider_var.get()))
        self.model_box.configure(values=())

    def _prepare_selected_provider(self) -> tuple[str, str]:
        provider = prepare_provider(self.provider_var.get().strip() or "openai", self.app_root / ".env")
        model = self.model_var.get().strip() or default_model(provider)
        return provider, model

    def load_models(self) -> None:
        try:
            provider, model = self._prepare_selected_provider()
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("Provider setup", str(exc))
            return
        self.load_models_btn.config(state="disabled")
        self.log(f"Loading available {provider} models...")
        threading.Thread(target=self._load_models_worker, args=(provider, model), daemon=True).start()

    def _load_models_worker(self, provider: str, model: str) -> None:
        try:
            models = ChatClient(provider, model).list_models()
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda error=str(exc): self._models_failed(error))
            return
        self.root.after(0, lambda: self._models_loaded(provider, models))

    def _models_failed(self, error: str) -> None:
        self.load_models_btn.config(state="normal")
        self.log(f"[models error] {error}")
        messagebox.showerror("Load models failed", error)

    def _models_loaded(self, provider: str, models: List[str]) -> None:
        self.load_models_btn.config(state="normal")
        if provider != self.provider_var.get().strip():
            return
        self.model_box.configure(values=models)
        if models and self.model_var.get().strip() not in models:
            preferred = default_model(provider)
            self.model_var.set(preferred if preferred in models else models[0])
        self.log(f"Loaded {len(models)} available {provider} model(s).")
        if not models:
            messagebox.showinfo("No models found", f"No models were returned by {provider}.")

    def test_connection(self) -> None:
        try:
            provider, model = self._prepare_selected_provider()
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("Provider setup", str(exc))
            return
        self.test_connection_btn.config(state="disabled")
        self.log(f"Testing {provider} connection with {model}...")
        threading.Thread(target=self._test_connection_worker, args=(provider, model), daemon=True).start()

    def _test_connection_worker(self, provider: str, model: str) -> None:
        try:
            response = ChatClient(provider, model).complete(
                "Return exactly the requested text, with no explanation.",
                "Return exactly: connection-ok",
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda error=str(exc): self._connection_failed(error))
            return
        self.root.after(0, lambda: self._connection_succeeded(provider, model, response))

    def _connection_failed(self, error: str) -> None:
        self.test_connection_btn.config(state="normal")
        self.log(f"[connection error] {error}")
        messagebox.showerror("Connection failed", error)

    def _connection_succeeded(self, provider: str, model: str, response: str) -> None:
        self.test_connection_btn.config(state="normal")
        self.log(f"[connection ok] {provider}/{model}: {response}")
        messagebox.showinfo("Connection successful", f"Connected to {provider} using {model}.")

    def choose_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select .pot files",
            filetypes=[("Portable Object Template", "*.pot"), ("All files", "*.*")],
        )
        if not files:
            return
        self._add_to_selection(Path(f) for f in files)

    def add_directory(self) -> None:
        directory = filedialog.askdirectory(title="Add directory of .pot files", mustexist=True)
        if not directory:
            return
        pot_paths = sorted(Path(directory).glob("*.pot"))
        if not pot_paths:
            messagebox.showinfo("No templates", f"No .pot files found in {directory}.")
            return
        self._add_to_selection(pot_paths)

    def _add_to_selection(self, paths: Iterable[Path]) -> None:
        existing = {path.resolve() for path in self.selected_files}
        added = False
        for path in paths:
            resolved = path.resolve()
            if resolved in existing:
                continue
            self.selected_files.append(resolved)
            self.files_list.insert("end", str(resolved))
            existing.add(resolved)
            added = True
        if not added:
            messagebox.showinfo("Already added", "All selected files are already in the list.")

    def clear_selection(self) -> None:
        self.selected_files = []
        self.files_list.delete(0, "end")

    def choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Output directory", mustexist=False)
        if directory:
            self.output_dir_var.set(directory)

    def choose_duplicate_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .po file",
            filetypes=[("Portable Object", "*.po"), ("All files", "*.*")],
        )
        if path:
            self.duplicate_source_var.set(path)

    def choose_duplicate_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select directory containing .po files", mustexist=True)
        if directory:
            self.duplicate_source_var.set(directory)

    def choose_duplicate_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Duplicate output directory", mustexist=False)
        if directory:
            self.duplicate_output_var.set(directory)

    def choose_cleanup_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .po file to clean",
            filetypes=[("Portable Object", "*.po"), ("All files", "*.*")],
        )
        if path:
            self.cleanup_source_var.set(path)

    def choose_cleanup_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select directory containing .po files", mustexist=True)
        if directory:
            self.cleanup_source_var.set(directory)

    def start_cleanup(self) -> None:
        source_value = self.cleanup_source_var.get().strip()
        if not source_value:
            messagebox.showerror("Source required", "Provide a .po file or directory.")
            return
        sources = duplicate_po_locale.collect_po_paths(source_value)
        if not sources:
            messagebox.showerror("Source missing", f"No .po files found at {source_value}.")
            return
        self.cleanup_btn.config(state="disabled")
        self.log(f"[cleanup] {source_value}")
        thread = threading.Thread(
            target=self._run_cleanup,
            args=(sources, self.cleanup_compile_var.get()),
            daemon=True,
        )
        thread.start()

    def start_duplication(self) -> None:
        source_value = self.duplicate_source_var.get().strip()
        if not source_value:
            messagebox.showerror("Source required", "Provide a .po file or directory.")
            return
        sources = duplicate_po_locale.collect_po_paths(source_value)
        if not sources:
            messagebox.showerror("Source missing", f"No .po files found at {source_value}.")
            return
        targets = [token.strip() for token in self.duplicate_targets_var.get().split(",") if token.strip()]
        if not targets:
            messagebox.showerror("No targets", "Provide at least one target locale.")
            return
        try:
            plural_overrides = self._parse_plural_overrides(self.duplicate_plural_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid plural overrides", str(exc))
            return
        output_dir = Path(self.duplicate_output_var.get() or "po")
        compile_mo = self.duplicate_compile_var.get()
        self.duplicate_btn.config(state="disabled")
        self.log(f"[duplication] {source_value} -> {', '.join(targets)}")
        thread = threading.Thread(
            target=self._run_duplication,
            args=(sources, targets, output_dir, compile_mo, plural_overrides),
            daemon=True,
        )
        thread.start()

    def _run_duplication(
        self,
        sources: List[Path],
        targets: List[str],
        output_dir: Path,
        compile_mo: bool,
        plural_overrides: Dict[str, str],
    ) -> None:
        try:
            results = duplicate_po_locale.duplicate_po_locales(
                sources=sources,
                targets=targets,
                output_root=output_dir,
                compile_mo=compile_mo,
                plural_overrides=plural_overrides,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"[duplication error] {exc}")
        else:
            for res in results:
                self.log(f"[dup {res.target_locale}] {res.source.name} -> {res.po_path}")
            if res.mo_path:
                self.log(f"  compiled {res.mo_path}")
        finally:
            self._on_duplication_done()

    def _run_cleanup(self, sources: List[Path], compile_mo: bool) -> None:
        try:
            for po_path in sources:
                if not po_path.exists():
                    self.log(f"[cleanup error] {po_path}: file not found.")
                    continue
                try:
                    cleaned, mo_path = clean_po_references(po_path, compile_mo)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[cleanup error] {po_path.name}: {exc}")
                    continue
                if not cleaned:
                    self.log(f"[cleanup skip] {po_path.name}: no AI references detected.")
                    continue
                self.log(f"[cleanup] {po_path.name}: cleaned {cleaned} entries.")
                if mo_path:
                    self.log(f"  compiled {mo_path}")
        finally:
            self._on_cleanup_done()

    def _on_duplication_done(self) -> None:
        self.duplicate_btn.config(state="normal")
        self.log("Duplication run complete.")

    def _on_cleanup_done(self) -> None:
        self.cleanup_btn.config(state="normal")
        self.log("Cleanup run complete.")

    def start_translation(self) -> None:
        if not self.selected_files:
            messagebox.showerror("No files", "Please select at least one .pot file.")
            return
        try:
            languages = translate_pot._parse_languages(self.languages_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid languages", str(exc))
            return
        if not languages:
            messagebox.showerror("No languages", "Provide at least one target language.")
            return
        try:
            max_entries = int(self.max_entries_var.get() or 0)
        except ValueError:
            messagebox.showerror("Invalid max entries", "Enter 0 or a positive integer.")
            return
        if max_entries < 0:
            messagebox.showerror("Invalid max entries", "Max entries cannot be negative.")
            return
        try:
            plural_overrides = self._parse_plural_overrides(self.plural_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid plural overrides", str(exc))
            return
        output_dir = Path(self.output_dir_var.get() or "po")
        dry_run = self.dry_run_var.get()
        provider = self.provider_var.get().strip() or "openai"
        if not dry_run:
            try:
                provider = prepare_provider(provider, self.app_root / ".env")
            except (RuntimeError, ValueError) as exc:
                messagebox.showerror("Provider setup", str(exc))
                return
        model = self.model_var.get().strip() or default_model(provider)
        rpm = max(1, self.rpm_var.get() or 1)
        default_context = resolve_default_context(self.default_context_var.get(), self.app_root)
        translator = None
        if not dry_run:
            try:
                translator = AITranslator(
                    model,
                    rpm,
                    default_context=default_context,
                    log_callback=self._log_ai_request,
                    provider=provider,
                )
            except Exception as exc:
                messagebox.showerror("Provider error", f"Failed to initialize translator: {exc}")
                return
        include_ai_comment = self.ai_comment_var.get()
        total_steps = len(self.selected_files) * len(languages)
        self._reset_progress(total_steps)
        self.translate_btn.config(state="disabled")
        thread = threading.Thread(
            target=self._run_translation,
            args=(
                list(self.selected_files),
                languages,
                output_dir,
                self.compile_var.get(),
                max_entries,
                plural_overrides,
                translator,
                dry_run,
                include_ai_comment,
            ),
            daemon=True,
        )
        thread.start()

    def _run_translation(
        self,
        files: List[Path],
        languages: List[str],
        output_dir: Path,
        compile_mo: bool,
        max_entries: int,
        plural_overrides: Dict[str, str],
        translator: Optional[AITranslator],
        dry_run: bool,
        include_ai_comment: bool,
    ) -> None:
        try:
            for pot_path in files:
                if not pot_path.exists():
                    self.log(f"[error] {pot_path}: file not found.")
                    self._increment_progress(step=len(languages))
                    continue
                self.log(f"[start] {pot_path.name}")
                try:
                    results = translate_pot_template(
                        pot_path=pot_path,
                        languages=languages,
                        translator=translator,
                        output_root=output_dir,
                        compile_mo=compile_mo,
                        max_entries=max_entries,
                        plural_overrides=plural_overrides,
                        include_ai_comment=include_ai_comment,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[error] {pot_path.name}: {exc}")
                    self._increment_progress(step=len(languages))
                    continue
                if not results:
                    self.log(f"[skip] {pot_path.name}: no translatable entries.")
                    self._increment_progress(step=len(languages))
                    continue
                for res in results:
                    self.log(
                        f"[{res.language}] {pot_path.name}: translated {res.translated_count} str "
                        f"({res.changed_count} changed), saved {res.po_path} (locale {res.locale})"
                    )
                    if res.mo_path:
                        self.log(f"  compiled {res.mo_path} (locale {res.locale})")
                    self._increment_progress()
        finally:
            self._on_done()

    def _log_ai_request(
        self,
        locale: str,
        entry_label: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
    ) -> None:
        snippet = entry_label.strip().splitlines()[0] if entry_label else "<no msgid>"
        if len(snippet) > 80:
            snippet = snippet[:77] + "…"
        self.log(f"[{self.provider_var.get()}] {locale}: {snippet}")
        self.log(f"  system: {system_prompt}")
        self.log(f"  user: {user_prompt}")
        self.log(f"  response: {response}")

    def _parse_plural_overrides(self, text: str) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        for chunk in text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Invalid override '{chunk}', expected lang=expr")
            lang, expr = chunk.split("=", 1)
            overrides[lang.strip().lower()] = expr.strip()
        return overrides

    def _on_done(self) -> None:
        self.translate_btn.config(state="normal")
        self._set_progress_complete()
        self.log("Translation run complete.")

    def _reset_progress(self, total: int) -> None:
        self.progress_total = total
        self.progress_done = 0
        self.progress_bar.configure(maximum=max(total, 1), value=0)
        self._update_progress_label()

        # Taskbar progress reset
        hwnd = self.taskbar.get_hwnd(self.root)
        self.taskbar.set_progress_state(hwnd, TBPFLAG.TBPF_NORMAL)
        self.taskbar.set_progress_value(hwnd, 0, max(total, 1))

    def _increment_progress(self, step: int = 1) -> None:
        self.progress_done = min(self.progress_done + step, self.progress_total or self.progress_done + step)
        self.progress_bar.configure(value=self.progress_done)
        self._update_progress_label()

    def _set_progress_complete(self) -> None:
        if self.progress_total:
            self.progress_done = self.progress_total
            self.progress_bar.configure(value=self.progress_total)
            self._update_progress_label()

    def _update_progress_label(self) -> None:
        percent = int((self.progress_done / self.progress_total) * 100) if self.progress_total else 0
        self.progress_label.configure(text=f"Progress: {percent}% ({self.progress_done}/{self.progress_total})")

        # Taskbar progress update
        hwnd = self.taskbar.get_hwnd(self.root)
        self.taskbar.set_progress_value(hwnd, self.progress_done, self.progress_total)

        self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    PotTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
