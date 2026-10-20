#!/usr/bin/env python3
"""A small Tkinter interface for the D2R model and texture conversion scripts."""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = os.path.dirname(os.path.abspath(__file__))
TEXTURE_FORMATS = [
    "bc1", "bc1_srgb", "bc2", "bc2_srgb", "bc3", "bc3_srgb",
    "bc4", "bc4_snorm", "bc5", "bc5_snorm", "bc6h", "bc7",
    "bc7_srgb", "rgba8", "rgba8_srgb", "bgra8", "bgra8_srgb",
]


class ModelToolGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("D2R Model Tool")
        self.geometry("760x610")
        self.minsize(620, 500)
        self.events = queue.Queue()
        self.run_buttons = []

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        self._build_model_export_tab()
        self._build_model_import_tab()
        self._build_texture_tab()
        self._build_log()
        self.after(100, self._drain_events)

    def _path_row(self, parent, label, variable, row, kind, filetypes):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        if kind == "save":
            command = lambda: self._save_path(variable, filetypes)
        else:
            command = lambda: self._open_path(variable, filetypes)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, padx=(8, 0), pady=6)
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _open_path(variable, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    @staticmethod
    def _save_path(variable, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _build_model_export_tab(self):
        tab = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(tab, text="Model to GLB")
        self.model_in = tk.StringVar()
        self.model_glb_out = tk.StringVar()
        self.model_export_meters = tk.BooleanVar()
        self._path_row(tab, "Input model", self.model_in, 0, "open", [("D2R model", "*.model"), ("All files", "*.*")])
        self._path_row(tab, "Output GLB", self.model_glb_out, 1, "save", [("glTF Binary", "*.glb")])
        self.model_in.trace_add("write", self._default_output(self.model_glb_out, ".glb"))
        ttk.Checkbutton(tab, text="Convert units to meters", variable=self.model_export_meters).grid(
            row=2, column=1, sticky="w", pady=(8, 4))
        self._add_run_button(tab, "Convert to GLB", self._model_export_command, row=3)

    def _build_model_import_tab(self):
        tab = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(tab, text="GLB to Model")
        self.glb_in = tk.StringVar()
        self.template_model = tk.StringVar()
        self.model_out = tk.StringVar()
        self.import_meters = tk.BooleanVar()
        self.force_size = tk.BooleanVar()
        self.keep_node_transform = tk.BooleanVar()
        self.mesh_filter = tk.StringVar()
        self._path_row(tab, "Edited GLB", self.glb_in, 0, "open", [("glTF Binary", "*.glb"), ("All files", "*.*")])
        self._path_row(tab, "Original model template", self.template_model, 1, "open", [("D2R model", "*.model"), ("All files", "*.*")])
        self._path_row(tab, "Output model", self.model_out, 2, "save", [("D2R model", "*.model")])
        self.template_model.trace_add("write", self._default_output(self.model_out, "_edited.model"))

        ttk.Label(tab, text="Meshes (optional)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(tab, textvariable=self.mesh_filter).grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Label(tab, text="Comma-separated names").grid(row=3, column=2, sticky="w", padx=(8, 0), pady=6)
        options = ttk.Frame(tab)
        options.grid(row=4, column=1, columnspan=2, sticky="w", pady=(8, 4))
        ttk.Checkbutton(options, text="GLB uses meters", variable=self.import_meters).pack(anchor="w", pady=2)
        ttk.Checkbutton(options, text="Keep skinned mesh node transforms", variable=self.keep_node_transform).pack(anchor="w", pady=2)
        ttk.Checkbutton(options, text="Force unusual size changes", variable=self.force_size).pack(anchor="w", pady=2)
        self._add_run_button(tab, "Write model", self._model_import_command, row=5)

    def _build_texture_tab(self):
        tab = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(tab, text="Textures")
        self.texture_action = tk.StringVar(value="Inspect texture")
        selector = ttk.Frame(tab)
        selector.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Label(selector, text="Action").pack(side="left", padx=(0, 8))
        action_box = ttk.Combobox(selector, textvariable=self.texture_action, state="readonly",
                                  values=("Inspect texture", "Texture to DDS", "DDS to texture"))
        action_box.pack(side="left", fill="x", expand=True)
        action_box.bind("<<ComboboxSelected>>", self._render_texture_fields)

        self.texture_fields = ttk.Frame(tab)
        self.texture_fields.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.texture_in = tk.StringVar()
        self.texture_out = tk.StringVar()
        self.texture_template = tk.StringVar()
        self.texture_format = tk.StringVar(value="Auto")
        self.texture_fmt_word = tk.StringVar()
        self.texture_dx10 = tk.BooleanVar()
        self._render_texture_fields()

    def _render_texture_fields(self, _event=None):
        for child in self.texture_fields.winfo_children():
            child.destroy()
        action = self.texture_action.get()
        if action == "Inspect texture":
            self._path_row(self.texture_fields, "Input texture", self.texture_in, 0, "open",
                           [("D2R texture", "*.texture"), ("All files", "*.*")])
        elif action == "Texture to DDS":
            self._path_row(self.texture_fields, "Input texture", self.texture_in, 0, "open",
                           [("D2R texture", "*.texture"), ("All files", "*.*")])
            self._path_row(self.texture_fields, "Output DDS", self.texture_out, 1, "save", [("DDS image", "*.dds")])
            self.texture_in.trace_add("write", self._default_output(self.texture_out, ".dds"))
            ttk.Label(self.texture_fields, text="Format").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
            ttk.Combobox(self.texture_fields, textvariable=self.texture_format, state="readonly",
                         values=("Auto", *TEXTURE_FORMATS)).grid(row=2, column=1, sticky="ew", pady=6)
            ttk.Checkbutton(self.texture_fields, text="Use DX10 DDS header", variable=self.texture_dx10).grid(
                row=3, column=1, sticky="w", pady=6)
        else:
            self._path_row(self.texture_fields, "Input DDS", self.texture_in, 0, "open",
                           [("DDS image", "*.dds"), ("All files", "*.*")])
            self._path_row(self.texture_fields, "Output texture", self.texture_out, 1, "save",
                           [("D2R texture", "*.texture")])
            self.texture_in.trace_add("write", self._default_output(self.texture_out, ".texture"))
            self._path_row(self.texture_fields, "Original template (optional)", self.texture_template, 2, "open",
                           [("D2R texture", "*.texture"), ("All files", "*.*")])
            ttk.Label(self.texture_fields, text="Format word (optional)").grid(
                row=3, column=0, sticky="w", padx=(0, 8), pady=6)
            ttk.Entry(self.texture_fields, textvariable=self.texture_fmt_word).grid(
                row=3, column=1, sticky="ew", pady=6)
            ttk.Label(self.texture_fields, text="Example: 0x0002003E").grid(
                row=3, column=2, sticky="w", padx=(8, 0), pady=6)
        self.texture_fields.columnconfigure(1, weight=1)
        self._add_run_button(self.texture_fields, "Run texture action", self._texture_command,
                             row=len(self.texture_fields.winfo_children()) + 1)

    @staticmethod
    def _default_output(output_var, extension):
        def update(*_args):
            current = output_var.get()
            if not current:
                return
            if extension == ".glb":
                source = None
            else:
                source = None
        return update

    def _add_run_button(self, parent, label, command_builder, row):
        button = ttk.Button(parent, text=label, command=lambda: self._run(command_builder))
        button.grid(row=row, column=1, sticky="w", pady=(14, 6))
        self.run_buttons.append(button)

    def _model_export_command(self):
        source, output = self.model_in.get().strip(), self.model_glb_out.get().strip()
        self._require_paths(source, output)
        command = [sys.executable, os.path.join(ROOT, "d2r_model_to_glb.py"), source, output]
        if self.model_export_meters.get():
            command.append("--meters")
        return command

    def _model_import_command(self):
        glb, template, output = (value.get().strip() for value in
                                 (self.glb_in, self.template_model, self.model_out))
        self._require_paths(glb, template, output)
        command = [sys.executable, os.path.join(ROOT, "d2r_glb_to_model.py"), glb, template, output]
        if self.import_meters.get():
            command.append("--meters")
        names = self.mesh_filter.get().strip()
        if names:
            command.extend(("--only", names))
        if self.keep_node_transform.get():
            command.append("--keep-node-transform")
        if self.force_size.get():
            command.append("--force")
        return command

    def _texture_command(self):
        source = self.texture_in.get().strip()
        action = self.texture_action.get()
        self._require_paths(source)
        script = os.path.join(ROOT, "d2r_texture.py")
        if action == "Inspect texture":
            return [sys.executable, script, "info", source]
        output = self.texture_out.get().strip()
        self._require_paths(output)
        if action == "Texture to DDS":
            command = [sys.executable, script, "to-dds", source, output]
            fmt = self.texture_format.get()
            if fmt != "Auto":
                command.extend(("--format", fmt))
            if self.texture_dx10.get():
                command.append("--dx10")
            return command
        command = [sys.executable, script, "from-dds", source, output]
        template = self.texture_template.get().strip()
        fmt_word = self.texture_fmt_word.get().strip()
        if template:
            command.extend(("--template", template))
        if fmt_word:
            command.extend(("--fmt-word", fmt_word))
        return command

    @staticmethod
    def _require_paths(*paths):
        if any(not path for path in paths):
            raise ValueError("Choose all required input and output files first.")

    def _run(self, command_builder):
        try:
            command = command_builder()
        except ValueError as error:
            messagebox.showerror("Missing file", str(error))
            return
        self._log("\n> " + subprocess.list2cmdline(command) + "\n")
        for button in self.run_buttons:
            button.configure(state="disabled")
        threading.Thread(target=self._execute, args=(command,), daemon=True).start()

    def _execute(self, command):
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in process.stdout:
                self.events.put(("output", line))
            code = process.wait()
            self.events.put(("finished", code))
        except OSError as error:
            self.events.put(("output", f"Could not start command: {error}\n"))
            self.events.put(("finished", 1))

    def _drain_events(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "output":
                self._log(value)
            else:
                self._log(f"\nFinished with exit code {value}.\n")
                for button in self.run_buttons:
                    button.configure(state="normal")
        self.after(100, self._drain_events)

    def _build_log(self):
        frame = ttk.LabelFrame(self, text="Output", padding=8)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log = tk.Text(frame, height=9, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    ModelToolGUI().mainloop()
