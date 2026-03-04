"""
olmOCR Training Data Extractor
================================
Single-file tkinter GUI for building olmOCR fine-tuning pairs.

For each PDF page it creates:
    <output_dir>/<stem>_page<N>.pdf   ← single-page PDF
    <output_dir>/<stem>_page<N>.md    ← YAML front matter + extracted text

Usage:
    python olmocr_trainer.py

No vLLM server needed. The VLM runs locally (optional — you can also
type/paste text manually without loading any model).
"""

import base64
import gc
import hashlib
import re
import shutil
import threading
from io import BytesIO
from pathlib import Path
from tkinter import (
    BooleanVar, END, IntVar, StringVar,
    filedialog, messagebox, scrolledtext, ttk,
)
import tkinter as tk
from datetime import datetime

from PIL import Image, ImageTk

# ── olmocr native helpers (graceful fallback) ────────────────────────────────
try:
    from olmocr.data.renderpdf import render_pdf_to_base64png as _render_b64
    _HAS_RENDER = True
except ImportError:
    _HAS_RENDER = False

try:
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    _OCR_PROMPT = build_no_anchoring_v4_yaml_prompt()
except ImportError:
    _OCR_PROMPT = (
        "Attached is one page of a document. Return the plain text as if reading "
        "it naturally. Convert equations to LaTeX and tables to HTML.\n"
        "Return as markdown with a YAML front matter block containing: "
        "primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram."
    )

try:
    from pypdf import PdfReader, PdfWriter
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

# ── PDF utilities ─────────────────────────────────────────────────────────────

def count_pages(pdf_path: str) -> int:
    if _HAS_PYPDF:
        try:
            return len(PdfReader(pdf_path).pages)
        except Exception:
            pass
    try:
        from pdf2image import convert_from_path
        return len(convert_from_path(pdf_path, dpi=72))
    except Exception:
        return 0


def render_page(pdf_path: str, page_num: int) -> Image.Image:
    """Render a single PDF page (1-indexed) → PIL Image."""
    if _HAS_RENDER:
        try:
            b64 = _render_b64(pdf_path, page_num, target_longest_image_dim=1288)
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            pass
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=150, first_page=page_num, last_page=page_num)
    return pages[0].convert("RGB") if pages else None


def extract_single_page_pdf(src_pdf: str, page_num: int, dest: str) -> bool:
    """Write page *page_num* (1-indexed) of *src_pdf* to *dest* as a 1-page PDF."""
    if not _HAS_PYPDF:
        raise RuntimeError("pypdf is required: pip install pypdf")
    reader = PdfReader(src_pdf)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num - 1])
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        writer.write(fh)
    return True


# ── YAML helpers ──────────────────────────────────────────────────────────────

YAML_DEFAULTS = {
    "primary_language": "en",
    "is_rotation_valid": True,
    "rotation_correction": 0,
    "is_table": False,
    "is_diagram": False,
}


def parse_yaml_response(text: str) -> tuple[dict, str]:
    """Split olmOCR YAML front-matter response into (meta_dict, body_text)."""
    meta = dict(YAML_DEFAULTS)
    m = re.search(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL | re.MULTILINE)
    if not m:
        return meta, text.strip()
    fm, body = m.group(1), m.group(2).strip()
    for line in fm.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "primary_language":
            meta[k] = None if v.lower() in ("null", "none", "") else v
        elif k in ("is_rotation_valid", "is_table", "is_diagram"):
            meta[k] = v.lower() in ("true", "yes", "1")
        elif k == "rotation_correction":
            try:
                meta[k] = int(v)
            except ValueError:
                pass
    return meta, body


def build_md(meta: dict, body: str) -> str:
    """Serialise meta + body into a .md string with YAML front matter."""
    lang = meta.get("primary_language")
    return "\n".join([
        "---",
        f"primary_language: {'null' if lang is None else lang}",
        f"is_rotation_valid: {str(meta.get('is_rotation_valid', True)).lower()}",
        f"rotation_correction: {meta.get('rotation_correction', 0)}",
        f"is_table: {str(meta.get('is_table', False)).lower()}",
        f"is_diagram: {str(meta.get('is_diagram', False)).lower()}",
        "---",
        body,
    ])


# ── VLM wrapper ───────────────────────────────────────────────────────────────

class VLM:
    """Lazy-loaded local vision-language model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self, log):
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        log(f"Loading processor: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        log("Loading model weights (first run downloads ~14 GB) …")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.loaded = True
        log("VLM ready.")

    def run(self, image: Image.Image, temperature: float = 0.1) -> str:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img = image.convert("RGB")
        # Resize to 1288 on longest side (pipeline default)
        w, h = img.size
        scale = 1288 / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": _OCR_PROMPT},
        ]}]
        text_in = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_in], images=[img],
                                padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs, temperature=temperature,
                max_new_tokens=2048, do_sample=(temperature > 0.05))

        result = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0]
        del inputs, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


# ── Main application ──────────────────────────────────────────────────────────

class TrainerApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("olmOCR Training Data Extractor")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 650)

        # State
        self.pages: list[tuple[str, int]] = []   # [(pdf_path, page_num), …]
        self.current_idx: int = -1
        self.page_images: dict[tuple, Image.Image] = {}
        self.output_dir: str = ""
        self.vlm: VLM | None = None
        self._stop = threading.Event()
        self._photo = None   # keep PhotoImage alive

        # Editable YAML fields
        self.lang_var = StringVar(value="en")
        self.rot_valid_var = BooleanVar(value=True)
        self.rot_correction_var = IntVar(value=0)
        self.is_table_var = BooleanVar(value=False)
        self.is_diagram_var = BooleanVar(value=False)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=6, pady=4)
        self._build_toolbar(toolbar)

        # Main 3-panel layout
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = ttk.Frame(paned, width=220)
        paned.add(left, weight=1)

        middle = ttk.Frame(paned, width=560)
        paned.add(middle, weight=3)

        right = ttk.Frame(paned, width=420)
        paned.add(right, weight=2)

        self._build_left(left)
        self._build_middle(middle)
        self._build_right(right)

        # Status bar
        self.status_var = StringVar(value="Ready — select PDFs to begin.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, side=tk.BOTTOM, padx=6, pady=2)

    def _build_toolbar(self, parent):
        ttk.Button(parent, text="📂 Add PDFs",
                   command=self._add_pdfs).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="📁 Add Folder",
                   command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="🗑 Clear List",
                   command=self._clear_list).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(parent, text="VLM model:").pack(side=tk.LEFT)
        self.model_var = StringVar(value="allenai/olmOCR-7B-0924-preview")
        ttk.Entry(parent, textvariable=self.model_var, width=34).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(parent, text="Load VLM",
                   command=self._load_vlm).pack(side=tk.LEFT, padx=2)
        self.vlm_label = ttk.Label(parent, text="(not loaded)",
                                   foreground="gray")
        self.vlm_label.pack(side=tk.LEFT, padx=6)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(parent, text="Output:").pack(side=tk.LEFT)
        self.out_var = StringVar(value="")
        ttk.Entry(parent, textvariable=self.out_var, width=28).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(parent, text="Browse",
                   command=self._pick_output).pack(side=tk.LEFT, padx=2)

    def _build_left(self, parent):
        ttk.Label(parent, text="Pages", font=("Arial", 10, "bold")).pack(pady=4)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        self.page_list = tk.Listbox(
            frame, yscrollcommand=sb.set,
            font=("Consolas", 8), activestyle="dotbox",
            selectmode=tk.SINGLE)
        sb.config(command=self.page_list.yview)
        self.page_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.page_list.bind("<<ListboxSelect>>", self._on_page_select)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="◀", command=self._prev_page,
                   width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="▶", command=self._next_page,
                   width=4).pack(side=tk.LEFT, padx=2)

    def _build_middle(self, parent):
        ttk.Label(parent, text="Page Preview",
                  font=("Arial", 10, "bold")).pack(pady=4)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#888")
        vsb = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                             command=self.canvas.yview)
        hsb = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL,
                             command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<MouseWheel>", self._scroll)
        self.canvas.bind("<Button-4>",  self._scroll)
        self.canvas.bind("<Button-5>",  self._scroll)

    def _build_right(self, parent):
        # ── YAML fields ──
        meta_frame = ttk.LabelFrame(parent, text="YAML Metadata", padding=6)
        meta_frame.pack(fill=tk.X, padx=4, pady=4)

        def _row(label, widget_fn, **kw):
            r = ttk.Frame(meta_frame)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=20, anchor="w").pack(side=tk.LEFT)
            widget_fn(r, **kw).pack(side=tk.LEFT, fill=tk.X, expand=True)

        _row("primary_language:", ttk.Entry, textvariable=self.lang_var)
        _row("is_rotation_valid:", ttk.Checkbutton, variable=self.rot_valid_var)
        _row("rotation_correction:", ttk.Combobox,
             textvariable=self.rot_correction_var,
             values=[0, 90, 180, 270], width=8, state="readonly")
        _row("is_table:", ttk.Checkbutton, variable=self.is_table_var)
        _row("is_diagram:", ttk.Checkbutton, variable=self.is_diagram_var)

        # ── Text editor ──
        text_frame = ttk.LabelFrame(parent, text="Extracted Text (Markdown)",
                                    padding=4)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.text_editor = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.text_editor.pack(fill=tk.BOTH, expand=True)

        # ── Action buttons ──
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(btn_frame, text="🤖 Extract (VLM)",
                   command=self._extract_current).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="💾 Save Page",
                   command=self._save_current).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        batch_frame = ttk.Frame(parent)
        batch_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(batch_frame, text="⚡ Extract & Save ALL",
                   command=self._batch_all).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(batch_frame, text="⏹ Stop",
                   command=lambda: self._stop.set()).pack(
            side=tk.LEFT, padx=2)

        # Progress
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill=tk.X, padx=4, pady=2)

        # Log
        log_frame = ttk.LabelFrame(parent, text="Log", padding=4)
        log_frame.pack(fill=tk.X, padx=4, pady=2)
        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=5, font=("Consolas", 8), wrap=tk.WORD)
        self.log_box.pack(fill=tk.X)

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg: str):
        self.log_box.insert(END, msg + "\n")
        self.log_box.see(END)
        self.status_var.set(msg)

    # ── File loading ─────────────────────────────────────────────────────────

    def _add_pdfs(self):
        paths = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        for path in paths:
            n = count_pages(path)
            if n == 0:
                self._log(f"Could not read {Path(path).name} — skipped")
                continue
            for pg in range(1, n + 1):
                self.pages.append((path, pg))
                label = f"{Path(path).stem}  p{pg}/{n}"
                self.page_list.insert(END, label)
            self._log(f"Added {Path(path).name} ({n} pages)")

        if self.pages and self.current_idx == -1:
            self._select_page(0)

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder — all PDFs inside will be added")
        if not folder:
            return
        pdf_paths = sorted(Path(folder).rglob("*.pdf"))
        if not pdf_paths:
            self._log(f"No PDFs found in {folder}")
            return
        self._log(f"Found {len(pdf_paths)} PDF(s) in {Path(folder).name} …")
        for path in pdf_paths:
            n = count_pages(str(path))
            if n == 0:
                self._log(f"  Could not read {path.name} — skipped")
                continue
            for pg in range(1, n + 1):
                self.pages.append((str(path), pg))
                self.page_list.insert(END, f"{path.stem}  p{pg}/{n}")
            self._log(f"  Added {path.name} ({n} pages)")
        if self.pages and self.current_idx == -1:
            self._select_page(0)

    def _clear_list(self):
        self.pages.clear()
        self.page_images.clear()
        self.page_list.delete(0, END)
        self.canvas.delete("all")
        self.current_idx = -1
        self._log("List cleared.")

    # ── Page navigation ───────────────────────────────────────────────────────

    def _on_page_select(self, _event=None):
        sel = self.page_list.curselection()
        if sel:
            self._select_page(sel[0])

    def _select_page(self, idx: int):
        if idx < 0 or idx >= len(self.pages):
            return
        self.current_idx = idx
        self.page_list.selection_clear(0, END)
        self.page_list.selection_set(idx)
        self.page_list.see(idx)
        self._load_preview(idx)

    def _prev_page(self):
        self._select_page(self.current_idx - 1)

    def _next_page(self):
        self._select_page(self.current_idx + 1)

    def _load_preview(self, idx: int):
        pdf_path, page_num = self.pages[idx]
        key = (pdf_path, page_num)
        self.status_var.set(f"Loading {Path(pdf_path).name} page {page_num} …")

        def _render():
            if key not in self.page_images:
                try:
                    img = render_page(pdf_path, page_num)
                    self.page_images[key] = img
                except Exception as exc:
                    self.root.after(0, lambda: self._log(f"Render error: {exc}"))
                    return
            self.root.after(0, self._show_preview, key)

        threading.Thread(target=_render, daemon=True).start()

    def _show_preview(self, key):
        img = self.page_images.get(key)
        if img is None:
            return
        # Fit to canvas width
        cw = self.canvas.winfo_width() or 550
        scale = min(1.0, cw / img.width)
        disp = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(disp)
        self._photo = photo   # prevent GC
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.configure(
            scrollregion=(0, 0, disp.width, disp.height))
        pdf_path, page_num = self.pages[self.current_idx]
        self.status_var.set(
            f"{Path(pdf_path).name}  —  page {page_num}")

    def _scroll(self, event):
        delta = -1 if (event.num == 5 or event.delta < 0) else 1
        self.canvas.yview_scroll(delta, "units")

    # ── VLM ──────────────────────────────────────────────────────────────────

    def _load_vlm(self):
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showwarning("No model", "Enter a model name first.")
            return
        self.vlm_label.config(text="Loading …", foreground="orange")

        def _load():
            try:
                vlm = VLM(model_name)
                vlm.load(self._log)
                self.vlm = vlm
                self.root.after(0, lambda: self.vlm_label.config(
                    text=f"✓ {Path(model_name).name}",
                    foreground="green"))
            except Exception as exc:
                self.root.after(0, lambda: self.vlm_label.config(
                    text="Load failed", foreground="red"))
                self._log(f"VLM load error: {exc}")

        threading.Thread(target=_load, daemon=True).start()

    def _run_vlm_on(self, idx: int) -> tuple[dict, str]:
        """Run VLM on page *idx*, return (meta, body)."""
        if self.vlm is None or not self.vlm.loaded:
            raise RuntimeError("VLM not loaded. Click 'Load VLM' first.")
        pdf_path, page_num = self.pages[idx]
        key = (pdf_path, page_num)
        if key not in self.page_images:
            self.page_images[key] = render_page(pdf_path, page_num)
        raw = self.vlm.run(self.page_images[key])
        return parse_yaml_response(raw)

    def _extract_current(self):
        if self.current_idx < 0:
            self._log("No page selected.")
            return

        def _run():
            try:
                self._log(f"Extracting page {self.current_idx + 1} …")
                meta, body = self._run_vlm_on(self.current_idx)
                self.root.after(0, self._fill_editors, meta, body)
                self._log("Extraction done.")
            except Exception as exc:
                self._log(f"Error: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def _fill_editors(self, meta: dict, body: str):
        lang = meta.get("primary_language")
        self.lang_var.set("" if lang is None else str(lang))
        self.rot_valid_var.set(bool(meta.get("is_rotation_valid", True)))
        self.rot_correction_var.set(int(meta.get("rotation_correction", 0)))
        self.is_table_var.set(bool(meta.get("is_table", False)))
        self.is_diagram_var.set(bool(meta.get("is_diagram", False)))
        self.text_editor.delete(1.0, END)
        self.text_editor.insert(1.0, body)

    # ── Save ─────────────────────────────────────────────────────────────────

    def _meta_from_ui(self) -> dict:
        lang = self.lang_var.get().strip()
        return {
            "primary_language": lang or None,
            "is_rotation_valid": self.rot_valid_var.get(),
            "rotation_correction": self.rot_correction_var.get(),
            "is_table": self.is_table_var.get(),
            "is_diagram": self.is_diagram_var.get(),
        }

    def _resolve_output(self) -> str:
        out = self.out_var.get().strip()
        if not out:
            out = str(Path.cwd() / "training_output")
        Path(out).mkdir(parents=True, exist_ok=True)
        return out

    def _save_current(self):
        if self.current_idx < 0:
            self._log("No page selected.")
            return
        try:
            out = self._resolve_output()
            self._save_page(self.current_idx, out)
            self._log(f"Saved page {self.current_idx + 1}.")
        except Exception as exc:
            self._log(f"Save error: {exc}")

    def _save_page(self, idx: int, out_dir: str):
        pdf_path, page_num = self.pages[idx]
        stem = f"{Path(pdf_path).stem}_page{page_num}"
        meta = self._meta_from_ui() if idx == self.current_idx else YAML_DEFAULTS
        body = self.text_editor.get(1.0, END).strip() if idx == self.current_idx else ""
        # Write .md
        md_path = str(Path(out_dir) / f"{stem}.md")
        Path(md_path).write_text(build_md(meta, body), encoding="utf-8")
        # Write single-page .pdf
        pdf_out = str(Path(out_dir) / f"{stem}.pdf")
        extract_single_page_pdf(pdf_path, page_num, pdf_out)

    # ── Batch ─────────────────────────────────────────────────────────────────

    def _batch_all(self):
        if not self.pages:
            self._log("No pages loaded.")
            return
        if self.vlm is None or not self.vlm.loaded:
            if not messagebox.askyesno(
                "VLM not loaded",
                "VLM is not loaded. Save pages with blank text?\n\n"
                "Click No to cancel and load the VLM first.",
            ):
                return

        self._stop.clear()

        def _run():
            out = self._resolve_output()
            total = len(self.pages)
            self.root.after(0, lambda: self.progress.configure(
                maximum=total, value=0))

            for idx in range(total):
                if self._stop.is_set():
                    self._log("Stopped.")
                    break

                pdf_path, page_num = self.pages[idx]
                self._log(
                    f"[{idx + 1}/{total}] {Path(pdf_path).name} p{page_num}")
                self.root.after(0, self._select_page, idx)

                try:
                    if self.vlm and self.vlm.loaded:
                        meta, body = self._run_vlm_on(idx)
                        self.root.after(0, self._fill_editors, meta, body)
                        import time; time.sleep(0.05)  # let UI refresh
                    else:
                        meta, body = YAML_DEFAULTS.copy(), ""

                    # Save directly (don't rely on UI state for non-current pages)
                    stem = f"{Path(pdf_path).stem}_page{page_num}"
                    md_path = str(Path(out) / f"{stem}.md")
                    pdf_out = str(Path(out) / f"{stem}.pdf")
                    Path(md_path).write_text(
                        build_md(meta, body), encoding="utf-8")
                    extract_single_page_pdf(pdf_path, page_num, pdf_out)

                    self._mark_done(idx)
                except Exception as exc:
                    self._log(f"  Error: {exc}")

                self.root.after(0, self.progress.configure,
                                {"value": idx + 1})

            self._log(f"Done. Output: {out}")

        threading.Thread(target=_run, daemon=True).start()

    def _mark_done(self, idx: int):
        self.root.after(0, self._mark_done_ui, idx)

    def _mark_done_ui(self, idx: int):
        label = self.page_list.get(idx)
        if not label.startswith("✓"):
            self.page_list.delete(idx)
            self.page_list.insert(idx, "✓ " + label)
            self.page_list.itemconfig(idx, fg="green")

    # ── Output dir ────────────────────────────────────────────────────────────

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self.out_var.set(folder)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = TrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
