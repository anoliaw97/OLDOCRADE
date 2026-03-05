"""
olmOCR Training Data Extractor
================================
Single-file tkinter GUI for building olmOCR fine-tuning pairs.

For each PDF page it creates:
    <output_dir>/train/<stem>_page<N>.pdf   ← single-page PDF (training split)
    <output_dir>/train/<stem>_page<N>.md    ← YAML front matter + extracted text
    <output_dir>/eval/<stem>_page<N>.pdf    ← (eval split)
    <output_dir>/eval/<stem>_page<N>.md

After extraction:
  - "Validate Dataset"  — checks all PDF/MD pairs are correct
  - "Generate Config"   — writes finetune_config.yaml ready for olmocr/train/train.py
  - "Launch Training"   — runs python -m olmocr.train.train --config ...

Usage:
    python olmocr_trainer.py

No vLLM server needed. The VLM runs locally (optional — you can also
type/paste text manually without loading any model).
"""

import base64
import gc
import json
import re
import subprocess
import sys
import threading
from io import BytesIO
from pathlib import Path
from tkinter import (
    BooleanVar, END, IntVar, StringVar,
    filedialog, messagebox, scrolledtext, ttk,
)
import tkinter as tk

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

REQUIRED_YAML_KEYS = set(YAML_DEFAULTS.keys())


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


# ── Dataset validation helpers ────────────────────────────────────────────────

def _check_md_pair(md_path: Path) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors = []
    pdf_path = md_path.with_suffix(".pdf")
    if not pdf_path.exists():
        errors.append(f"Missing PDF: {pdf_path.name}")
        return errors
    # Single-page check
    if _HAS_PYPDF:
        try:
            n = len(PdfReader(str(pdf_path)).pages)
            if n != 1:
                errors.append(f"PDF has {n} pages (must be 1)")
        except Exception as exc:
            errors.append(f"PDF read error: {exc}")
    # YAML front matter check
    try:
        text = md_path.read_text(encoding="utf-8")
        m = re.search(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL | re.MULTILINE)
        if not m:
            errors.append("Missing YAML front matter")
        else:
            found_keys = set()
            for line in m.group(1).splitlines():
                if ":" in line:
                    found_keys.add(line.split(":")[0].strip())
            missing = REQUIRED_YAML_KEYS - found_keys
            if missing:
                errors.append(f"Missing YAML keys: {', '.join(sorted(missing))}")
    except Exception as exc:
        errors.append(f"MD read error: {exc}")
    return errors


# ── Training config generator ─────────────────────────────────────────────────

def _generate_training_config(output_dir: str) -> str:
    """Write finetune_config.yaml to output_dir and return the path."""
    out = Path(output_dir).resolve()
    train_dir = str(out / "train")
    eval_dir = str(out / "eval")
    ckpt_dir = str(out / "checkpoints")

    config = f"""# olmOCR Fine-Tuning Configuration
# Generated by olmocr_trainer.py
# Launch with: python -m olmocr.train.train --config {out / 'finetune_config.yaml'}

project_name: olmocr-finetuning
run_name: my-finetune

model:
  name: allenai/olmOCR-2-7B-1025
  trust_remote_code: true
  torch_dtype: bfloat16
  use_flash_attention: true
  attn_implementation: flash_attention_2
  use_lora: true
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

dataset:
  train:
    - name: my_train
      root_dir: {train_dir}
      pipeline: &basic_pipeline
        - name: FrontMatterParser
          front_matter_class: PageResponse
        - name: FilterOutRotatedDocuments
        - name: PDFRenderer
          target_longest_image_dim: 1288
        - name: NewYamlFinetuningPromptWithNoAnchoring
        - name: FrontMatterOutputFormat
        - name: InstructUserMessages
          prompt_first: true
        - name: Tokenizer
          masking_index: -100
          end_of_message_token: "<|im_end|>"
  eval:
    - name: my_eval
      root_dir: {eval_dir}
      pipeline: *basic_pipeline

training:
  output_dir: {ckpt_dir}
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: false
  collator_max_token_len: 8192
  learning_rate: 2.0e-5
  lr_scheduler_type: linear
  warmup_ratio: 0.1
  optim: adamw_torch
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 300
  data_seed: 301
  evaluation_strategy: steps
  eval_steps: 500
  save_strategy: steps
  save_steps: 500
  save_total_limit: 5
  load_best_model_at_end: false
  metric_for_best_model: eval_my_eval_loss
  greater_is_better: false
  report_to:
    - tensorboard
"""
    config_path = out / "finetune_config.yaml"
    config_path.write_text(config, encoding="utf-8")
    return str(config_path)


# ── VLM wrapper ───────────────────────────────────────────────────────────────

class VLM:
    """Lazy-loaded local vision-language model."""

    def __init__(self, model_name: str,
                 processor_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.model_name = model_name
        self.processor_name = processor_name
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self, log):
        if self.loaded:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        log(f"Loading processor: {self.processor_name}")
        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        log(f"Loading model weights: {self.model_name} (first run may download ~8 GB) …")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto"
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
        self.root.geometry("1440x900")
        self.root.minsize(1100, 680)

        # State
        self.pages: list[tuple[str, int]] = []   # [(pdf_path, page_num), …]
        self.page_splits: list[str] = []          # "train" or "eval" per page
        self.page_statuses: list[str] = []        # "○", "✓", "✗" per page
        self.current_idx: int = -1
        self.page_images: dict[tuple, Image.Image] = {}
        self.output_dir: str = ""
        self.vlm: VLM | None = None
        self._stop = threading.Event()
        self._train_proc: subprocess.Popen | None = None
        self._photo = None   # keep PhotoImage alive

        # Editable YAML fields
        self.lang_var = StringVar(value="en")
        self.rot_valid_var = BooleanVar(value=True)
        self.rot_correction_var = IntVar(value=0)
        self.is_table_var = BooleanVar(value=False)
        self.is_diagram_var = BooleanVar(value=False)

        self._build_ui()
        self._build_context_menu()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbars
        tb1 = ttk.Frame(self.root)
        tb1.pack(fill=tk.X, padx=6, pady=(4, 0))
        self._build_toolbar_row1(tb1)

        tb2 = ttk.Frame(self.root)
        tb2.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._build_toolbar_row2(tb2)

        # Main 3-panel layout
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = ttk.Frame(paned, width=240)
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

    def _build_toolbar_row1(self, parent):
        """Row 1: file loading, VLM, output dir."""
        ttk.Button(parent, text="📂 Add PDFs",
                   command=self._add_pdfs).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="📁 Add Folder",
                   command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="🗑 Clear List",
                   command=self._clear_list).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(parent, text="Model:").pack(side=tk.LEFT)
        self.model_var = StringVar(value="allenai/olmOCR-2-7B-1025-FP8")
        ttk.Entry(parent, textvariable=self.model_var, width=30).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(parent, text="Processor:").pack(side=tk.LEFT)
        self.processor_var = StringVar(value="Qwen/Qwen2.5-VL-7B-Instruct")
        ttk.Entry(parent, textvariable=self.processor_var, width=26).pack(
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

    def _build_toolbar_row2(self, parent):
        """Row 2: split assignment, validation, config, training."""
        ttk.Label(parent, text="Split:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(parent, text="⚖ Auto 80/20",
                   command=self._auto_split).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="All → train",
                   command=lambda: self._set_all_splits("train")).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(parent, text="All → eval",
                   command=lambda: self._set_all_splits("eval")).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(parent, text="✔ Validate Dataset",
                   command=self._validate_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="⚙ Generate Config",
                   command=self._generate_config).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        self.launch_btn = ttk.Button(parent, text="▶ Launch Training",
                                     command=self._launch_training)
        self.launch_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(parent, text="■ Stop Training",
                   command=self._stop_training).pack(side=tk.LEFT, padx=2)

    def _build_left(self, parent):
        ttk.Label(parent, text="Pages", font=("Arial", 10, "bold")).pack(pady=4)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL)

        cols = ("file", "split", "st")
        self.page_tree = ttk.Treeview(
            frame, columns=cols, show="headings",
            yscrollcommand=sb.set, selectmode="browse")
        self.page_tree.heading("file",  text="File")
        self.page_tree.heading("split", text="Split")
        self.page_tree.heading("st",    text="St.")
        self.page_tree.column("file",  width=145, stretch=True)
        self.page_tree.column("split", width=42,  anchor="center", stretch=False)
        self.page_tree.column("st",    width=24,  anchor="center", stretch=False)

        # Tag colours
        self.page_tree.tag_configure("done",    foreground="green")
        self.page_tree.tag_configure("error",   foreground="red")
        self.page_tree.tag_configure("normal",  foreground="")

        sb.config(command=self.page_tree.yview)
        self.page_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.page_tree.bind("<<TreeviewSelect>>", self._on_page_select)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="◀", command=self._prev_page,
                   width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="▶", command=self._next_page,
                   width=4).pack(side=tk.LEFT, padx=2)

    def _build_context_menu(self):
        """Right-click menu on the page tree for split assignment."""
        self._ctx_menu = tk.Menu(self.root, tearoff=0)
        self._ctx_menu.add_command(
            label="Set → train",
            command=lambda: self._set_split_selected("train"))
        self._ctx_menu.add_command(
            label="Set → eval",
            command=lambda: self._set_split_selected("eval"))
        self.page_tree.bind("<Button-3>", self._show_context_menu)

    def _show_context_menu(self, event):
        iid = self.page_tree.identify_row(event.y)
        if iid:
            self.page_tree.selection_set(iid)
            self._ctx_menu.post(event.x_root, event.y_root)

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

    # ── Treeview helpers ─────────────────────────────────────────────────────

    def _tree_insert(self, idx: int, label: str, split: str, status: str = "○"):
        """Insert a row at position *idx*."""
        self.page_tree.insert(
            "", END, iid=str(idx),
            values=(label, split, status),
            tags=("normal",))

    def _tree_set_split(self, idx: int, split: str):
        self.page_splits[idx] = split
        self.page_tree.set(str(idx), "split", split)

    def _tree_set_status(self, idx: int, status: str):
        self.page_statuses[idx] = status
        self.page_tree.set(str(idx), "st", status)
        tag = "done" if status == "✓" else ("error" if status == "✗" else "normal")
        self.page_tree.item(str(idx), tags=(tag,))

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
                idx = len(self.pages)
                self.pages.append((path, pg))
                self.page_splits.append("train")
                self.page_statuses.append("○")
                label = f"{Path(path).stem}  p{pg}/{n}"
                self._tree_insert(idx, label, "train")
            self._log(f"Added {Path(path).name} ({n} pages)")

        if self.pages and self.current_idx == -1:
            self._select_page(0)

    def _add_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder — all PDFs inside will be added")
        if not folder:
            return
        self.status_var.set("Scanning …")

        def _insert_one(path, n):
            if n == 0:
                self._log_ui(f"  Could not read {path.name} — skipped")
                return
            for pg in range(1, n + 1):
                idx = len(self.pages)
                self.pages.append((str(path), pg))
                self.page_splits.append("train")
                self.page_statuses.append("○")
                self._tree_insert(idx, f"{path.stem}  p{pg}/{n}", "train")
            self._log_ui(f"  Added {path.name} ({n} pages)")

        def _drain(it):
            """Process one PDF per call; reschedule self so the event loop
            can handle user input between each PDF's tree inserts."""
            try:
                path, n = next(it)
            except StopIteration:
                # All PDFs inserted — update status, no auto-preview.
                self.status_var.set(
                    f"Ready — {len(self.pages)} pages total. "
                    "Click a page to preview.")
                self._log_ui("Folder loaded.")
                return
            _insert_one(path, n)
            self.root.after(0, _drain, it)   # yield to event loop, then next

        def _worker():
            pdf_paths = sorted(Path(folder).rglob("*.pdf"))
            if not pdf_paths:
                self.root.after(0, self._log_ui,
                                f"No PDFs found in {Path(folder).name}")
                self.root.after(0, self.status_var.set, "Ready.")
                return
            self.root.after(0, self._log_ui,
                            f"Found {len(pdf_paths)} PDF(s) in "
                            f"{Path(folder).name} — scanning …")
            results = []
            for path in pdf_paths:
                self.root.after(0, self.status_var.set,
                                f"Scanning {path.name} …")
                n = count_pages(str(path))
                results.append((path, n))
            # Hand off to the main-thread drain chain (single after call)
            self.root.after(0, _drain, iter(results))

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_list(self):
        self.pages.clear()
        self.page_splits.clear()
        self.page_statuses.clear()
        self.page_images.clear()
        for item in self.page_tree.get_children():
            self.page_tree.delete(item)
        self.canvas.delete("all")
        self.current_idx = -1
        self._log("List cleared.")

    # ── Split assignment ──────────────────────────────────────────────────────

    def _auto_split(self):
        """Assign first 80% to train, last 20% to eval."""
        n = len(self.pages)
        if n == 0:
            return
        n_train = max(1, round(n * 0.8))
        for i in range(n):
            split = "train" if i < n_train else "eval"
            self._tree_set_split(i, split)
        self._log(f"Auto split: {n_train} train, {n - n_train} eval.")

    def _set_all_splits(self, split: str):
        for i in range(len(self.pages)):
            self._tree_set_split(i, split)
        self._log(f"All pages set to {split}.")

    def _set_split_selected(self, split: str):
        sel = self.page_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._tree_set_split(idx, split)

    # ── Page navigation ───────────────────────────────────────────────────────

    def _on_page_select(self, _event=None):
        sel = self.page_tree.selection()
        if sel:
            self._select_page(int(sel[0]))

    def _select_page(self, idx: int):
        if idx < 0 or idx >= len(self.pages):
            return
        self.current_idx = idx
        self.page_tree.selection_set(str(idx))
        self.page_tree.see(str(idx))
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
        cw = self.canvas.winfo_width() or 550
        scale = min(1.0, cw / img.width)
        disp = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(disp)
        self._photo = photo
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.configure(scrollregion=(0, 0, disp.width, disp.height))
        pdf_path, page_num = self.pages[self.current_idx]
        split = self.page_splits[self.current_idx]
        self.status_var.set(
            f"{Path(pdf_path).name}  —  page {page_num}  [{split}]")

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
                vlm = VLM(model_name,
                          processor_name=self.processor_var.get().strip())
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
            self._save_page(self.current_idx, out,
                            meta=self._meta_from_ui(),
                            body=self.text_editor.get(1.0, END).strip())
            self._mark_done_ui(self.current_idx)
            self._log(f"Saved page {self.current_idx + 1} "
                      f"[{self.page_splits[self.current_idx]}].")
        except Exception as exc:
            self._log(f"Save error: {exc}")

    def _save_page(self, idx: int, out_root: str,
                   meta: dict | None = None, body: str = ""):
        """Save page to <out_root>/<split>/<stem>.pdf + .md"""
        pdf_path, page_num = self.pages[idx]
        split = self.page_splits[idx]
        stem = f"{Path(pdf_path).stem}_page{page_num}"
        split_dir = Path(out_root) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        if meta is None:
            meta = YAML_DEFAULTS.copy()

        # Write .md
        md_path = split_dir / f"{stem}.md"
        md_path.write_text(build_md(meta, body), encoding="utf-8")
        # Write single-page .pdf
        pdf_out = str(split_dir / f"{stem}.pdf")
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
                split = self.page_splits[idx]

                # Resume support: count extracted .md files for this PDF and
                # compare to total page count instead of checking individual files
                pdf_stem  = Path(pdf_path).stem
                split_dir = Path(out) / split
                total_pdf_pages = max(
                    pn for pp, pn in self.pages if pp == pdf_path
                )
                extracted_count = len(list(split_dir.glob(
                    f"{pdf_stem}_page*.md"
                )))
                if extracted_count >= total_pdf_pages:
                    self._log(
                        f"[{idx + 1}/{total}] skipped"
                        f" (complete: {extracted_count}/{total_pdf_pages})"
                    )
                    self.root.after(0, self._mark_done_ui, idx)
                    self.root.after(0, self.progress.configure, {"value": idx + 1})
                    continue
                elif page_num <= extracted_count:
                    self._log(f"[{idx + 1}/{total}] skipped (already extracted)")
                    self.root.after(0, self._mark_done_ui, idx)
                    self.root.after(0, self.progress.configure, {"value": idx + 1})
                    continue

                self._log(
                    f"[{idx + 1}/{total}] {Path(pdf_path).name} p{page_num} [{split}]")

                try:
                    if self.vlm and self.vlm.loaded:
                        meta, body = self._run_vlm_on(idx)
                        self.root.after(0, self._fill_editors, meta, body)
                    else:
                        meta, body = YAML_DEFAULTS.copy(), ""

                    self._save_page(idx, out, meta=meta, body=body)
                    self.root.after(0, self._mark_done_ui, idx)
                except Exception as exc:
                    self._log(f"  Error: {exc}")
                    self.root.after(0, self._tree_set_status, idx, "✗")

                self.root.after(0, self.progress.configure, {"value": idx + 1})

            self._log(f"Done. Output: {out}")

        threading.Thread(target=_run, daemon=True).start()

    def _mark_done_ui(self, idx: int):
        self._tree_set_status(idx, "✓")

    # ── Output dir ────────────────────────────────────────────────────────────

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self.out_var.set(folder)

    # ── Validate Dataset ──────────────────────────────────────────────────────

    def _validate_dataset(self):
        out = self._resolve_output()
        out_path = Path(out)
        self._log("=== Validating dataset …")
        total = valid = invalid = 0
        for split in ("train", "eval"):
            split_dir = out_path / split
            if not split_dir.exists():
                self._log(f"  [{split}] directory not found — skipped")
                continue
            md_files = list(split_dir.glob("*.md"))
            self._log(f"  [{split}] {len(md_files)} .md file(s)")
            for md_path in sorted(md_files):
                total += 1
                errors = _check_md_pair(md_path)
                if errors:
                    invalid += 1
                    for e in errors:
                        self._log(f"    ✗ {md_path.name}: {e}")
                else:
                    valid += 1

        if total == 0:
            self._log("  No files found. Run 'Extract & Save ALL' first.")
        else:
            self._log(f"=== Result: {valid}/{total} valid, {invalid} invalid.")
            if invalid == 0:
                self._log("  Dataset is ready for training!")

    # ── Generate Training Config ──────────────────────────────────────────────

    def _generate_config(self):
        out = self._resolve_output()
        try:
            config_path = _generate_training_config(out)
            self._log(f"Config written: {config_path}")
            self._log("Launch with:")
            self._log(f"  python -m olmocr.train.train --config {config_path}")
            # Show in a popup
            popup = tk.Toplevel(self.root)
            popup.title("Generated Training Config")
            popup.geometry("700x520")
            txt = scrolledtext.ScrolledText(
                popup, font=("Consolas", 9), wrap=tk.NONE)
            txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            txt.insert(1.0, Path(config_path).read_text(encoding="utf-8"))
            txt.config(state="disabled")
            ttk.Button(popup, text="Close",
                       command=popup.destroy).pack(pady=4)
        except Exception as exc:
            self._log(f"Config generation error: {exc}")

    # ── Launch / Stop Training ────────────────────────────────────────────────

    def _launch_training(self):
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning(
                "No output dir",
                "Set the output directory and generate a config first.")
            return
        config_path = Path(out) / "finetune_config.yaml"
        if not config_path.exists():
            messagebox.showwarning(
                "No config",
                f"Config not found:\n{config_path}\n\nClick 'Generate Config' first.")
            return

        if self._train_proc and self._train_proc.poll() is None:
            messagebox.showinfo("Already running", "Training is already running.")
            return

        self._log(f"=== Launching training …")
        self._log(f"Config: {config_path}")

        def _run():
            try:
                cmd = [sys.executable, "-m", "olmocr.train.train",
                       "--config", str(config_path)]
                self._train_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for line in self._train_proc.stdout:
                    self._log(line.rstrip())
                self._train_proc.wait()
                rc = self._train_proc.returncode
                self._log(f"=== Training finished (exit code {rc}).")
            except Exception as exc:
                self._log(f"Training error: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def _stop_training(self):
        if self._train_proc and self._train_proc.poll() is None:
            self._train_proc.terminate()
            self._log("Training process terminated.")
        else:
            self._log("No training process running.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = TrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
