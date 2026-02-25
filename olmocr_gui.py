"""
olmOCR Document Assistant - Enhanced Python GUI
Agentic document extraction powered by the olmOCR model family.

Improvements over the original version:
  - Native olmocr utilities (render_pdf_to_base64png, get_anchor_text)
  - Dual extraction modes: OCR (natural text) and Custom (structured JSON)
  - PageResponse structured output for OCR mode
  - Temperature-ramp retry logic matching the real olmOCR pipeline
  - Per-page status tracking with visual badges
  - Tabbed right panel: PDF Preview | Results Viewer
  - Results viewer with page-level text display and copy/export
  - GPU VRAM display in status bar
  - Persistent configuration (JSON)
  - Export to JSON / CSV / Excel / Markdown
  - Model selector (VLM and LLM)
  - Settings tab with all tunable parameters
  - Hallucination / repeat-pattern detection
  - Anchor text toggle for improved extraction quality
"""

import base64
import gc
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Optional olmocr utilities (graceful fallback when package not installed)
# ---------------------------------------------------------------------------
try:
    from olmocr.data.renderpdf import render_pdf_to_base64png
    _HAS_RENDERPDF = True
except ImportError:
    _HAS_RENDERPDF = False

try:
    from olmocr.prompts.anchor import get_anchor_text
    _HAS_ANCHOR = True
except ImportError:
    _HAS_ANCHOR = False

try:
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt, PageResponse
    _HAS_OLMOCR_PROMPT = True
except ImportError:
    _HAS_OLMOCR_PROMPT = False

try:
    from olmocr.repeatdetect import RepeatDetector
    _HAS_REPEATDETECT = True
except ImportError:
    _HAS_REPEATDETECT = False

# fallback PDF renderer using pdf2image
try:
    from pdf2image import convert_from_path as _pdf2image_convert
    _HAS_PDF2IMAGE = True
except ImportError:
    _HAS_PDF2IMAGE = False

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_VLM_MODEL = "allenai/olmOCR-7B-0924-preview"
ALT_VLM_MODELS = [
    "allenai/olmOCR-7B-0924-preview",
    "allenai/olmOCR-2-7B-1025",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ALT_LLM_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

# Temperature schedule mirroring the real olmOCR pipeline
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

CONFIG_FILE = Path.home() / ".olmocr_gui_config.json"

DEFAULT_CUSTOM_PROMPT = (
    "Extract data ONLY if a table is present. "
    "If no table: return {\"no_table\": true}\n"
    "If table present, extract ALL rows in exact order with exact column "
    "names as a JSON array of objects."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("olmocr_gui")

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}

def save_config(cfg: dict):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

# ============================================================================
# UTILITIES
# ============================================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    used_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
    return f"GPU: {props.name} | {used_gb:.1f}/{total_gb:.1f} GB free"

def resize_image_for_vlm(img: Image.Image, target: int = 1288) -> Image.Image:
    w, h = img.size
    if max(w, h) == target:
        return img
    if w >= h:
        return img.resize((target, int(h * target / w)), Image.LANCZOS)
    return img.resize((int(w * target / h), target), Image.LANCZOS)

def parse_json_from_text(text: str):
    """Try to extract a JSON list or object from raw model output."""
    text = text.replace("```json", "").replace("```", "").strip()
    for pattern in [r"\[\s*\{.*?\}\s*\]", r"\{[^{}]*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
                return obj if isinstance(obj, list) else [obj]
            except json.JSONDecodeError:
                pass
    return []

def parse_page_response_yaml(text: str) -> dict:
    """
    Parse the olmOCR YAML front matter response into a dict.
    Expected format:
        ---
        primary_language: en
        is_rotation_valid: true
        rotation_correction: 0
        is_table: false
        is_diagram: false
        ---
        Natural text content...
    """
    result = {
        "primary_language": None,
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": text,
    }

    fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL | re.MULTILINE)
    if fm_match:
        front_matter = fm_match.group(1)
        result["natural_text"] = fm_match.group(2).strip()
        for line in front_matter.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().lower()
                if key == "primary_language":
                    result[key] = val if val not in ("null", "none", "") else None
                elif key in ("is_rotation_valid", "is_table", "is_diagram"):
                    result[key] = val == "true"
                elif key == "rotation_correction":
                    try:
                        result[key] = int(val)
                    except ValueError:
                        pass
    return result

def detect_repeats(text: str, threshold: int = 5) -> bool:
    """Simple repeat pattern detection (fallback when RepeatDetector unavailable)."""
    if _HAS_REPEATDETECT:
        try:
            rd = RepeatDetector()
            counts = rd.ngram_repeats(text)
            return any(c >= threshold for c in counts)
        except Exception:
            pass
    # Fallback: check if any 20-char window repeats > 5 times
    words = text.split()
    if len(words) < 20:
        return False
    for size in [5, 10, 20]:
        chunks = [" ".join(words[i:i+size]) for i in range(0, len(words)-size, size)]
        for chunk in chunks:
            if chunk and chunks.count(chunk) > threshold:
                return True
    return False

# ============================================================================
# PDF RENDERING (with fallback)
# ============================================================================

def render_page_to_image(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    """
    Render a single PDF page to a PIL Image.
    Tries olmocr's native renderer first, then falls back to pdf2image.
    Page numbers are 1-indexed.
    """
    if _HAS_RENDERPDF:
        try:
            b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1288)
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            pass
    if _HAS_PDF2IMAGE:
        pages = _pdf2image_convert(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
        return pages[0].convert("RGB") if pages else None
    raise RuntimeError("No PDF renderer available. Install olmocr or pdf2image.")

def count_pdf_pages(pdf_path: str) -> int:
    """Count total pages in a PDF."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except Exception:
        if _HAS_PDF2IMAGE:
            return len(_pdf2image_convert(pdf_path, dpi=72))
        return 0

# ============================================================================
# VLM EXTRACTOR
# ============================================================================

class VLMExtractor:
    """Wraps a vision-language model for page-level extraction."""

    def __init__(self, model_name: str = DEFAULT_VLM_MODEL):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.loaded = False

    def load_model(self, progress_cb=None):
        if self.loaded:
            return "already loaded"
        if progress_cb:
            progress_cb(f"Loading processor: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        if progress_cb:
            progress_cb("Loading model weights (this may take a few minutes)...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.loaded = True
        return f"loaded {self.model_name}"

    def unload(self):
        del self.model, self.processor
        self.model = self.processor = None
        self.loaded = False
        clear_gpu()

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
    ) -> str:
        if not self.loaded:
            self.load_model()

        clear_gpu()
        img = resize_image_for_vlm(image.convert("RGB"))

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input], images=[img], padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
            )

        decoded = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        del inputs, output_ids
        clear_gpu()
        return decoded

    def extract_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> tuple[str, int]:
        """
        Retry extraction with increasing temperature (mirrors the real pipeline).
        Returns (decoded_text, attempt_index).
        """
        for attempt, temp in enumerate(TEMPERATURE_BY_ATTEMPT):
            result = self.extract(image, prompt, temperature=temp, max_new_tokens=max_new_tokens)
            if not detect_repeats(result):
                return result, attempt
            logger.warning("Repeat detected on attempt %d, retrying with higher temperature.", attempt)
        return result, len(TEMPERATURE_BY_ATTEMPT) - 1  # last attempt result

# ============================================================================
# LOCAL LLM ASSISTANT
# ============================================================================

class IntelligentAssistant:
    """Chat LLM that understands extraction context and helps the user."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.history: list = []

    def load_model(self, progress_cb=None):
        if self.loaded:
            return "already loaded"
        if progress_cb:
            progress_cb(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if progress_cb:
            progress_cb("Loading LLM weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.loaded = True
        return f"loaded {self.model_name}"

    def unload(self):
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        self.loaded = False
        clear_gpu()

    def chat(self, message: str, system_context: str = None) -> str:
        if not self.loaded:
            self.load_model()

        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        for msg in self.history[-8:]:
            messages.append(msg)
        messages.append({"role": "user", "content": message})

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=512,
                temperature=0.7, do_sample=True, top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        )
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.history = []

# ============================================================================
# PAGE RESULT
# ============================================================================

class PageResult:
    """Holds all information about one extracted page."""

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"
    STATUS_SKIPPED = "skipped"

    def __init__(self, pdf_path: str, page_num: int):
        self.pdf_path = pdf_path
        self.page_num = page_num         # 1-indexed
        self.status = self.STATUS_PENDING
        self.raw_output: str = ""
        self.parsed: dict = {}           # OCR mode: PageResponse fields
        self.structured_data: list = []  # Custom mode: list of row dicts
        self.attempt: int = 0
        self.duration_s: float = 0.0
        self.error_msg: str = ""
        self.repeated: bool = False

    @property
    def display_text(self) -> str:
        if self.parsed.get("natural_text"):
            return self.parsed["natural_text"]
        if self.structured_data:
            return json.dumps(self.structured_data, indent=2)
        return self.raw_output

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "page_num": self.page_num,
            "status": self.status,
            "attempt": self.attempt,
            "duration_s": round(self.duration_s, 3),
            "repeated": self.repeated,
            "error": self.error_msg,
            "parsed": self.parsed,
            "structured_data": self.structured_data,
            "raw_output": self.raw_output,
        }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class OlmOCRApp:
    """Main GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("olmOCR Document Assistant")
        self.root.geometry("1680x960")
        self.root.minsize(1200, 700)

        self._cfg = load_config()

        # ---- state ----
        self.mode = tk.StringVar(value=self._cfg.get("mode", "single"))
        self.extraction_mode = tk.StringVar(value=self._cfg.get("extraction_mode", "ocr"))
        self.use_anchor = tk.BooleanVar(value=self._cfg.get("use_anchor", False))
        self.anchor_engine = tk.StringVar(value=self._cfg.get("anchor_engine", "pdftotext"))
        self.max_new_tokens = tk.IntVar(value=self._cfg.get("max_new_tokens", 2048))
        self.vlm_model_var = tk.StringVar(value=self._cfg.get("vlm_model", DEFAULT_VLM_MODEL))
        self.llm_model_var = tk.StringVar(value=self._cfg.get("llm_model", DEFAULT_LLM_MODEL))

        self.selected_files: list[str] = []
        self.current_pdf: str = ""
        self.pdf_total_pages: int = 0
        self.page_images: dict[int, Image.Image] = {}   # 1-indexed cache
        self.page_vars: list[tuple[int, tk.BooleanVar]] = []
        self.selected_pages: list[int] = []             # 0-indexed
        self.page_results: list[PageResult] = []
        self.output_dir: str = ""

        self.vlm: VLMExtractor = None
        self.llm: IntelligentAssistant = None
        self.stop_flag = False
        self.extracting = False

        self._photo_refs: list = []   # keep tkinter PhotoImages alive

        self._setup_style()
        self._build_ui()
        self._update_gpu_label()

    # ------------------------------------------------------------------
    # STYLE
    # ------------------------------------------------------------------

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Arial", 10, "bold"))
        style.configure("Status.TLabel", font=("Consolas", 9))
        style.configure("Badge.done.TLabel", foreground="white", background="#28a745",
                        font=("Arial", 8, "bold"), padding=2)
        style.configure("Badge.error.TLabel", foreground="white", background="#dc3545",
                        font=("Arial", 8, "bold"), padding=2)
        style.configure("Badge.processing.TLabel", foreground="white", background="#fd7e14",
                        font=("Arial", 8, "bold"), padding=2)
        style.configure("Badge.pending.TLabel", foreground="#555", background="#e9ecef",
                        font=("Arial", 8), padding=2)

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- top status bar ----
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill=tk.X, padx=8, pady=(4, 0))

        ttk.Label(top_bar, text="olmOCR Document Assistant",
                  style="Title.TLabel").pack(side=tk.LEFT)

        self._model_status_var = tk.StringVar(value="No models loaded")
        ttk.Label(top_bar, textvariable=self._model_status_var,
                  foreground="gray", font=("Consolas", 9)).pack(side=tk.RIGHT, padx=10)

        self._gpu_label_var = tk.StringVar(value="")
        ttk.Label(top_bar, textvariable=self._gpu_label_var,
                  foreground="#555", font=("Consolas", 9)).pack(side=tk.RIGHT, padx=15)

        # ---- main paned window ----
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(main_pw)
        right = ttk.Frame(main_pw)
        main_pw.add(left, weight=2)
        main_pw.add(right, weight=3)

        self._build_left(left)
        self._build_right(right)

    # ---------- LEFT PANEL ----------

    def _build_left(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Chat
        chat_tab = ttk.Frame(nb)
        nb.add(chat_tab, text="  Chat  ")
        self._build_chat_tab(chat_tab)

        # Tab 2: Extraction controls
        extract_tab = ttk.Frame(nb)
        nb.add(extract_tab, text="  Extraction  ")
        self._build_extraction_tab(extract_tab)

        # Tab 3: Settings
        settings_tab = ttk.Frame(nb)
        nb.add(settings_tab, text="  Settings  ")
        self._build_settings_tab(settings_tab)

    def _build_chat_tab(self, parent):
        # model buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(btn_frame, text="Load VLM", command=self._cmd_load_vlm,
                   width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load LLM", command=self._cmd_load_llm,
                   width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear Chat", command=self._clear_chat,
                   width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Unload All", command=self._cmd_unload_all,
                   width=12).pack(side=tk.RIGHT, padx=2)

        # chat display
        self.chat_display = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 9),
            bg="#f7f7f7", state=tk.NORMAL, height=22,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)
        self.chat_display.tag_config("user",      foreground="#0055cc", font=("Consolas", 9, "bold"))
        self.chat_display.tag_config("assistant", foreground="#008800")
        self.chat_display.tag_config("system",    foreground="#666666", font=("Consolas", 8, "italic"))
        self.chat_display.tag_config("error",     foreground="#cc0000")
        self.chat_display.tag_config("warn",      foreground="#cc7700")

        # chat input
        inp_frame = ttk.Frame(parent)
        inp_frame.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(inp_frame, text="You:").pack(side=tk.LEFT, padx=2)
        self.chat_entry = ttk.Entry(inp_frame, font=("Arial", 10))
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.chat_entry.bind("<Return>", lambda _: self._send_message())
        ttk.Button(inp_frame, text="Send", command=self._send_message,
                   width=8).pack(side=tk.LEFT)

        self._sys_msg("olmOCR Document Assistant ready.")
        self._sys_msg("Type 'help' for a list of commands, or use the Extraction tab.")

    def _build_extraction_tab(self, parent):
        # ---- file selection ----
        f_files = ttk.LabelFrame(parent, text="Files", style="Section.TLabelframe")
        f_files.pack(fill=tk.X, padx=6, pady=4)

        mode_row = ttk.Frame(f_files)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_row, text="Single PDF", variable=self.mode,
                        value="single").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_row, text="Batch Folder", variable=self.mode,
                        value="batch").pack(side=tk.LEFT, padx=4)

        file_row = ttk.Frame(f_files)
        file_row.pack(fill=tk.X, pady=2)
        ttk.Button(file_row, text="Select Files/Folder",
                   command=self._select_files, width=20).pack(side=tk.LEFT, padx=4)
        self._file_label = ttk.Label(file_row, text="None selected", foreground="gray")
        self._file_label.pack(side=tk.LEFT, padx=6)

        # ---- page selection ----
        f_pages = ttk.LabelFrame(parent, text="Page Selection", style="Section.TLabelframe")
        f_pages.pack(fill=tk.X, padx=6, pady=4)

        pg_btn_row = ttk.Frame(f_pages)
        pg_btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(pg_btn_row, text="All", command=self._select_all_pages,
                   width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(pg_btn_row, text="None", command=self._deselect_all_pages,
                   width=8).pack(side=tk.LEFT, padx=2)
        self._page_count_label = ttk.Label(pg_btn_row, text="0 pages selected",
                                           foreground="gray")
        self._page_count_label.pack(side=tk.LEFT, padx=8)

        range_row = ttk.Frame(f_pages)
        range_row.pack(fill=tk.X, pady=2)
        ttk.Label(range_row, text="Range:").pack(side=tk.LEFT, padx=4)
        self._range_entry = ttk.Entry(range_row, width=22)
        self._range_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(range_row, text="Apply", command=self._apply_range,
                   width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_row, text="e.g. 1-5, 8, 10-12",
                  foreground="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=4)

        # ---- output ----
        f_output = ttk.LabelFrame(parent, text="Output", style="Section.TLabelframe")
        f_output.pack(fill=tk.X, padx=6, pady=4)

        out_row = ttk.Frame(f_output)
        out_row.pack(fill=tk.X, pady=2)
        ttk.Button(out_row, text="Output Directory",
                   command=self._select_output, width=18).pack(side=tk.LEFT, padx=4)
        self._output_label = ttk.Label(out_row, text="Not set", foreground="gray")
        self._output_label.pack(side=tk.LEFT, padx=6)

        # ---- prompt ----
        f_prompt = ttk.LabelFrame(parent, text="Prompt (Custom mode)",
                                  style="Section.TLabelframe")
        f_prompt.pack(fill=tk.X, padx=6, pady=4)

        prompt_btns = ttk.Frame(f_prompt)
        prompt_btns.pack(fill=tk.X)
        ttk.Button(prompt_btns, text="Load", command=self._load_prompt,
                   width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(prompt_btns, text="Save", command=self._save_prompt,
                   width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(prompt_btns, text="Reset", command=self._reset_prompt,
                   width=8).pack(side=tk.LEFT, padx=2)

        self.prompt_text = scrolledtext.ScrolledText(
            f_prompt, height=4, wrap=tk.WORD, font=("Consolas", 8),
        )
        self.prompt_text.pack(fill=tk.X, pady=4)
        self._reset_prompt()

        # ---- extraction controls ----
        f_run = ttk.LabelFrame(parent, text="Run", style="Section.TLabelframe")
        f_run.pack(fill=tk.X, padx=6, pady=4)

        mode_r = ttk.Frame(f_run)
        mode_r.pack(fill=tk.X, pady=2)
        ttk.Label(mode_r, text="Extraction:").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_r, text="OCR (natural text)",
                        variable=self.extraction_mode, value="ocr").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_r, text="Custom (JSON tables)",
                        variable=self.extraction_mode, value="custom").pack(side=tk.LEFT, padx=4)

        btn_r = ttk.Frame(f_run)
        btn_r.pack(fill=tk.X, pady=4)
        self._start_btn = ttk.Button(btn_r, text="▶  START EXTRACTION",
                                     command=self._start_extraction)
        self._start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._stop_btn = ttk.Button(btn_r, text="■  STOP",
                                    command=self._stop_extraction, state=tk.DISABLED, width=10)
        self._stop_btn.pack(side=tk.LEFT, padx=4)

        self._progress_bar = ttk.Progressbar(f_run, mode="indeterminate")
        self._progress_bar.pack(fill=tk.X, padx=4, pady=2)
        self._progress_label = ttk.Label(f_run, text="Ready", foreground="gray",
                                         style="Status.TLabel")
        self._progress_label.pack(pady=2)

        # ---- export ----
        f_export = ttk.LabelFrame(parent, text="Export Results", style="Section.TLabelframe")
        f_export.pack(fill=tk.X, padx=6, pady=4)

        exp_r = ttk.Frame(f_export)
        exp_r.pack(fill=tk.X, pady=2)
        ttk.Button(exp_r, text="JSON", command=lambda: self._export("json"),
                   width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_r, text="CSV", command=lambda: self._export("csv"),
                   width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_r, text="Excel", command=lambda: self._export("xlsx"),
                   width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_r, text="Markdown", command=lambda: self._export("md"),
                   width=10).pack(side=tk.LEFT, padx=2)

    def _build_settings_tab(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # VLM model
        ttk.Label(f, text="VLM Model:", font=("Arial", 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=4)
        vlm_combo = ttk.Combobox(f, textvariable=self.vlm_model_var,
                                  values=ALT_VLM_MODELS, width=40)
        vlm_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        # LLM model
        ttk.Label(f, text="LLM Model:", font=("Arial", 9, "bold")).grid(
            row=1, column=0, sticky="w", pady=4)
        llm_combo = ttk.Combobox(f, textvariable=self.llm_model_var,
                                  values=ALT_LLM_MODELS, width=40)
        llm_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        # Max new tokens
        ttk.Label(f, text="Max New Tokens:", font=("Arial", 9, "bold")).grid(
            row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(f, from_=256, to=8192, increment=256,
                    textvariable=self.max_new_tokens, width=10).grid(
            row=2, column=1, sticky="w", padx=8, pady=4)

        # Anchor text
        ttk.Label(f, text="Anchor Text (OCR mode):", font=("Arial", 9, "bold")).grid(
            row=3, column=0, sticky="w", pady=4)
        anchor_frame = ttk.Frame(f)
        anchor_frame.grid(row=3, column=1, sticky="w", padx=8)
        ttk.Checkbutton(anchor_frame, text="Enable", variable=self.use_anchor).pack(side=tk.LEFT)
        anchor_engines = ["pdftotext", "pdfium", "pypdf", "topcoherency"]
        ttk.Combobox(anchor_frame, textvariable=self.anchor_engine,
                     values=anchor_engines, width=14).pack(side=tk.LEFT, padx=6)

        # Capability info
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=10)
        caps = []
        caps.append(f"olmocr renderpdf:  {'yes' if _HAS_RENDERPDF else 'no (using pdf2image)'}")
        caps.append(f"olmocr anchor:     {'yes' if _HAS_ANCHOR else 'no'}")
        caps.append(f"olmocr prompts:    {'yes' if _HAS_OLMOCR_PROMPT else 'no'}")
        caps.append(f"olmocr repeatdetect: {'yes' if _HAS_REPEATDETECT else 'no'}")
        caps.append(f"pdf2image fallback: {'yes' if _HAS_PDF2IMAGE else 'no'}")
        caps.append(f"PyTorch CUDA:      {'yes — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
        cap_text = "\n".join(caps)
        ttk.Label(f, text=cap_text, font=("Consolas", 9),
                  foreground="#444", justify=tk.LEFT).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=4)

        # Save button
        ttk.Button(f, text="Save Settings", command=self._save_settings).grid(
            row=6, column=1, sticky="e", pady=10)

        f.columnconfigure(1, weight=1)

    # ---------- RIGHT PANEL ----------

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: PDF Preview
        preview_tab = ttk.Frame(nb)
        nb.add(preview_tab, text="  PDF Preview  ")
        self._build_preview_tab(preview_tab)

        # Tab 2: Results viewer
        results_tab = ttk.Frame(nb)
        nb.add(results_tab, text="  Results  ")
        self._build_results_tab(results_tab)

        self._right_nb = nb

    def _build_preview_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=6, pady=4)
        self._preview_info = ttk.Label(top, text="Load a PDF to see preview.",
                                       font=("Arial", 10))
        self._preview_info.pack(side=tk.LEFT, padx=6)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(canvas_frame, bg="#d0d0d0")
        vscroll = ttk.Scrollbar(canvas_frame, orient="vertical",
                                command=self.preview_canvas.yview)
        hscroll = ttk.Scrollbar(canvas_frame, orient="horizontal",
                                 command=self.preview_canvas.xview)
        self.preview_canvas.configure(
            yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Mouse-wheel scroll
        self.preview_canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.preview_canvas.bind("<Button-4>", self._on_canvas_scroll)
        self.preview_canvas.bind("<Button-5>", self._on_canvas_scroll)

    def _build_results_tab(self, parent):
        # Page list on left, text on right
        pw = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(pw)
        pw.add(list_frame, weight=1)
        text_frame = ttk.Frame(pw)
        pw.add(text_frame, weight=3)

        # Page list with status
        ttk.Label(list_frame, text="Pages", font=("Arial", 10, "bold")).pack(pady=4)
        self._results_listbox = tk.Listbox(list_frame, font=("Consolas", 9),
                                           activestyle="dotbox")
        self._results_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._results_listbox.bind("<<ListboxSelect>>", self._on_result_select)

        # Text viewer
        top_text = ttk.Frame(text_frame)
        top_text.pack(fill=tk.X)
        ttk.Label(top_text, text="Extracted Content",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Button(top_text, text="Copy", command=self._copy_result,
                   width=8).pack(side=tk.RIGHT, padx=4)

        self._result_text = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=("Consolas", 9),
            bg="#fafafa",
        )
        self._result_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # GPU LABEL UPDATER
    # ------------------------------------------------------------------

    def _update_gpu_label(self):
        self._gpu_label_var.set(gpu_info())
        self.root.after(5000, self._update_gpu_label)

    # ------------------------------------------------------------------
    # CHAT HELPERS
    # ------------------------------------------------------------------

    def _sys_msg(self, msg: str):
        self.chat_display.insert(tk.END, f"{msg}\n", "system")
        self.chat_display.see(tk.END)

    def _user_msg(self, msg: str):
        self.chat_display.insert(tk.END, f"\nYou: {msg}\n", "user")
        self.chat_display.see(tk.END)

    def _asst_msg(self, msg: str):
        self.chat_display.insert(tk.END, f"Assistant: {msg}\n\n", "assistant")
        self.chat_display.see(tk.END)

    def _err_msg(self, msg: str):
        self.chat_display.insert(tk.END, f"[ERROR] {msg}\n", "error")
        self.chat_display.see(tk.END)

    def _warn_msg(self, msg: str):
        self.chat_display.insert(tk.END, f"[WARN] {msg}\n", "warn")
        self.chat_display.see(tk.END)

    def _clear_chat(self):
        self.chat_display.delete(1.0, tk.END)
        if self.llm:
            self.llm.clear_history()
        self._sys_msg("Chat cleared.")

    # ------------------------------------------------------------------
    # MESSAGE ROUTING
    # ------------------------------------------------------------------

    def _send_message(self):
        msg = self.chat_entry.get().strip()
        if not msg:
            return
        self.chat_entry.delete(0, tk.END)
        self._user_msg(msg)
        t = threading.Thread(target=self._route_message, args=(msg,), daemon=True)
        t.start()

    def _route_message(self, msg: str):
        ml = msg.lower().strip()

        if ml in ("load vlm", "load vision model", "load extraction model"):
            self._cmd_load_vlm(); return
        if ml in ("load llm", "load chat", "load assistant"):
            self._cmd_load_llm(); return
        if ml in ("extract", "start extraction", "run extraction"):
            self.root.after(0, self._start_extraction); return
        if ml in ("stop", "stop extraction", "cancel"):
            self._stop_extraction(); return
        if ml in ("status", "progress"):
            self._chat_status(); return
        if ml in ("show data", "view data"):
            self._chat_show_data(); return
        if ml in ("stats", "statistics", "summary"):
            self._chat_stats(); return
        if ml in ("excel", "export excel"):
            self.root.after(0, lambda: self._export("xlsx")); return
        if ml in ("clear", "clear chat"):
            self.root.after(0, self._clear_chat); return
        if "help" in ml:
            self._chat_help(); return
        if ml in ("unload", "unload all", "free memory"):
            self._cmd_unload_all(); return
        if ml.startswith("gpu"):
            self.root.after(0, lambda: self._asst_msg(gpu_info())); return

        # fallback: LLM conversation
        if not self.llm or not self.llm.loaded:
            self.root.after(0, lambda: self._asst_msg(
                "LLM not loaded. Type 'load llm' to enable AI chat."))
            return

        ctx = self._build_llm_context()
        try:
            response = self.llm.chat(msg, system_context=ctx)
            self.root.after(0, lambda r=response: self._asst_msg(r))
        except Exception as e:
            self.root.after(0, lambda: self._err_msg(f"LLM error: {e}"))

    def _build_llm_context(self) -> str:
        ctx = "You are an intelligent document extraction assistant powered by the olmOCR model. "
        ctx += f"VLM loaded: {'yes (' + self.vlm.model_name + ')' if self.vlm and self.vlm.loaded else 'no'}. "
        ctx += f"LLM loaded: {'yes' if self.llm and self.llm.loaded else 'no'}. "
        ctx += f"Files selected: {len(self.selected_files)}. "
        ctx += f"Pages selected: {len(self.selected_pages)}. "
        ctx += f"Results available: {len(self.page_results)}. "
        if self.page_results:
            done = sum(1 for r in self.page_results if r.status == PageResult.STATUS_DONE)
            ctx += f"Done pages: {done}/{len(self.page_results)}. "
        return ctx

    def _chat_status(self):
        if self.extracting:
            done = sum(1 for r in self.page_results if r.status == PageResult.STATUS_DONE)
            total = len(self.page_results)
            self._asst_msg(f"Extraction in progress: {done}/{total} pages done.")
        elif self.page_results:
            done = sum(1 for r in self.page_results if r.status == PageResult.STATUS_DONE)
            errors = sum(1 for r in self.page_results if r.status == PageResult.STATUS_ERROR)
            self._asst_msg(f"Last extraction: {done} done, {errors} errors out of {len(self.page_results)} pages.")
        else:
            self._asst_msg("No extraction has been run yet.")

    def _chat_show_data(self):
        if not self.page_results:
            self._asst_msg("No data extracted yet.")
            return
        done = [r for r in self.page_results if r.status == PageResult.STATUS_DONE]
        preview = json.dumps([r.to_dict() for r in done[:2]], indent=2)
        self._asst_msg(f"{len(done)} completed pages.\n\nPreview (first 2):\n{preview}")

    def _chat_stats(self):
        all_rows = []
        for r in self.page_results:
            all_rows.extend(r.structured_data)
        if not all_rows:
            # OCR mode: count characters
            all_text = " ".join(
                r.parsed.get("natural_text", "") or "" for r in self.page_results
            )
            self._asst_msg(
                f"OCR results: {len(self.page_results)} pages, "
                f"~{len(all_text.split())} words extracted."
            )
            return
        df = pd.DataFrame(all_rows)
        self._asst_msg(
            f"Structured rows: {len(df)}\n"
            f"Columns: {list(df.columns)}\n"
            f"Numeric cols: {list(df.select_dtypes('number').columns)}\n\n"
            f"{df.describe().to_string()}"
        )

    def _chat_help(self):
        self._asst_msg(
            "Commands:\n"
            "  load vlm / load llm — load models\n"
            "  unload all          — free GPU memory\n"
            "  extract             — start extraction\n"
            "  stop                — stop extraction\n"
            "  status              — show progress\n"
            "  show data           — preview results\n"
            "  stats               — data statistics\n"
            "  excel               — export to Excel\n"
            "  gpu                 — show GPU info\n"
            "  clear               — clear chat history\n"
            "  help                — this message\n"
            "\nOr just ask me anything about your documents!"
        )

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------

    def _cmd_load_vlm(self):
        if self.vlm and self.vlm.loaded:
            self.root.after(0, lambda: self._sys_msg(f"VLM already loaded: {self.vlm.model_name}"))
            return
        model_name = self.vlm_model_var.get().strip()
        self.root.after(0, lambda: self._sys_msg(f"Loading VLM: {model_name} …"))

        def _load():
            try:
                self.vlm = VLMExtractor(model_name)
                result = self.vlm.load_model(
                    progress_cb=lambda m: self.root.after(0, lambda msg=m: self._sys_msg(msg))
                )
                self.root.after(0, lambda: [
                    self._sys_msg(f"VLM loaded: {result}"),
                    self._update_model_status(),
                ])
            except Exception as e:
                self.vlm = None
                self.root.after(0, lambda: self._err_msg(f"VLM load failed: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def _cmd_load_llm(self):
        if self.llm and self.llm.loaded:
            self.root.after(0, lambda: self._sys_msg(f"LLM already loaded: {self.llm.model_name}"))
            return
        model_name = self.llm_model_var.get().strip()
        self.root.after(0, lambda: self._sys_msg(f"Loading LLM: {model_name} …"))

        def _load():
            try:
                self.llm = IntelligentAssistant(model_name)
                result = self.llm.load_model(
                    progress_cb=lambda m: self.root.after(0, lambda msg=m: self._sys_msg(msg))
                )
                self.root.after(0, lambda: [
                    self._sys_msg(f"LLM loaded: {result}"),
                    self._asst_msg("Hi! I'm your AI assistant. How can I help with document extraction?"),
                    self._update_model_status(),
                ])
            except Exception as e:
                self.llm = None
                self.root.after(0, lambda: self._err_msg(f"LLM load failed: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def _cmd_unload_all(self):
        def _unload():
            if self.vlm and self.vlm.loaded:
                self.vlm.unload()
                self.root.after(0, lambda: self._sys_msg("VLM unloaded."))
            if self.llm and self.llm.loaded:
                self.llm.unload()
                self.root.after(0, lambda: self._sys_msg("LLM unloaded."))
            self.root.after(0, self._update_model_status)
        threading.Thread(target=_unload, daemon=True).start()

    def _update_model_status(self):
        parts = []
        if self.vlm and self.vlm.loaded:
            parts.append(f"VLM: {Path(self.vlm.model_name).name}")
        if self.llm and self.llm.loaded:
            parts.append(f"LLM: {Path(self.llm.model_name).name}")
        if parts:
            self._model_status_var.set(" | ".join(parts))
        else:
            self._model_status_var.set("No models loaded")

    # ------------------------------------------------------------------
    # FILE SELECTION
    # ------------------------------------------------------------------

    def _select_files(self):
        if self.mode.get() == "single":
            path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
            if not path:
                return
            self.selected_files = [path]
            self._file_label.config(text=Path(path).name, foreground="black")
            self._load_pdf_preview(path)
        else:
            folder = filedialog.askdirectory(title="Select Folder with PDFs")
            if not folder:
                return
            pdfs = []
            for dirpath, _, filenames in os.walk(folder):
                for fn in filenames:
                    if fn.lower().endswith(".pdf"):
                        pdfs.append(os.path.join(dirpath, fn))
            if not pdfs:
                messagebox.showwarning("No PDFs", f"No PDF files found in {folder}")
                return
            self.selected_files = sorted(pdfs)
            self._file_label.config(
                text=f"{len(pdfs)} PDFs in {Path(folder).name}", foreground="black")
            self._sys_msg(f"Found {len(pdfs)} PDFs in {folder}")
            for p in pdfs[:10]:
                self._sys_msg(f"  • {Path(p).name}")
            if len(pdfs) > 10:
                self._sys_msg(f"  … and {len(pdfs)-10} more")
            self._clear_preview()

    def _load_pdf_preview(self, path: str):
        self._sys_msg(f"Loading preview: {Path(path).name} …")

        def _render():
            try:
                n = count_pdf_pages(path)
                if n == 0:
                    self.root.after(0, lambda: self._err_msg("Could not read PDF pages."))
                    return
                self.current_pdf = path
                self.pdf_total_pages = n
                self.page_images = {}
                # Render thumbnails for preview (lazy, render first 50 pages max initially)
                limit = min(n, 50)
                images = []
                for pg in range(1, limit + 1):
                    img = render_page_to_image(path, pg, dpi=100)
                    self.page_images[pg] = img
                    images.append((pg, img))
                self.root.after(0, lambda: self._display_previews(images, n))
            except Exception as e:
                self.root.after(0, lambda: self._err_msg(f"Preview error: {e}"))

        threading.Thread(target=_render, daemon=True).start()

    def _display_previews(self, images: list, total_pages: int):
        self.preview_canvas.delete("all")
        self._photo_refs.clear()
        self.page_vars.clear()

        y = 10
        thumb_w = 300

        for pg, img in images:
            thumb = img.copy()
            ratio = thumb_w / img.width
            new_h = int(img.height * ratio)
            thumb = thumb.resize((thumb_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self._photo_refs.append(photo)

            var = tk.BooleanVar(value=False)

            cb = tk.Checkbutton(
                self.preview_canvas, text=f"  Page {pg}", variable=var,
                command=self._update_page_count,
                font=("Arial", 9), bg="#d0d0d0", activebackground="#c0c0c0",
                anchor="w",
            )
            self.preview_canvas.create_window(8, y, anchor="nw", window=cb, width=thumb_w)

            img_id = self.preview_canvas.create_image(8, y + 22, image=photo, anchor="nw")

            def _make_toggle(v):
                def _handler(e):
                    v.set(not v.get())
                    self._update_page_count()
                return _handler

            self.preview_canvas.tag_bind(img_id, "<Button-1>", _make_toggle(var))

            self.page_vars.append((pg - 1, var))  # 0-indexed in page_vars
            y += new_h + 36

        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
        self._preview_info.config(
            text=f"{total_pages} pages total — click thumbnails or checkboxes to select"
        )
        self._sys_msg(f"Preview loaded: {total_pages} pages")

    def _clear_preview(self):
        self.preview_canvas.delete("all")
        self.page_images = {}
        self.page_vars = []
        self.selected_pages = []
        self._photo_refs.clear()
        self._preview_info.config(text="No PDF loaded.")

    def _on_canvas_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self.preview_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.preview_canvas.yview_scroll(1, "units")

    # ------------------------------------------------------------------
    # PAGE SELECTION
    # ------------------------------------------------------------------

    def _update_page_count(self):
        if self.page_vars:
            self.selected_pages = [i for i, var in self.page_vars if var.get()]
        self._page_count_label.config(
            text=f"{len(self.selected_pages)} pages selected"
        )

    def _select_all_pages(self):
        for _, var in self.page_vars:
            var.set(True)
        self._update_page_count()

    def _deselect_all_pages(self):
        for _, var in self.page_vars:
            var.set(False)
        self._update_page_count()

    def _apply_range(self):
        if not self.page_vars:
            self._err_msg("Load a PDF first.")
            return
        raw = self._range_entry.get().strip()
        if not raw:
            self._err_msg("Enter a page range (e.g. 1-5, 8, 10-12).")
            return
        try:
            target: set[int] = set()
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    for pg in range(int(a.strip()), int(b.strip()) + 1):
                        target.add(pg - 1)   # convert to 0-indexed
                else:
                    target.add(int(part.strip()) - 1)

            for idx, var in self.page_vars:
                var.set(idx in target)
            self._update_page_count()
            self._sys_msg(f"Range applied: {raw} → {len(target)} pages selected")
        except Exception as e:
            self._err_msg(f"Invalid range: {e}")

    # ------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------------

    def _select_output(self):
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir = folder
            self._output_label.config(text=str(Path(folder).name), foreground="black")

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------

    def _reset_prompt(self):
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, DEFAULT_CUSTOM_PROMPT)

    def _load_prompt(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*")])
        if path:
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(1.0, Path(path).read_text())
            self._sys_msg(f"Prompt loaded from {Path(path).name}")

    def _save_prompt(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            Path(path).write_text(self.prompt_text.get(1.0, tk.END))
            self._sys_msg(f"Prompt saved to {Path(path).name}")

    # ------------------------------------------------------------------
    # SETTINGS
    # ------------------------------------------------------------------

    def _save_settings(self):
        cfg = {
            "mode": self.mode.get(),
            "extraction_mode": self.extraction_mode.get(),
            "use_anchor": self.use_anchor.get(),
            "anchor_engine": self.anchor_engine.get(),
            "max_new_tokens": self.max_new_tokens.get(),
            "vlm_model": self.vlm_model_var.get(),
            "llm_model": self.llm_model_var.get(),
        }
        save_config(cfg)
        self._sys_msg("Settings saved.")

    # ------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------

    def _start_extraction(self):
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select PDF file(s) first.")
            return
        if self.mode.get() == "single" and not self.selected_pages:
            messagebox.showwarning("No Pages", "Please select at least one page.")
            return
        if not self.vlm:
            self._sys_msg("VLM not loaded — loading now …")
            self._cmd_load_vlm()
            # Give the model a moment to start loading, then re-check
            messagebox.showinfo("Loading", "VLM is loading. Please wait for it to finish, then click Start again.")
            return

        if not self.output_dir:
            # Auto-create next to first PDF
            auto = Path(self.selected_files[0]).parent / "olmocr_output"
            auto.mkdir(exist_ok=True)
            self.output_dir = str(auto)
            self._output_label.config(text=auto.name, foreground="black")
            self._sys_msg(f"Auto-created output dir: {auto}")

        self.stop_flag = False
        self.extracting = True
        self.page_results = []
        self._results_listbox.delete(0, tk.END)
        self._result_text.delete(1.0, tk.END)

        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._progress_bar.start()
        self._progress_label.config(text="Extracting …", foreground="blue")

        threading.Thread(target=self._run_extraction, daemon=True).start()

    def _stop_extraction(self):
        self.stop_flag = True
        self._progress_label.config(text="Stopping after current page …", foreground="orange")

    def _run_extraction(self):
        try:
            start_ts = time.time()
            ext_mode = self.extraction_mode.get()

            if ext_mode == "ocr":
                if _HAS_OLMOCR_PROMPT:
                    base_prompt = build_no_anchoring_v4_yaml_prompt()
                else:
                    base_prompt = (
                        "Attached is one page of a document. "
                        "Return the plain text as markdown, with a YAML front matter block "
                        "containing: primary_language, is_rotation_valid, rotation_correction, "
                        "is_table, is_diagram."
                    )
            else:
                base_prompt = self.prompt_text.get(1.0, tk.END).strip()

            if self.mode.get() == "single":
                self._extract_single_pdf(
                    self.selected_files[0], self.selected_pages, base_prompt, ext_mode)
            else:
                self._extract_batch(base_prompt, ext_mode)

            total_time = time.time() - start_ts
            done = sum(1 for r in self.page_results if r.status == PageResult.STATUS_DONE)
            errors = sum(1 for r in self.page_results if r.status == PageResult.STATUS_ERROR)

            self._save_results_json(total_time)
            self.root.after(0, lambda: self._asst_msg(
                f"Extraction complete in {total_time:.1f}s — "
                f"{done} done, {errors} errors."
            ))

        except Exception as e:
            logger.exception("Extraction error")
            self.root.after(0, lambda: self._err_msg(f"Extraction error: {e}"))
        finally:
            self.root.after(0, self._extraction_done)

    def _extract_single_pdf(
        self,
        pdf_path: str,
        page_indices: list[int],   # 0-indexed
        base_prompt: str,
        ext_mode: str,
    ):
        for idx in page_indices:
            if self.stop_flag:
                break
            pg = idx + 1  # 1-indexed
            result = PageResult(pdf_path, pg)
            self.page_results.append(result)
            self.root.after(0, lambda r=result: self._add_result_to_list(r))

            self.root.after(0, lambda p=pg: self._progress_label.config(
                text=f"Processing page {p} …"))

            result.status = PageResult.STATUS_PROCESSING
            self.root.after(0, lambda r=result: self._refresh_result_in_list(r))

            t0 = time.time()
            try:
                # Get image
                img = self.page_images.get(pg)
                if img is None:
                    img = render_page_to_image(pdf_path, pg)
                    self.page_images[pg] = img

                # Build prompt with optional anchor
                prompt = self._build_prompt(base_prompt, pdf_path, pg, ext_mode)

                # Extract
                raw, attempt = self.vlm.extract_with_retry(
                    img, prompt, max_new_tokens=self.max_new_tokens.get())

                result.raw_output = raw
                result.attempt = attempt
                result.duration_s = time.time() - t0
                result.repeated = detect_repeats(raw)

                if ext_mode == "ocr":
                    result.parsed = parse_page_response_yaml(raw)
                else:
                    result.structured_data = parse_json_from_text(raw)

                result.status = PageResult.STATUS_DONE

                if attempt > 0:
                    self.root.after(0, lambda p=pg, a=attempt: self._warn_msg(
                        f"Page {p}: needed {a+1} attempts (repeat detected)"))

            except Exception as e:
                result.status = PageResult.STATUS_ERROR
                result.error_msg = str(e)
                result.duration_s = time.time() - t0
                self.root.after(0, lambda p=pg, err=str(e): self._err_msg(
                    f"Page {p} error: {err}"))

            self.root.after(0, lambda r=result: self._refresh_result_in_list(r))

    def _extract_batch(self, base_prompt: str, ext_mode: str):
        for fi, pdf_path in enumerate(self.selected_files, 1):
            if self.stop_flag:
                break
            pdf_name = Path(pdf_path).name
            self.root.after(0, lambda n=pdf_name, i=fi, total=len(self.selected_files):
                            self._sys_msg(f"[{i}/{total}] {n}"))

            n_pages = count_pdf_pages(pdf_path)
            all_indices = list(range(n_pages))
            self._extract_single_pdf(pdf_path, all_indices, base_prompt, ext_mode)

    def _build_prompt(
        self, base_prompt: str, pdf_path: str, page_num: int, ext_mode: str
    ) -> str:
        if ext_mode != "ocr" or not self.use_anchor.get() or not _HAS_ANCHOR:
            return base_prompt
        try:
            anchor = get_anchor_text(
                pdf_path, page_num,
                pdf_engine=self.anchor_engine.get(),
                target_length=4000,
            )
            return (
                f"Below is the image of one page of a document, as well as some raw textual "
                f"content previously extracted from it.\n{base_prompt}\n"
                f"RAW_TEXT_START\n{anchor}\nRAW_TEXT_END"
            )
        except Exception:
            return base_prompt

    def _extraction_done(self):
        self.extracting = False
        self._progress_bar.stop()
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._progress_label.config(text="Ready", foreground="gray")
        # Switch to results tab
        self._right_nb.select(1)

    def _save_results_json(self, duration: float):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(self.output_dir) / f"olmocr_results_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "total_duration_s": round(duration, 3),
            "extraction_mode": self.extraction_mode.get(),
            "vlm_model": self.vlm.model_name if self.vlm else None,
            "pages": [r.to_dict() for r in self.page_results],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        self.root.after(0, lambda p=out_path: self._sys_msg(f"Results saved: {p.name}"))

    # ------------------------------------------------------------------
    # RESULTS UI
    # ------------------------------------------------------------------

    def _add_result_to_list(self, result: PageResult):
        label = f"P{result.page_num}  [{result.status}]"
        self._results_listbox.insert(tk.END, label)
        idx = self._results_listbox.size() - 1
        self._set_listbox_color(idx, result.status)

    def _refresh_result_in_list(self, result: PageResult):
        # Find result index in list
        for i, r in enumerate(self.page_results):
            if r is result:
                label = f"P{result.page_num:>3}  [{result.status}]"
                if result.status == PageResult.STATUS_DONE and result.attempt > 0:
                    label += f" (r{result.attempt+1})"
                if result.repeated:
                    label += " !"
                self._results_listbox.delete(i)
                self._results_listbox.insert(i, label)
                self._set_listbox_color(i, result.status)
                return

    def _set_listbox_color(self, idx: int, status: str):
        color_map = {
            PageResult.STATUS_DONE:       ("#d4edda", "#155724"),
            PageResult.STATUS_ERROR:      ("#f8d7da", "#721c24"),
            PageResult.STATUS_PROCESSING: ("#fff3cd", "#856404"),
            PageResult.STATUS_PENDING:    ("#f8f9fa", "#333"),
            PageResult.STATUS_SKIPPED:    ("#e2e3e5", "#555"),
        }
        bg, fg = color_map.get(status, ("#fff", "#000"))
        self._results_listbox.itemconfig(idx, bg=bg, fg=fg)

    def _on_result_select(self, event):
        sel = self._results_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.page_results):
            return
        result = self.page_results[idx]
        self._result_text.delete(1.0, tk.END)

        header = (
            f"Page: {result.page_num}  |  Status: {result.status}  |  "
            f"Duration: {result.duration_s:.2f}s  |  Attempts: {result.attempt + 1}\n"
            f"Source: {Path(result.pdf_path).name}\n"
        )
        if result.repeated:
            header += "[WARNING: repeat patterns detected in output]\n"
        if result.error_msg:
            header += f"Error: {result.error_msg}\n"
        header += "-" * 60 + "\n\n"

        self._result_text.insert(tk.END, header)
        self._result_text.insert(tk.END, result.display_text)

    def _copy_result(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self._result_text.get(1.0, tk.END))

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def _export(self, fmt: str):
        if not self.page_results:
            messagebox.showwarning("No Data", "No extraction results to export.")
            return

        ext_map = {"json": ".json", "csv": ".csv", "xlsx": ".xlsx", "md": ".md"}
        ext = ext_map.get(fmt, ".txt")
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(fmt.upper(), f"*{ext}"), ("All", "*")],
        )
        if not path:
            return

        try:
            if fmt == "json":
                payload = [r.to_dict() for r in self.page_results]
                Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            elif fmt in ("csv", "xlsx"):
                ext_mode = self.extraction_mode.get()
                if ext_mode == "custom":
                    all_rows = []
                    for r in self.page_results:
                        for row in r.structured_data:
                            row["_page"] = r.page_num
                            row["_source"] = Path(r.pdf_path).name
                            all_rows.append(row)
                    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
                else:
                    rows = []
                    for r in self.page_results:
                        rows.append({
                            "source": Path(r.pdf_path).name,
                            "page": r.page_num,
                            "status": r.status,
                            "language": r.parsed.get("primary_language"),
                            "is_table": r.parsed.get("is_table"),
                            "is_diagram": r.parsed.get("is_diagram"),
                            "rotation_correction": r.parsed.get("rotation_correction"),
                            "text": (r.parsed.get("natural_text") or "")[:500],
                            "attempts": r.attempt + 1,
                            "duration_s": r.duration_s,
                        })
                    df = pd.DataFrame(rows)

                if fmt == "csv":
                    df.to_csv(path, index=False)
                else:
                    df.to_excel(path, index=False)

            elif fmt == "md":
                lines = ["# olmOCR Extraction Results\n"]
                lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
                for r in self.page_results:
                    lines.append(f"## Page {r.page_num} — {Path(r.pdf_path).name}\n")
                    if r.status == PageResult.STATUS_DONE:
                        lines.append(r.display_text)
                    else:
                        lines.append(f"*{r.status}*")
                    lines.append("\n\n---\n\n")
                Path(path).write_text("\n".join(lines), encoding="utf-8")

            self._sys_msg(f"Exported {fmt.upper()}: {Path(path).name}")

        except Exception as e:
            self._err_msg(f"Export error: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    root = tk.Tk()
    app = OlmOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
