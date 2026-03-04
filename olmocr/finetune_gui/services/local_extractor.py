"""
Local VLM extractor for the fine-tuning GUI.

Runs olmOCR model inference locally (via transformers, no vLLM required)
and writes output in the exact workspace JSONL format that
`olmocr.data.prepare_workspace` expects.

Output layout:
    <workspace>/
        results/
            output_<sha1>.jsonl     ← one file per source PDF

Each JSONL line is a Dolma document::

    {
        "id": "<sha1>",
        "text": "<full_doc_text>",
        "source": "olmocr-local",
        "added": "YYYY-MM-DD",
        "created": "YYYY-MM-DD",
        "metadata": {
            "Source-File": "/abs/path/to.pdf",
            "olmocr-version": "local",
            "pdf-total-pages": N,
            ...
        },
        "attributes": {
            "pdf_page_numbers": [[start, end, page_num], ...],
            "primary_language": [...],
            "is_rotation_valid": [...],
            "rotation_correction": [...],
            "is_table": [...],
            "is_diagram": [...],
        }
    }
"""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import re
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

# ── olmocr native imports (with graceful fallback) ──────────────────────────
try:
    from olmocr.data.renderpdf import render_pdf_to_base64png as _render_b64
    _HAS_RENDERPDF = True
except ImportError:
    _HAS_RENDERPDF = False

try:
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    _PROMPT = build_no_anchoring_v4_yaml_prompt()
except ImportError:
    _PROMPT = (
        "Attached is one page of a document that you must process. "
        "Just return the plain text representation of this document as if you "
        "were reading it naturally. Convert equations to LaTeX and tables to HTML.\n"
        "Return your output as markdown, with a front matter section on top "
        "specifying values for the primary_language, is_rotation_valid, "
        "rotation_correction, is_table, and is_diagram parameters."
    )

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

# ── temperature schedule (mirrors the real pipeline) ────────────────────────
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

# ── YAML front-matter parser ─────────────────────────────────────────────────

def _parse_yaml_response(text: str) -> Dict:
    """Parse olmOCR YAML front-matter response into a structured dict."""
    defaults = {
        "primary_language": None,
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": text,
    }
    m = re.search(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL | re.MULTILINE)
    if not m:
        return defaults
    fm, body = m.group(1), m.group(2).strip()
    defaults["natural_text"] = body
    for line in fm.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip().lower()
        if key == "primary_language":
            defaults[key] = None if val in ("null", "none", "") else val
        elif key in ("is_rotation_valid", "is_table", "is_diagram"):
            defaults[key] = val in ("true", "yes", "1")
        elif key == "rotation_correction":
            try:
                defaults[key] = int(val)
            except ValueError:
                pass
    return defaults


def _count_pages(pdf_path: str) -> int:
    if _HAS_PYPDF:
        try:
            return len(PdfReader(pdf_path).pages)
        except Exception:
            pass
    # Fallback: pdf2image
    try:
        from pdf2image import convert_from_path
        return len(convert_from_path(pdf_path, dpi=72))
    except Exception:
        return 0


def _render_page(pdf_path: str, page_num: int) -> Image.Image:
    """Render a single PDF page (1-indexed) to a PIL Image."""
    if _HAS_RENDERPDF:
        try:
            b64 = _render_b64(pdf_path, page_num, target_longest_image_dim=1288)
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            pass
    from pdf2image import convert_from_path as c
    pages = c(pdf_path, dpi=150, first_page=page_num, last_page=page_num)
    return pages[0].convert("RGB") if pages else None


def _resize(img: Image.Image, target: int = 1288) -> Image.Image:
    w, h = img.size
    long = max(w, h)
    if long <= target:
        return img
    scale = target / long
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _detect_repeats(text: str, threshold: int = 5) -> bool:
    try:
        from olmocr.repeatdetect import RepeatDetector
        rd = RepeatDetector()
        return any(c >= threshold for c in rd.ngram_repeats(text))
    except Exception:
        pass
    words = text.split()
    if len(words) < 20:
        return False
    for size in (5, 10):
        chunks = [" ".join(words[i : i + size]) for i in range(0, len(words) - size, size)]
        for chunk in chunks:
            if chunk and chunks.count(chunk) > threshold:
                return True
    return False


# ── Model wrapper ─────────────────────────────────────────────────────────────

class _LocalVLM:
    """Lazy-loaded local VLM (singleton per model name)."""

    _instance: Optional["_LocalVLM"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.loaded = False

    @classmethod
    def get(cls, model_name: str) -> "_LocalVLM":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    def load(self, log_fn: Callable[[str], None] = print) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        log_fn(f"Loading VLM processor: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        log_fn("Loading VLM model weights …")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.loaded = True
        log_fn(f"VLM ready: {self.model_name}")

    def infer(self, image: Image.Image, prompt: str, temperature: float = 0.1, max_new_tokens: int = 2048) -> str:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img = _resize(image.convert("RGB"))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_in], images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.05),
            )
        decoded = self.processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        del inputs, out_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return decoded

    def infer_with_retry(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048) -> Tuple[str, int]:
        """Retry with escalating temperature to break repetition loops."""
        for attempt, temp in enumerate(TEMPERATURE_BY_ATTEMPT):
            result = self.infer(image, prompt, temperature=temp, max_new_tokens=max_new_tokens)
            if not _detect_repeats(result):
                return result, attempt
        return result, len(TEMPERATURE_BY_ATTEMPT) - 1


# ── Public extraction entry point ─────────────────────────────────────────────

def extract_pdfs_to_workspace(
    pdf_paths: List[str],
    workspace: str,
    model_name: str = "allenai/olmOCR-7B-0924-preview",
    max_pages: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda c, t: None,
) -> None:
    """
    Run local VLM inference on *pdf_paths* and write workspace JSONL output.

    Parameters
    ----------
    pdf_paths   : list of local PDF paths to process
    workspace   : output workspace directory (results/ subdir created here)
    model_name  : HuggingFace model ID for the VLM
    max_pages   : optional cap on total pages processed (for quick testing)
    cancel_event: threading.Event — set by caller to request cancellation
    log_fn      : callable(str) — emit a log line to the UI
    progress_fn : callable(current, total) — update progress counter
    """
    workspace_path = Path(workspace)
    results_dir = workspace_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    vlm = _LocalVLM.get(model_name)
    vlm.load(log_fn=log_fn)

    # Count total pages for progress
    page_totals = {}
    total_pages = 0
    for pdf in pdf_paths:
        n = min(_count_pages(pdf), max_pages or 9999999)
        page_totals[pdf] = n
        total_pages += n

    if max_pages:
        total_pages = min(total_pages, max_pages)

    log_fn(f"Total pages to extract: {total_pages} across {len(pdf_paths)} PDF(s)")

    processed = 0

    for pdf_path in pdf_paths:
        if cancel_event and cancel_event.is_set():
            log_fn("Cancelled.")
            break

        n_pages = page_totals[pdf_path]
        pdf_name = Path(pdf_path).name
        log_fn(f"\n▶ {pdf_name}  ({n_pages} pages)")

        # Skip if already done (resume support)
        stem = Path(pdf_path).stem
        existing = list(results_dir.glob(f"*{stem}*.jsonl"))
        if existing:
            log_fn(f"  ↩ Already extracted ({existing[0].name}) — skipping")
            processed += n_pages
            progress_fn(min(processed, total_pages), total_pages)
            continue

        pages_text: List[Tuple[int, str, Dict]] = []  # (page_num, text, parsed)

        for pg in range(1, n_pages + 1):
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled mid-document.")
                break
            if max_pages and processed >= max_pages:
                log_fn(f"  Reached max_pages={max_pages} cap.")
                break

            log_fn(f"  Page {pg}/{n_pages} …")
            try:
                img = _render_page(pdf_path, pg)
                if img is None:
                    log_fn(f"  [WARN] Could not render page {pg} — skipping")
                    continue
                raw, attempt = vlm.infer_with_retry(img, _PROMPT, max_new_tokens=2048)
                parsed = _parse_yaml_response(raw)
                if attempt > 0:
                    log_fn(f"  [WARN] Page {pg}: needed {attempt + 1} attempts (repeat avoidance)")
                pages_text.append((pg, parsed.get("natural_text", ""), parsed))
            except Exception as exc:
                log_fn(f"  [ERROR] Page {pg}: {exc}")

            processed += 1
            progress_fn(min(processed, total_pages), total_pages)

        if not pages_text:
            log_fn(f"  No pages extracted from {pdf_name} — skipping")
            continue

        # Build Dolma document
        full_text = ""
        page_boundaries: List[List] = []
        primary_languages: List = []
        is_rotation_valids: List[bool] = []
        rotation_corrections: List[int] = []
        is_tables: List[bool] = []
        is_diagrams: List[bool] = []

        for pg_num, pg_text, parsed in pages_text:
            start = len(full_text)
            full_text += pg_text
            end = len(full_text)
            full_text += "\n"
            page_boundaries.append([start, end, pg_num])
            primary_languages.append(parsed.get("primary_language"))
            is_rotation_valids.append(parsed.get("is_rotation_valid", True))
            rotation_corrections.append(parsed.get("rotation_correction", 0))
            is_tables.append(parsed.get("is_table", False))
            is_diagrams.append(parsed.get("is_diagram", False))

        doc_id = hashlib.sha1(full_text.encode()).hexdigest()
        today = datetime.now().strftime("%Y-%m-%d")

        dolma_doc = {
            "id": doc_id,
            "text": full_text,
            "source": "olmocr-local",
            "added": today,
            "created": today,
            "metadata": {
                "Source-File": str(Path(pdf_path).resolve()),
                "olmocr-version": "local",
                "pdf-total-pages": n_pages,
                "total-input-tokens": 0,
                "total-output-tokens": 0,
                "total-fallback-pages": 0,
            },
            "attributes": {
                "pdf_page_numbers": page_boundaries,
                "primary_language": primary_languages,
                "is_rotation_valid": is_rotation_valids,
                "rotation_correction": rotation_corrections,
                "is_table": is_tables,
                "is_diagram": is_diagrams,
            },
        }

        out_path = results_dir / f"output_{doc_id[:16]}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(dolma_doc, ensure_ascii=False) + "\n")

        log_fn(f"  ✓ {pdf_name} → {out_path.name}")

    log_fn("\nExtraction finished.")
