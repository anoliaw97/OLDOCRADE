"""
olmOCR Fine-Tuning Dataset GUI
================================
A Gradio-based tool for building reviewed, training-ready datasets for olmOCR.

Workflow (five tabs):
  1. Input          — pick PDFs, set workspace / output directories
  2. Extract        — run local VLM inference → workspace/results/*.jsonl
  3. Convert        — call prepare_workspace → single-page .pdf + .md pairs
  4. Review & Edit  — QA, correct YAML metadata and markdown, mark skips
  5. Export         — split into train/ eval/, validate, package

Run:
    python -m olmocr.finetune_gui       # or
    olmocr-finetune-gui
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Gradio — optional import with a helpful error message
# ---------------------------------------------------------------------------
try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is required for the fine-tuning GUI.\n"
        "Install it with:\n"
        "  pip install 'olmocr[finetune_gui]'\n"
        "or:\n"
        "  pip install gradio>=4.0"
    )

from PIL import Image

from olmocr.finetune_gui.services.dataset_validator import (
    ValidationReport,
    validate_directory,
    validate_export_split,
)
from olmocr.finetune_gui.services.job_runner import JobRunner
from olmocr.finetune_gui.services.local_extractor import extract_pdfs_to_workspace
from olmocr.finetune_gui.services.manifest import DatasetManifest, PageEntry
from olmocr.finetune_gui.services.workspace_converter import convert_workspace
from olmocr.finetune_gui.services.yaml_utils import apply_patches_to_md, read_md, write_md

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BASE = str(Path.cwd() / "finetune_runs")
_DEFAULT_VLM = "allenai/olmOCR-7B-0924-preview"


def _default_run_dir() -> str:
    return str(Path(_DEFAULT_BASE) / datetime.now().strftime("%Y%m%d_%H%M%S"))


def _render_page_image(pdf_path: str) -> Optional[Image.Image]:
    """Render first (only) page of a single-page PDF to a PIL Image."""
    try:
        from olmocr.data.renderpdf import render_pdf_to_base64png
        import base64
        from io import BytesIO
        b64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=900)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=120, first_page=1, last_page=1)
        return pages[0].convert("RGB") if pages else None
    except Exception:
        return None


# _read_md, _write_md, _apply_patches_to_md are re-exported from yaml_utils
_read_md = read_md
_write_md = write_md
_apply_patches_to_md = apply_patches_to_md


# ---------------------------------------------------------------------------
# Global state (shared across all Gradio event handlers)
# ---------------------------------------------------------------------------

_extractor_runner = JobRunner()
_converter_runner = JobRunner()
_export_runner = JobRunner()

# Current session state (module-level dicts for simplicity)
_state: Dict = {
    "run_dir": "",
    "workspace": "",
    "prepared_dir": "",
    "manifest": None,          # DatasetManifest | None
    "page_keys": [],           # ordered list of page keys for review tab
    "review_idx": 0,
}


def _manifest() -> Optional[DatasetManifest]:
    return _state.get("manifest")


def _ensure_dirs(run_dir: str) -> Tuple[str, str]:
    """Return (workspace, prepared_dir) under run_dir."""
    rp = Path(run_dir)
    ws = str(rp / "workspace")
    pd = str(rp / "prepared_dataset")
    Path(ws).mkdir(parents=True, exist_ok=True)
    Path(pd).mkdir(parents=True, exist_ok=True)
    return ws, pd


# ---------------------------------------------------------------------------
# Tab 1 — Input handlers
# ---------------------------------------------------------------------------

def on_init_session(run_dir: str, vlm_model: str) -> Tuple[str, str, str, str]:
    """Set up directories and load or create manifest."""
    if not run_dir:
        run_dir = _default_run_dir()
    _state["run_dir"] = run_dir
    ws, pd = _ensure_dirs(run_dir)
    _state["workspace"] = ws
    _state["prepared_dir"] = pd
    _state["vlm_model"] = vlm_model or _DEFAULT_VLM

    # Load existing manifest if present
    manifest = DatasetManifest.load(Path(run_dir))
    _state["manifest"] = manifest

    counts = manifest.counts()
    summary = (
        f"Run dir : {run_dir}\n"
        f"Workspace : {ws}\n"
        f"Prepared  : {pd}\n"
        f"Manifest  : {len(manifest)} entries — {counts}"
    )
    return run_dir, ws, pd, summary


# ---------------------------------------------------------------------------
# Tab 2 — Extract handlers
# ---------------------------------------------------------------------------

def start_extraction(
    pdf_files,         # list of file-upload dicts OR paths
    run_dir: str,
    use_existing: bool,
    max_pages_str: str,
    vlm_model: str,
) -> gr.update:
    """Validate inputs and return an error string if something is missing."""
    if not run_dir:
        return gr.update(value="⚠️  Set a Run Directory in the Input tab first.\n")
    if not use_existing and not pdf_files:
        return gr.update(value="⚠️  Select PDF files or enable 'Use existing workspace'.\n")
    return gr.update(value="Starting …\n")


def run_extraction_generator(
    pdf_files,
    run_dir: str,
    use_existing: bool,
    max_pages_str: str,
    vlm_model: str,
):
    """Gradio generator: start extraction in background, stream logs."""
    if _extractor_runner.is_running:
        yield "⚠️  Extraction already running.\n"
        return

    if not run_dir:
        yield "⚠️  Set Run Directory first.\n"
        return

    # Re-initialise directories / manifest
    on_init_session(run_dir, vlm_model)
    ws = _state["workspace"]

    if use_existing:
        yield f"Using existing workspace: {ws}\n"
        return

    if not pdf_files:
        yield "⚠️  No PDF files selected.\n"
        return

    # Normalise Gradio file upload objects to plain string paths
    paths: List[str] = []
    for f in pdf_files:
        p = f.name if hasattr(f, "name") else str(f)
        if p.lower().endswith(".pdf"):
            paths.append(p)

    if not paths:
        yield "⚠️  No PDF files found in selection.\n"
        return

    try:
        max_pages = int(max_pages_str) if max_pages_str.strip() else None
    except ValueError:
        max_pages = None

    model = vlm_model.strip() or _DEFAULT_VLM

    yield from _extractor_runner.run_and_stream(
        extract_pdfs_to_workspace,
        paths,
        ws,
        model_name=model,
        max_pages=max_pages,
    )

    # Persist updated manifest
    m = _manifest()
    if m:
        m.save()


def cancel_extraction() -> str:
    _extractor_runner.cancel()
    return "Cancel requested …\n"


# ---------------------------------------------------------------------------
# Tab 3 — Convert handlers
# ---------------------------------------------------------------------------

def run_convert_generator(run_dir: str, max_examples_str: str):
    """Stream logs from the workspace → prepared_dataset conversion."""
    if _converter_runner.is_running:
        yield "⚠️  Conversion already running.\n"
        return

    if not run_dir:
        yield "⚠️  Set Run Directory first.\n"
        return

    on_init_session(run_dir, _state.get("vlm_model", _DEFAULT_VLM))
    ws = _state["workspace"]
    pd = _state["prepared_dir"]
    manifest = _manifest()

    try:
        max_ex = int(max_examples_str) if max_examples_str.strip() else None
    except ValueError:
        max_ex = None

    yield from _converter_runner.run_and_stream(
        convert_workspace,
        ws,
        pd,
        manifest,
        max_examples=max_ex,
    )

    # Refresh page list for review tab
    _refresh_page_keys()


def cancel_convert() -> str:
    _converter_runner.cancel()
    return "Cancel requested …\n"


def get_conversion_stats(run_dir: str) -> str:
    if not run_dir:
        return "No run directory set."
    pd = str(Path(run_dir) / "prepared_dataset")
    if not Path(pd).exists():
        return "Prepared dataset directory does not exist yet."

    report = validate_directory(pd)
    return report.summary


# ---------------------------------------------------------------------------
# Tab 4 — Review & Edit handlers
# ---------------------------------------------------------------------------

def _refresh_page_keys() -> None:
    m = _manifest()
    if m is None:
        _state["page_keys"] = []
        return
    _state["page_keys"] = [
        e.key for e in m.all()
        if e.status in ("prepared", "reviewed")
    ]
    _state["review_idx"] = 0


def get_page_list() -> List[str]:
    _refresh_page_keys()
    m = _manifest()
    if m is None:
        return []
    rows = []
    for e in m.all():
        if e.status in ("prepared", "reviewed", "skipped"):
            flag = "⏭" if e.status == "skipped" else ("✓" if e.status == "reviewed" else "○")
            rows.append(f"{flag} [{e.split}] {e.key}")
    return rows


def load_page_for_review(selected: str):
    """
    Given a page-list row string, load the page data for the review panel.
    Returns: (image, language, rotation_valid, rotation_correction, is_table, is_diagram, body_text, status_msg)
    """
    if not selected:
        return None, "en", True, 0, False, False, "", "No page selected."

    # Extract key from the formatted string "○ [train] abc123_page1"
    parts = selected.strip().split()
    key = parts[-1] if parts else ""

    m = _manifest()
    if m is None or not key:
        return None, "en", True, 0, False, False, "", "Manifest not loaded."

    entry = m.get(key)
    if entry is None:
        return None, "en", True, 0, False, False, "", f"Entry '{key}' not found."

    # Apply any existing patches to a temp copy for display
    md_path = entry.md_path
    if not Path(md_path).exists():
        return None, "en", True, 0, False, False, "", f"MD file not found: {md_path}"

    # Build effective yaml by merging stored patches
    yaml_vals, body = _read_md(md_path)
    for field, val in entry.review_patches.items():
        if field == "body":
            body = val
        else:
            yaml_vals[field] = val

    # Render PDF image
    img = None
    if entry.pdf_path and Path(entry.pdf_path).exists():
        img = _render_page_image(entry.pdf_path)

    status = f"Page: {entry.key}  |  Status: {entry.status}  |  Split: {entry.split}"
    lang = yaml_vals.get("primary_language") or "en"
    return (
        img,
        str(lang),
        bool(yaml_vals.get("is_rotation_valid", True)),
        int(yaml_vals.get("rotation_correction", 0)),
        bool(yaml_vals.get("is_table", False)),
        bool(yaml_vals.get("is_diagram", False)),
        body,
        status,
    )


def save_page_review(
    selected: str,
    lang: str,
    rot_valid: bool,
    rot_correction: int,
    is_table: bool,
    is_diagram: bool,
    body: str,
) -> str:
    """Persist user edits back to the .md file and manifest."""
    parts = selected.strip().split()
    key = parts[-1] if parts else ""
    m = _manifest()
    if m is None or not key:
        return "⚠️  Cannot save — manifest not loaded."

    entry = m.get(key)
    if entry is None:
        return f"⚠️  Entry '{key}' not found."

    yaml_vals = {
        "primary_language": lang.strip() or None,
        "is_rotation_valid": rot_valid,
        "rotation_correction": int(rot_correction),
        "is_table": is_table,
        "is_diagram": is_diagram,
    }

    # Write directly to the .md file
    _write_md(entry.md_path, yaml_vals, body)

    # Record patches in manifest (for export consistency)
    entry.review_patches = dict(yaml_vals)
    entry.review_patches["body"] = body
    entry.status = "reviewed"
    m.save()

    return f"✓ Saved {key}"


def mark_skip(selected: str) -> Tuple[str, List[str]]:
    parts = selected.strip().split()
    key = parts[-1] if parts else ""
    m = _manifest()
    if m and key:
        m.update_status(key, "skipped")
        m.save()
        return f"Marked '{key}' as skipped.", get_page_list()
    return "Nothing to skip.", get_page_list()


def set_page_split(selected: str, split: str) -> str:
    parts = selected.strip().split()
    key = parts[-1] if parts else ""
    m = _manifest()
    if m and key:
        m.set_split(key, split)
        m.save()
        return f"Set '{key}' → {split}"
    return "No page selected."


def apply_split_strategy(strategy: str, eval_pct_str: str) -> str:
    m = _manifest()
    if m is None:
        return "Manifest not loaded."
    try:
        pct = float(eval_pct_str)
    except ValueError:
        return "Invalid eval percentage."
    m.apply_split_strategy(strategy, eval_pct=pct)
    m.save()
    counts = m.split_counts()
    return f"Splits updated: {counts}"


# ---------------------------------------------------------------------------
# Tab 5 — Export handlers
# ---------------------------------------------------------------------------

def run_export_generator(
    run_dir: str,
    export_dir: str,
    export_mode: str,       # "all" | "reviewed_only"
    split_strategy: str,
    eval_pct_str: str,
    wipe_existing: bool,
):
    """Export train/ + eval/ directories with matched .pdf + .md pairs."""
    if _export_runner.is_running:
        yield "⚠️  Export already running.\n"
        return

    if not run_dir:
        yield "⚠️  Set Run Directory first.\n"
        return

    on_init_session(run_dir, _state.get("vlm_model", _DEFAULT_VLM))
    m = _manifest()
    if m is None or len(m) == 0:
        yield "⚠️  No pages in manifest. Run Convert first.\n"
        return

    if not export_dir:
        export_dir = str(Path(run_dir) / "export")

    try:
        eval_pct = float(eval_pct_str)
    except ValueError:
        eval_pct = 0.1

    yield from _export_runner.run_and_stream(
        _do_export,
        m,
        export_dir,
        export_mode,
        split_strategy,
        eval_pct,
        wipe_existing,
    )


def _do_export(
    manifest: DatasetManifest,
    export_dir: str,
    export_mode: str,
    split_strategy: str,
    eval_pct: float,
    wipe_existing: bool,
    cancel_event: threading.Event,
    log_fn,
    progress_fn,
) -> None:
    """Worker function (runs inside JobRunner thread)."""
    # Apply split strategy first
    manifest.apply_split_strategy(split_strategy, eval_pct=eval_pct)
    manifest.save()
    log_fn(f"Split strategy '{split_strategy}', eval={eval_pct} applied.")
    log_fn(f"Splits: {manifest.split_counts()}")

    export_path = Path(export_dir)
    if wipe_existing and export_path.exists():
        shutil.rmtree(export_path)
        log_fn("Wiped existing export directory.")

    (export_path / "train").mkdir(parents=True, exist_ok=True)
    (export_path / "eval").mkdir(parents=True, exist_ok=True)

    # Select entries to export
    statuses = ("prepared", "reviewed") if export_mode == "reviewed_only" else ("prepared", "reviewed", "exported")
    entries = manifest.by_status(*statuses)
    log_fn(f"Exporting {len(entries)} pages (mode={export_mode}) …")

    ok = fail = 0
    for i, entry in enumerate(entries):
        if cancel_event.is_set():
            log_fn("Cancelled.")
            break

        split = entry.split or "train"
        dest_dir = export_path / split
        stem = entry.key

        src_pdf = Path(entry.pdf_path)
        src_md = Path(entry.md_path)

        if not src_pdf.exists() or not src_md.exists():
            log_fn(f"  [SKIP] Missing file(s) for {stem}")
            fail += 1
            continue

        dst_pdf = dest_dir / f"{stem}.pdf"
        dst_md = dest_dir / f"{stem}.md"

        try:
            shutil.copy2(src_pdf, dst_pdf)
            # Apply any in-memory patches that weren't written to .md yet
            shutil.copy2(src_md, dst_md)
            if entry.review_patches:
                _apply_patches_to_md(str(dst_md), entry.review_patches)

            entry.status = "exported"
            ok += 1
        except Exception as exc:
            log_fn(f"  [ERROR] {stem}: {exc}")
            fail += 1

        progress_fn(i + 1, len(entries))

    manifest.save()
    log_fn(f"\nExport done: {ok} OK, {fail} failed → {export_dir}")

    # Validate the export
    log_fn("\nValidating export …")
    report = validate_export_split(export_dir)
    log_fn(report.summary)


def get_export_stats(run_dir: str, export_dir: str) -> str:
    if not export_dir:
        export_dir = str(Path(run_dir) / "export") if run_dir else ""
    if not export_dir or not Path(export_dir).exists():
        return "Export directory does not exist yet."
    report = validate_export_split(export_dir)
    return report.summary


# ---------------------------------------------------------------------------
# Training config generator
# ---------------------------------------------------------------------------

def generate_training_config(run_dir: str, export_dir: str, model_name: str) -> str:
    """Return a YAML training config snippet for the exported dataset."""
    if not export_dir:
        export_dir = str(Path(run_dir) / "export") if run_dir else "./export"
    train_dir = str(Path(export_dir) / "train")
    eval_dir = str(Path(export_dir) / "eval")
    model = model_name.strip() or "Qwen/Qwen2.5-VL-7B-Instruct"
    return f"""\
# Auto-generated training config snippet
# Place inside your full olmOCR training YAML (e.g. v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml)

model:
  name: {model}
  torch_dtype: bfloat16
  use_flash_attention: true

dataset:
  train:
    - name: my_finetune_train
      root_dir: {train_dir}
      pipeline:
        - name: FrontMatterParser
          front_matter_class: PageResponse
        - name: FilterOutRotatedDocuments
        - name: ReformatLatexBoldItalic
        - name: DatasetTextRuleFilter
        - name: PDFRenderer
          target_longest_image_dim: 1288
        - name: RotationAugmentation
          probability: 0.02
        - name: NewYamlFinetuningPromptWithNoAnchoring
        - name: FrontMatterOutputFormat
        - name: InstructUserMessages
          prompt_first: true
        - name: Tokenizer
          masking_index: -100
          end_of_message_token: "<|im_end|>"

  eval:
    - name: my_finetune_eval
      root_dir: {eval_dir}
      pipeline:
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

training:
  output_dir: ./olmocr-finetuned
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  learning_rate: 2e-5
"""


# ---------------------------------------------------------------------------
# Gradio app layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="olmOCR Fine-Tuning Dataset GUI",
        theme=gr.themes.Soft(),
        css="""
            .log-box textarea { font-family: monospace; font-size: 12px; }
            .section-header { font-weight: bold; margin-top: 12px; }
        """,
    ) as demo:

        gr.Markdown("# 🔬 olmOCR Fine-Tuning Dataset GUI")
        gr.Markdown(
            "Build, review, and export training-ready datasets for olmOCR fine-tuning. "
            "Work through the tabs left to right."
        )

        # ================================================================
        # Tab 1: Input
        # ================================================================
        with gr.Tab("1 · Input"):
            gr.Markdown("### Session & Directory Setup")
            with gr.Row():
                with gr.Column(scale=2):
                    run_dir_box = gr.Textbox(
                        label="Run Directory (will be created if absent)",
                        value=_default_run_dir(),
                        placeholder="./finetune_runs/20250101_120000",
                    )
                    vlm_model_box = gr.Textbox(
                        label="VLM Model (HuggingFace ID)",
                        value=_DEFAULT_VLM,
                    )
                    init_btn = gr.Button("⚡ Initialise / Resume Session", variant="primary")

                with gr.Column(scale=1):
                    session_info = gr.Textbox(
                        label="Session Info", lines=8, interactive=False
                    )

            init_btn.click(
                fn=on_init_session,
                inputs=[run_dir_box, vlm_model_box],
                outputs=[run_dir_box, gr.Textbox(visible=False), gr.Textbox(visible=False), session_info],
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Next step →** Go to the **Extract** tab to run VLM inference on your PDFs, "
                "or go to **Convert** if you already have a workspace."
            )

        # ================================================================
        # Tab 2: Extract
        # ================================================================
        with gr.Tab("2 · Extract (Workspace)"):
            gr.Markdown("### Run olmOCR local inference on your PDFs")
            gr.Markdown(
                "Produces `workspace/results/*.jsonl` (Dolma format) which the Convert step reads. "
                "Skips PDFs that already have a result file (safe to re-run / resume)."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_upload = gr.File(
                        label="Input PDFs",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    use_existing_chk = gr.Checkbox(
                        label="Skip extraction — use existing workspace",
                        value=False,
                    )
                    max_pages_box = gr.Textbox(
                        label="Max pages (blank = all, use small number for quick tests)",
                        value="",
                        placeholder="e.g. 50",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**Controls**")
                    extract_btn = gr.Button("▶ Start Extraction", variant="primary")
                    cancel_ext_btn = gr.Button("■ Cancel", variant="stop")
                    ext_progress = gr.Textbox(
                        label="Progress",
                        value="Ready.",
                        interactive=False,
                        lines=2,
                    )

            extract_log = gr.Textbox(
                label="Extraction Log",
                lines=20,
                interactive=False,
                elem_classes=["log-box"],
            )

            extract_btn.click(
                fn=run_extraction_generator,
                inputs=[pdf_upload, run_dir_box, use_existing_chk, max_pages_box, vlm_model_box],
                outputs=[extract_log],
            )
            cancel_ext_btn.click(fn=cancel_extraction, outputs=[extract_log])

        # ================================================================
        # Tab 3: Convert
        # ================================================================
        with gr.Tab("3 · Convert to Training Format"):
            gr.Markdown("### Prepare workspace → single-page PDF + .md pairs")
            gr.Markdown(
                "Calls `olmocr.data.prepare_workspace.process_workspace` to split each "
                "document into individual pages. Each page becomes one `{key}.pdf` + `{key}.md`."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    max_ex_box = gr.Textbox(
                        label="Max documents (blank = all)",
                        value="",
                        placeholder="e.g. 100",
                    )
                with gr.Column(scale=1):
                    convert_btn = gr.Button("▶ Prepare Dataset", variant="primary")
                    cancel_conv_btn = gr.Button("■ Cancel", variant="stop")

            convert_log = gr.Textbox(
                label="Conversion Log",
                lines=16,
                interactive=False,
                elem_classes=["log-box"],
            )

            gr.Markdown("### Validation")
            with gr.Row():
                validate_btn = gr.Button("🔍 Validate Prepared Dataset")
                validate_out = gr.Textbox(label="Validation Report", lines=8, interactive=False)

            convert_btn.click(
                fn=run_convert_generator,
                inputs=[run_dir_box, max_ex_box],
                outputs=[convert_log],
            )
            cancel_conv_btn.click(fn=cancel_convert, outputs=[convert_log])
            validate_btn.click(
                fn=get_conversion_stats,
                inputs=[run_dir_box],
                outputs=[validate_out],
            )

        # ================================================================
        # Tab 4: Review & Edit
        # ================================================================
        with gr.Tab("4 · Review & Edit"):
            gr.Markdown("### QA — correct YAML metadata and extracted text")

            with gr.Row():
                # Left: page list
                with gr.Column(scale=1):
                    refresh_list_btn = gr.Button("🔄 Refresh List")
                    page_listbox = gr.Dropdown(
                        label="Pages",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    gr.Markdown("**Split assignment**")
                    split_radio = gr.Radio(
                        choices=["train", "eval"],
                        label="Set split for selected page",
                        value="train",
                    )
                    set_split_btn = gr.Button("Set Split")
                    skip_btn = gr.Button("⏭ Mark as Skip", variant="stop")
                    split_msg = gr.Textbox(label="", lines=1, interactive=False)

                # Middle: PDF preview
                with gr.Column(scale=2):
                    page_image = gr.Image(
                        label="PDF Page Preview",
                        type="pil",
                        interactive=False,
                        height=600,
                    )
                    page_status_box = gr.Textbox(
                        label="Page Info", lines=1, interactive=False
                    )

                # Right: editors
                with gr.Column(scale=2):
                    gr.Markdown("**YAML Metadata**")
                    lang_box = gr.Textbox(label="primary_language (e.g. 'en', leave blank for null)", value="en")
                    rot_valid_chk = gr.Checkbox(label="is_rotation_valid", value=True)
                    rot_corr_box = gr.Dropdown(
                        label="rotation_correction",
                        choices=[0, 90, 180, 270],
                        value=0,
                    )
                    table_chk = gr.Checkbox(label="is_table", value=False)
                    diagram_chk = gr.Checkbox(label="is_diagram", value=False)
                    gr.Markdown("**Extracted Text (Markdown)**")
                    body_editor = gr.Textbox(
                        label="",
                        lines=14,
                        max_lines=30,
                        placeholder="Markdown content …",
                    )
                    save_btn = gr.Button("💾 Save", variant="primary")
                    save_msg = gr.Textbox(label="", lines=1, interactive=False)

            # Auto-fill split strategy
            gr.Markdown("---")
            gr.Markdown("### Bulk split assignment")
            with gr.Row():
                strat_radio = gr.Radio(
                    choices=["random_pct", "first_N"],
                    label="Strategy",
                    value="random_pct",
                )
                eval_pct_strat_box = gr.Textbox(
                    label="Eval fraction (random_pct) or count (first_N)",
                    value="0.1",
                )
                apply_split_btn = gr.Button("Apply")
                split_strat_msg = gr.Textbox(label="", lines=1, interactive=False)

            # Wire up
            refresh_list_btn.click(fn=get_page_list, outputs=[page_listbox])
            page_listbox.change(
                fn=load_page_for_review,
                inputs=[page_listbox],
                outputs=[
                    page_image, lang_box, rot_valid_chk,
                    rot_corr_box, table_chk, diagram_chk,
                    body_editor, page_status_box,
                ],
            )
            save_btn.click(
                fn=save_page_review,
                inputs=[page_listbox, lang_box, rot_valid_chk, rot_corr_box,
                        table_chk, diagram_chk, body_editor],
                outputs=[save_msg],
            )
            skip_btn.click(
                fn=mark_skip,
                inputs=[page_listbox],
                outputs=[split_msg, page_listbox],
            )
            set_split_btn.click(
                fn=set_page_split,
                inputs=[page_listbox, split_radio],
                outputs=[split_msg],
            )
            apply_split_btn.click(
                fn=apply_split_strategy,
                inputs=[strat_radio, eval_pct_strat_box],
                outputs=[split_strat_msg],
            )

        # ================================================================
        # Tab 5: Export
        # ================================================================
        with gr.Tab("5 · Export"):
            gr.Markdown("### Export train/ + eval/ splits for olmOCR training")

            with gr.Row():
                with gr.Column(scale=2):
                    export_dir_box = gr.Textbox(
                        label="Export Directory (default: <run_dir>/export)",
                        value="",
                        placeholder="./finetune_runs/20250101_120000/export",
                    )
                    export_mode_radio = gr.Radio(
                        choices=["all", "reviewed_only"],
                        label="Which pages to export",
                        value="all",
                    )
                    wipe_chk = gr.Checkbox(
                        label="Wipe existing export directory first",
                        value=False,
                    )
                    gr.Markdown("**Split strategy** (applied at export time; overrides per-page assignments)")
                    exp_strat_radio = gr.Radio(
                        choices=["random_pct", "first_N"],
                        label="Strategy",
                        value="random_pct",
                    )
                    exp_eval_pct_box = gr.Textbox(
                        label="Eval fraction / count",
                        value="0.1",
                    )

                with gr.Column(scale=1):
                    export_btn = gr.Button("📦 Export", variant="primary")
                    cancel_exp_btn = gr.Button("■ Cancel", variant="stop")
                    exp_validate_btn = gr.Button("🔍 Validate Export")
                    exp_validate_out = gr.Textbox(
                        label="Validation", lines=8, interactive=False
                    )

            export_log = gr.Textbox(
                label="Export Log",
                lines=14,
                interactive=False,
                elem_classes=["log-box"],
            )

            gr.Markdown("---")
            gr.Markdown("### Training Config Snippet")
            with gr.Row():
                cfg_model_box = gr.Textbox(
                    label="Base model for config",
                    value="Qwen/Qwen2.5-VL-7B-Instruct",
                )
                gen_cfg_btn = gr.Button("Generate Config")
            training_cfg_box = gr.Code(
                label="Paste this into your training YAML",
                language="yaml",
                lines=30,
            )

            export_btn.click(
                fn=run_export_generator,
                inputs=[
                    run_dir_box, export_dir_box, export_mode_radio,
                    exp_strat_radio, exp_eval_pct_box, wipe_chk,
                ],
                outputs=[export_log],
            )
            cancel_exp_btn.click(fn=lambda: _export_runner.cancel(), outputs=[])
            exp_validate_btn.click(
                fn=get_export_stats,
                inputs=[run_dir_box, export_dir_box],
                outputs=[exp_validate_out],
            )
            gen_cfg_btn.click(
                fn=generate_training_config,
                inputs=[run_dir_box, export_dir_box, cfg_model_box],
                outputs=[training_cfg_box],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="olmOCR Fine-Tuning Dataset GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    args = parser.parse_args()

    demo = build_app()
    demo.queue()   # enable Gradio queue for generator support
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
