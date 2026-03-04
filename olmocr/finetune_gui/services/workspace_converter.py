"""
Thin wrapper around olmocr.data.prepare_workspace.process_workspace.

Bridges the prepare_workspace tqdm-based progress into our log_fn / progress_fn
callback model and populates the DatasetManifest with the resulting page entries.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

from olmocr.data.prepare_workspace import (
    load_jsonl_files,
    parse_jsonl_entry,
    process_document,
)

from olmocr.finetune_gui.services.manifest import DatasetManifest, PageEntry


def convert_workspace(
    workspace: str,
    prepared_dir: str,
    manifest: DatasetManifest,
    max_examples: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    log_fn: Callable[[str], None] = print,
    progress_fn: Callable[[int, int], None] = lambda c, t: None,
) -> int:
    """
    Read workspace/results/*.jsonl, produce single-page PDFs + .md pairs
    under *prepared_dir*, and register every new page in *manifest*.

    Returns the number of successfully written pages.
    """
    workspace_path = Path(workspace)
    results_dir = workspace_path / "results"
    output_dir = Path(prepared_dir)
    cache_dir = output_dir / ".pdf_cache"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    jsonl_files = load_jsonl_files(results_dir)
    if not jsonl_files:
        log_fn(f"[ERROR] No JSONL files found in {results_dir}")
        return 0

    # Collect all entries
    all_entries = []
    for jf in jsonl_files:
        log_fn(f"Reading {jf.name} …")
        with jf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                import json
                try:
                    entry = parse_jsonl_entry(json.loads(line))
                    if entry:
                        all_entries.append(entry)
                except Exception as exc:
                    log_fn(f"  [WARN] Skipping malformed line: {exc}")

    if max_examples and len(all_entries) > max_examples:
        all_entries = all_entries[:max_examples]
        log_fn(f"Capped to {max_examples} documents.")

    log_fn(f"Processing {len(all_entries)} document(s) …")
    total_ok = 0
    total_fail = 0

    for idx, entry in enumerate(all_entries):
        if cancel_event and cancel_event.is_set():
            log_fn("Cancelled.")
            break

        doc_id = entry.get("id", "")
        source = entry.get("source_file", "")
        log_fn(f"[{idx + 1}/{len(all_entries)}] {Path(source).name}")

        ok, fail = process_document(entry, output_dir, cache_dir)
        total_ok += ok
        total_fail += fail

        # Register new entries in manifest
        _register_pages(entry, output_dir, manifest, source)

        progress_fn(idx + 1, len(all_entries))
        log_fn(f"  ✓ {ok} pages written, {fail} failed")

    manifest.save()
    log_fn(f"\nConversion done — {total_ok} pages OK, {total_fail} failed.")
    return total_ok


def _register_pages(
    entry: dict,
    output_dir: Path,
    manifest: DatasetManifest,
    source_pdf: str,
) -> None:
    """Add or update manifest entries for every page produced by process_document."""
    doc_id = entry.get("id", "")
    pdf_page_numbers = entry.get("pdf_page_numbers", [])

    subdir = doc_id[:4] if len(doc_id) >= 4 else "misc"
    doc_dir = output_dir / subdir

    for _start, _end, page_num in pdf_page_numbers:
        base = f"{doc_id}_page{page_num}"
        pdf_p = doc_dir / f"{base}.pdf"
        md_p = doc_dir / f"{base}.md"

        if not pdf_p.exists() or not md_p.exists():
            continue  # process_document may have failed for this page

        key = base
        existing = manifest.get(key)
        if existing is None:
            manifest.upsert(
                PageEntry(
                    key=key,
                    source_pdf=source_pdf,
                    page_num=page_num,
                    doc_id=doc_id,
                    pdf_path=str(pdf_p),
                    md_path=str(md_p),
                    status="prepared",
                    split="train",
                )
            )
        else:
            existing.pdf_path = str(pdf_p)
            existing.md_path = str(md_p)
            if existing.status == "extracted":
                existing.status = "prepared"
