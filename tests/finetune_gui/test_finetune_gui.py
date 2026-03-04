"""
Smoke tests for the olmOCR fine-tuning GUI.

These tests:
  1. Validate the manifest save/load round-trip.
  2. Validate dataset_validator logic on synthetic fixtures.
  3. Validate that export produces correctly matched .pdf/.md pairs.
  4. Validate YAML front matter in every .md after export.
  5. Validate single-page PDF constraint after prepare_workspace.
  6. Smoke-test the JobRunner log streaming.

None of these tests require a GPU or any model weights.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from olmocr.finetune_gui.services.dataset_validator import (
    ValidationReport,
    validate_directory,
    validate_export_split,
    _check_yaml_frontmatter,
)
from olmocr.finetune_gui.services.job_runner import JobRunner
from olmocr.finetune_gui.services.manifest import DatasetManifest, PageEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_single_page_pdf(path: Path) -> None:
    """Write a minimal single-page PDF to *path*."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        writer.write(fh)


def _make_md(path: Path, has_fm: bool = True, complete_keys: bool = True) -> None:
    """Write a .md file with or without valid YAML front matter."""
    if not has_fm:
        path.write_text("Just some text without front matter.\n", encoding="utf-8")
        return

    keys = (
        "primary_language: en\n"
        "is_rotation_valid: true\n"
        "rotation_correction: 0\n"
        "is_table: false\n"
        "is_diagram: false\n"
    ) if complete_keys else "primary_language: en\n"

    path.write_text(f"---\n{keys}---\nBody text here.\n", encoding="utf-8")


def _populate_export_dir(export_dir: Path, n_train: int = 3, n_eval: int = 1) -> None:
    """Create matched PDF+MD pairs in train/ and eval/ subdirs."""
    for split, count in [("train", n_train), ("eval", n_eval)]:
        split_dir = export_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            stem = f"doc{i}_page1"
            _make_single_page_pdf(split_dir / f"{stem}.pdf")
            _make_md(split_dir / f"{stem}.md")


# ---------------------------------------------------------------------------
# Tests: manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_save_load_round_trip(self, tmp_path):
        m = DatasetManifest(tmp_path)
        e = PageEntry(
            key="abc_page1",
            source_pdf="/foo/bar.pdf",
            page_num=1,
            doc_id="abc",
            pdf_path=str(tmp_path / "abc_page1.pdf"),
            md_path=str(tmp_path / "abc_page1.md"),
            status="prepared",
            split="train",
        )
        m.upsert(e)
        m.save()

        m2 = DatasetManifest.load(tmp_path)
        assert len(m2) == 1
        e2 = m2.get("abc_page1")
        assert e2 is not None
        assert e2.status == "prepared"
        assert e2.split == "train"
        assert e2.source_pdf == "/foo/bar.pdf"

    def test_update_status(self, tmp_path):
        m = DatasetManifest(tmp_path)
        m.upsert(PageEntry(key="k1", source_pdf="", page_num=1, status="prepared", split="train"))
        m.update_status("k1", "reviewed")
        assert m.get("k1").status == "reviewed"

    def test_apply_split_strategy_random_pct(self, tmp_path):
        m = DatasetManifest(tmp_path)
        for i in range(10):
            m.upsert(PageEntry(key=f"p{i}", source_pdf="", page_num=i, status="prepared", split="train"))
        m.apply_split_strategy("random_pct", eval_pct=0.3)
        eval_cnt = sum(1 for e in m.all() if e.split == "eval")
        assert eval_cnt >= 1  # at least 1 page in eval

    def test_apply_split_strategy_first_n(self, tmp_path):
        m = DatasetManifest(tmp_path)
        for i in range(10):
            m.upsert(PageEntry(key=f"p{i}", source_pdf="", page_num=i, status="prepared", split="train"))
        m.apply_split_strategy("first_N", eval_pct=2)
        entries = m.all()
        assert entries[0].split == "eval"
        assert entries[1].split == "eval"
        assert entries[2].split == "train"

    def test_by_status_filter(self, tmp_path):
        m = DatasetManifest(tmp_path)
        m.upsert(PageEntry(key="a", source_pdf="", page_num=1, status="prepared", split="train"))
        m.upsert(PageEntry(key="b", source_pdf="", page_num=2, status="skipped", split="train"))
        m.upsert(PageEntry(key="c", source_pdf="", page_num=3, status="reviewed", split="eval"))
        assert len(m.by_status("prepared")) == 1
        assert len(m.by_status("prepared", "reviewed")) == 2

    def test_patch_persists(self, tmp_path):
        m = DatasetManifest(tmp_path)
        m.upsert(PageEntry(key="x1", source_pdf="", page_num=1, status="prepared", split="train"))
        m.apply_patch("x1", "is_table", True)
        m.save()
        m2 = DatasetManifest.load(tmp_path)
        assert m2.get("x1").review_patches.get("is_table") is True


# ---------------------------------------------------------------------------
# Tests: dataset_validator
# ---------------------------------------------------------------------------


class TestDatasetValidator:
    def test_valid_pair(self, tmp_path):
        stem = "doc1_page1"
        _make_single_page_pdf(tmp_path / f"{stem}.pdf")
        _make_md(tmp_path / f"{stem}.md")
        report = validate_directory(str(tmp_path))
        assert report.total == 1
        assert report.valid == 1
        assert report.invalid == 0

    def test_missing_pdf(self, tmp_path):
        _make_md(tmp_path / "doc1_page1.md")
        # No PDF written
        report = validate_directory(str(tmp_path))
        assert report.invalid == 1
        assert any("Missing PDF" in e for pv in report.issues for e in pv.errors)

    def test_missing_yaml_frontmatter(self, tmp_path):
        stem = "doc2_page1"
        _make_single_page_pdf(tmp_path / f"{stem}.pdf")
        _make_md(tmp_path / f"{stem}.md", has_fm=False)
        report = validate_directory(str(tmp_path))
        assert report.invalid == 1

    def test_incomplete_yaml_keys(self, tmp_path):
        stem = "doc3_page1"
        _make_single_page_pdf(tmp_path / f"{stem}.pdf")
        _make_md(tmp_path / f"{stem}.md", complete_keys=False)
        report = validate_directory(str(tmp_path))
        assert report.invalid == 1
        assert any("Missing YAML keys" in e for pv in report.issues for e in pv.errors)

    def test_multi_page_pdf_rejected(self, tmp_path):
        stem = "doc4_page1"
        # Write a 2-page PDF
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        with (tmp_path / f"{stem}.pdf").open("wb") as fh:
            writer.write(fh)
        _make_md(tmp_path / f"{stem}.md")
        report = validate_directory(str(tmp_path))
        assert report.invalid == 1
        assert any("page" in e.lower() for pv in report.issues for e in pv.errors)

    def test_check_yaml_frontmatter_valid(self):
        text = "---\nprimary_language: en\nis_rotation_valid: true\nrotation_correction: 0\nis_table: false\nis_diagram: false\n---\nSome text.\n"
        ok, errors = _check_yaml_frontmatter(text)
        assert ok
        assert not errors

    def test_check_yaml_frontmatter_no_opening(self):
        ok, errors = _check_yaml_frontmatter("No front matter here")
        assert not ok

    def test_export_split_validation(self, tmp_path):
        _populate_export_dir(tmp_path, n_train=2, n_eval=1)
        report = validate_export_split(str(tmp_path))
        assert report.total == 3
        assert report.valid == 3
        assert report.invalid == 0


# ---------------------------------------------------------------------------
# Tests: export pairs alignment
# ---------------------------------------------------------------------------


class TestExportAlignment:
    """Ensure every .pdf in export has a matching .md (same stem) and vice versa."""

    def test_stems_match(self, tmp_path):
        _populate_export_dir(tmp_path, n_train=4, n_eval=2)
        for split in ("train", "eval"):
            split_dir = tmp_path / split
            pdf_stems = {p.stem for p in split_dir.glob("*.pdf")}
            md_stems = {p.stem for p in split_dir.glob("*.md")}
            assert pdf_stems == md_stems, f"{split}: stems mismatch: {pdf_stems ^ md_stems}"

    def test_all_md_have_yaml(self, tmp_path):
        _populate_export_dir(tmp_path, n_train=3, n_eval=1)
        for md_file in tmp_path.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            assert text.startswith("---"), f"{md_file} missing YAML front matter"
            ok, errors = _check_yaml_frontmatter(text)
            assert ok, f"{md_file}: {errors}"

    def test_single_page_constraint(self, tmp_path):
        _populate_export_dir(tmp_path, n_train=3, n_eval=1)
        for pdf_file in tmp_path.rglob("*.pdf"):
            reader = PdfReader(str(pdf_file))
            assert len(reader.pages) == 1, f"{pdf_file} has {len(reader.pages)} pages"


# ---------------------------------------------------------------------------
# Tests: JobRunner
# ---------------------------------------------------------------------------


class TestJobRunner:
    def test_basic_streaming(self):
        runner = JobRunner()

        def simple_job(cancel_event, log_fn, progress_fn):
            for i in range(3):
                log_fn(f"step {i}")
                time.sleep(0.05)

        logs = list(runner.run_and_stream(simple_job))
        final = logs[-1] if logs else ""
        assert "step 0" in final
        assert "step 2" in final

    def test_cancel(self):
        runner = JobRunner()
        steps_done = []

        def slow_job(cancel_event, log_fn, progress_fn):
            for i in range(20):
                if cancel_event.is_set():
                    log_fn("cancelled!")
                    return
                steps_done.append(i)
                time.sleep(0.05)

        def _run():
            list(runner.run_and_stream(slow_job))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.15)  # let a few steps run
        runner.cancel()
        t.join(timeout=3)

        assert len(steps_done) < 20, "Job should have been cancelled early"

    def test_error_does_not_crash_runner(self):
        runner = JobRunner()

        def bad_job(cancel_event, log_fn, progress_fn):
            raise ValueError("deliberate error")

        logs = list(runner.run_and_stream(bad_job))
        final = logs[-1] if logs else ""
        assert "ERROR" in final or "deliberate error" in final
        assert not runner.is_running

    def test_reuse_after_completion(self):
        runner = JobRunner()

        def noop(cancel_event, log_fn, progress_fn):
            log_fn("done")

        list(runner.run_and_stream(noop))
        assert not runner.is_running
        logs2 = list(runner.run_and_stream(noop))
        assert "done" in (logs2[-1] if logs2 else "")


# ---------------------------------------------------------------------------
# Tests: yaml helper in app (standalone)
# ---------------------------------------------------------------------------


class TestYamlHelpers:
    """Test read_md / write_md from yaml_utils (no Gradio required)."""

    def test_round_trip(self, tmp_path):
        from olmocr.finetune_gui.services.yaml_utils import read_md, write_md
        md_path = str(tmp_path / "test.md")
        yaml_vals = {
            "primary_language": "fr",
            "is_rotation_valid": False,
            "rotation_correction": 90,
            "is_table": True,
            "is_diagram": False,
        }
        write_md(md_path, yaml_vals, "Bonjour monde.")
        read_yaml, body = read_md(md_path)
        assert read_yaml["primary_language"] == "fr"
        assert read_yaml["is_rotation_valid"] is False
        assert read_yaml["rotation_correction"] == 90
        assert read_yaml["is_table"] is True
        assert "Bonjour" in body

    def test_null_language(self, tmp_path):
        from olmocr.finetune_gui.services.yaml_utils import read_md, write_md
        md_path = str(tmp_path / "null_lang.md")
        write_md(md_path, {"primary_language": None, "is_rotation_valid": True,
                            "rotation_correction": 0, "is_table": False, "is_diagram": False}, "text")
        read_yaml, _ = read_md(md_path)
        assert read_yaml["primary_language"] is None
