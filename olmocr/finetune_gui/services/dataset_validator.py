"""
Dataset validation utilities.

Checks that every .md/.pdf pair meets olmOCR training requirements:
  1. .md begins with YAML front-matter delimiters (---)
  2. All required YAML keys are present
  3. Matching .pdf exists in the same directory
  4. PDF contains exactly one page
  5. pypdf can load the PDF without error
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

REQUIRED_YAML_KEYS = {
    "primary_language",
    "is_rotation_valid",
    "rotation_correction",
    "is_table",
    "is_diagram",
}

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL | re.MULTILINE)


@dataclass
class PageValidation:
    key: str
    md_path: str
    pdf_path: str
    ok: bool = True
    errors: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.errors.append(reason)


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    issues: List[PageValidation] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"Pages checked : {self.total}",
            f"Valid         : {self.valid}",
            f"Invalid       : {self.invalid}",
        ]
        if self.issues:
            lines.append("\nFailed pages:")
            for pv in self.issues:
                lines.append(f"  {pv.key}: {'; '.join(pv.errors)}")
        return "\n".join(lines)


def _check_yaml_frontmatter(md_text: str) -> Tuple[bool, List[str]]:
    """Return (ok, list_of_errors) for the front-matter block."""
    errors: List[str] = []
    if not md_text.startswith("---"):
        errors.append("Does not start with '---'")
        return False, errors

    m = _FM_RE.match(md_text)
    if not m:
        errors.append("Front matter block not closed with '---'")
        return False, errors

    fm_text = m.group(1)
    found_keys = set()
    for line in fm_text.splitlines():
        if ":" in line:
            found_keys.add(line.split(":")[0].strip())

    missing = REQUIRED_YAML_KEYS - found_keys
    if missing:
        errors.append(f"Missing YAML keys: {sorted(missing)}")

    return len(errors) == 0, errors


def _check_single_page_pdf(pdf_path: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, error_or_None)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        n = len(reader.pages)
        if n != 1:
            return False, f"Expected 1 page, found {n}"
        return True, None
    except Exception as exc:
        return False, f"Failed to open PDF: {exc}"


def validate_directory(directory: str) -> ValidationReport:
    """
    Recursively scan *directory* for .md files and validate each pair.
    """
    report = ValidationReport()
    root = Path(directory)

    for md_file in sorted(root.rglob("*.md")):
        pdf_file = md_file.with_suffix(".pdf")
        key = md_file.stem
        pv = PageValidation(key=key, md_path=str(md_file), pdf_path=str(pdf_file))
        report.total += 1

        # 1) matching PDF exists?
        if not pdf_file.exists():
            pv.fail(f"Missing PDF: {pdf_file.name}")
            report.invalid += 1
            report.issues.append(pv)
            continue

        # 2) YAML front matter
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as exc:
            pv.fail(f"Cannot read .md: {exc}")
            report.invalid += 1
            report.issues.append(pv)
            continue

        fm_ok, fm_errors = _check_yaml_frontmatter(text)
        for e in fm_errors:
            pv.fail(e)

        # 3) single-page PDF
        pdf_ok, pdf_error = _check_single_page_pdf(str(pdf_file))
        if not pdf_ok:
            pv.fail(pdf_error)

        if pv.ok:
            report.valid += 1
        else:
            report.invalid += 1
            report.issues.append(pv)

    return report


def validate_export_split(export_dir: str) -> ValidationReport:
    """Validate both train/ and eval/ subdirs of an export directory."""
    root = Path(export_dir)
    combined = ValidationReport()
    for split in ("train", "eval"):
        split_dir = root / split
        if split_dir.exists():
            sub = validate_directory(str(split_dir))
            combined.total += sub.total
            combined.valid += sub.valid
            combined.invalid += sub.invalid
            combined.issues.extend(sub.issues)
    return combined
