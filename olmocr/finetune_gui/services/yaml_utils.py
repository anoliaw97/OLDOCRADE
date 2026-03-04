"""
YAML front-matter read/write helpers shared between app.py and tests.
No Gradio dependency.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple


def read_md(md_path: str) -> Tuple[Dict, str]:
    """Parse an olmOCR .md file into (yaml_dict, body_text)."""
    defaults: Dict = {
        "primary_language": "en",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
    }
    try:
        text = Path(md_path).read_text(encoding="utf-8")
    except Exception:
        return defaults, ""

    m = re.search(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL | re.MULTILINE)
    if not m:
        return defaults, text

    fm_text, body = m.group(1), m.group(2)
    yaml_vals = dict(defaults)
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "primary_language":
            yaml_vals[k] = None if v.lower() in ("null", "none", "") else v
        elif k in ("is_rotation_valid", "is_table", "is_diagram"):
            yaml_vals[k] = v.lower() in ("true", "yes", "1")
        elif k == "rotation_correction":
            try:
                yaml_vals[k] = int(v)
            except ValueError:
                pass
    return yaml_vals, body.strip()


def write_md(md_path: str, yaml_vals: Dict, body: str) -> None:
    """Write YAML front matter + body to *md_path*."""
    lang = yaml_vals.get("primary_language")
    lang_str = "null" if lang is None else str(lang)
    lines = [
        "---",
        f"primary_language: {lang_str}",
        f"is_rotation_valid: {str(yaml_vals.get('is_rotation_valid', True)).lower()}",
        f"rotation_correction: {yaml_vals.get('rotation_correction', 0)}",
        f"is_table: {str(yaml_vals.get('is_table', False)).lower()}",
        f"is_diagram: {str(yaml_vals.get('is_diagram', False)).lower()}",
        "---",
        body,
    ]
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def apply_patches_to_md(md_path: str, patches: Dict) -> None:
    """Apply review patches on top of the current .md content."""
    if not patches:
        return
    yaml_vals, body = read_md(md_path)
    for k, v in patches.items():
        if k == "body":
            body = v
        else:
            yaml_vals[k] = v
    write_md(md_path, yaml_vals, body)
