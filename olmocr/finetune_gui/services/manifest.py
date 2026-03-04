"""
Dataset manifest: tracks every page item through the pipeline lifecycle.

Lifecycle states:
    extracted  — JSONL written to workspace/results/
    prepared   — single-page .pdf + .md written by prepare_workspace
    reviewed   — user has opened and saved this page in the review tab
    skipped    — user wants this page excluded from export
    exported   — included in train/ or eval/ output

The manifest is saved as JSON to <run_dir>/manifest.json so work can resume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class PageEntry:
    """Mutable record for one page in the dataset."""

    __slots__ = (
        "key",
        "source_pdf",
        "page_num",
        "doc_id",
        "pdf_path",
        "md_path",
        "status",
        "split",
        "review_patches",
        "added_at",
    )

    def __init__(
        self,
        key: str,
        source_pdf: str,
        page_num: int,
        doc_id: str = "",
        pdf_path: str = "",
        md_path: str = "",
        status: str = "extracted",
        split: str = "train",
        review_patches: Optional[Dict] = None,
        added_at: Optional[float] = None,
    ) -> None:
        self.key = key                                  # unique: "{doc_id}_page{page_num}"
        self.source_pdf = source_pdf
        self.page_num = page_num
        self.doc_id = doc_id
        self.pdf_path = pdf_path                        # absolute or relative to run_dir
        self.md_path = md_path
        self.status = status                            # extracted|prepared|reviewed|skipped|exported
        self.split = split                              # train|eval
        self.review_patches = review_patches or {}      # field → new value
        self.added_at = added_at or time.time()

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: Dict) -> "PageEntry":
        return cls(**{k: d.get(k, cls.__init__.__defaults__[i] if i < len(cls.__init__.__defaults__ or []) else None)
                      for i, k in enumerate(cls.__slots__)})


class DatasetManifest:
    """
    Ordered collection of PageEntry objects.
    Persists to / loads from a JSON file.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._path = self.run_dir / "manifest.json"
        self._entries: Dict[str, PageEntry] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(
                {k: v.to_dict() for k, v in self._entries.items()},
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    @classmethod
    def load(cls, run_dir: Path) -> "DatasetManifest":
        m = cls(run_dir)
        path = Path(run_dir) / "manifest.json"
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
            for k, d in raw.items():
                m._entries[k] = PageEntry(**{s: d.get(s) for s in PageEntry.__slots__})
        return m

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def upsert(self, entry: PageEntry) -> None:
        self._entries[entry.key] = entry

    def get(self, key: str) -> Optional[PageEntry]:
        return self._entries.get(key)

    def all(self) -> List[PageEntry]:
        return list(self._entries.values())

    def by_status(self, *statuses: str) -> List[PageEntry]:
        return [e for e in self._entries.values() if e.status in statuses]

    def by_split(self, split: str) -> List[PageEntry]:
        return [e for e in self._entries.values() if e.split == split and e.status not in ("skipped",)]

    def update_status(self, key: str, status: str) -> None:
        if key in self._entries:
            self._entries[key].status = status

    def set_split(self, key: str, split: str) -> None:
        if key in self._entries:
            self._entries[key].split = split

    def apply_patch(self, key: str, field: str, value) -> None:
        if key in self._entries:
            self._entries[key].review_patches[field] = value

    def apply_split_strategy(self, strategy: str, eval_pct: float = 0.1) -> None:
        """
        Assign train/eval splits.
        strategy='random_pct': randomly assign eval_pct fraction to eval.
        strategy='first_N':    first int(eval_pct) entries to eval.
        """
        import random
        prepared = [e for e in self._entries.values() if e.status in ("prepared", "reviewed", "exported")]
        if strategy == "random_pct":
            eval_set = set(random.sample([e.key for e in prepared], k=max(1, int(len(prepared) * eval_pct))))
            for e in prepared:
                e.split = "eval" if e.key in eval_set else "train"
        elif strategy == "first_N":
            n = max(1, int(eval_pct))
            for i, e in enumerate(prepared):
                e.split = "eval" if i < n else "train"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(e.status for e in self._entries.values()))

    def split_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(
            e.split for e in self._entries.values()
            if e.status not in ("skipped",)
        ))

    def __len__(self) -> int:
        return len(self._entries)
