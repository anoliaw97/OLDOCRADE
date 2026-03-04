# olmOCR Fine-Tuning Dataset GUI

A **Gradio** web application for building, reviewing, and exporting
training-ready datasets for fine-tuning olmOCR models — end-to-end
on any Windows or Linux machine.

---

## Quickstart

### 1. Install

```bash
# Inside the olmocr repo root
pip install -e ".[finetune_gui]"
```

Or install Gradio separately:

```bash
pip install gradio>=4.0
```

### 2. Launch

**Linux / macOS**
```bash
python -m olmocr.finetune_gui
# or, if you installed via pip:
olmocr-finetune-gui
```

**Windows** (PowerShell / cmd)
```powershell
python -m olmocr.finetune_gui
```

The GUI opens in your browser at `http://127.0.0.1:7860`.

Optional flags:
```
--host 0.0.0.0    # listen on all interfaces
--port 8080       # change port
--share           # create a public Gradio share URL
```

---

## Workflow (5 tabs)

### Tab 1 · Input

Set the **Run Directory** (one directory per dataset iteration) and the
**VLM model** to use for extraction.  Each run directory holds:

```
<run_dir>/
    workspace/          ← olmOCR JSONL results
    prepared_dataset/   ← single-page .pdf + .md pairs
    export/             ← final train/ eval/ splits
    manifest.json       ← tracks every page's lifecycle
```

Click **Initialise / Resume Session**.  If the run directory already
exists the manifest is loaded and prior work is preserved.

---

### Tab 2 · Extract (Workspace)

Upload PDFs (multiple allowed) and click **Start Extraction**.

Under the hood this runs **local VLM inference** (no vLLM server
required) using the `transformers` library and writes:

```
<run_dir>/workspace/results/output_<hash>.jsonl
```

Each `.jsonl` line is a Dolma document containing:
- full document text
- per-page character boundaries
- per-page metadata (`primary_language`, `is_rotation_valid`, …)

**Safe for large PDFs**:
- Progress is shown page by page.
- Click **Cancel** to stop cleanly after the current page.
- Re-running skips PDFs that already have a result file.

Alternatively, if you already have a workspace produced by the real
`python -m olmocr.pipeline` command, tick **Skip extraction — use
existing workspace** and proceed directly to Tab 3.

---

### Tab 3 · Convert to Training Format

Click **Prepare Dataset**.  This calls
`olmocr.data.prepare_workspace.process_workspace` directly (no
subprocess) to produce:

```
<run_dir>/prepared_dataset/
    <subdir>/
        <doc_id>_page<N>.pdf    ← single-page PDF
        <doc_id>_page<N>.md     ← YAML front matter + extracted text
```

The **Validate** button checks that every `.md` has a matching `.pdf`,
has valid YAML front matter with all required keys, and that each PDF
contains exactly one page.

---

### Tab 4 · Review & Edit

Quality-assurance panel:

| Panel | Contents |
|---|---|
| Left | Scrollable page list with status badges (○ prepared / ✓ reviewed / ⏭ skipped) |
| Middle | PDF page rendered as an image |
| Right | YAML field editors + full Markdown text editor |

**Actions**
- **Save** — writes edits back to the `.md` file and marks the page as `reviewed`.
- **Mark as Skip** — excludes the page from export.
- **Set Split** — manually assign a page to `train` or `eval`.
- **Bulk split strategy** — randomly assign or by position.

Edits are saved immediately to the manifest (`manifest.json`) and to
the `.md` file, so they survive a browser refresh or GUI restart.

---

### Tab 5 · Export

Click **Export** to copy the selected pages into:

```
<export_dir>/
    train/
        <doc_id>_page<N>.pdf
        <doc_id>_page<N>.md
    eval/
        <doc_id>_page<N>.pdf
        <doc_id>_page<N>.md
```

Export modes:
- **all** — export every non-skipped prepared/reviewed page.
- **reviewed_only** — only pages the user explicitly reviewed.

The **Validate Export** button runs the full validation suite:
- Every `.md` starts with `---`.
- All five required YAML keys exist.
- Every `.pdf` is a single-page document.
- Every `.md` has a matching `.pdf` (same stem, same directory).

The **Generate Config** button produces a YAML snippet you can paste
directly into your olmOCR training configuration.

---

## Dataset format (training compatibility)

The exported structure is identical to what
`olmocr.train.dataloader.PdfDataset` expects:

```yaml
dataset:
  train:
    - name: my_finetune_train
      root_dir: /path/to/export/train    # ← point here
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

  eval:
    - name: my_finetune_eval
      root_dir: /path/to/export/eval
      pipeline: *basic_pipeline
```

---

## Running tests

```bash
# From the repo root
pytest tests/finetune_gui/ -v
```

The tests are self-contained (no GPU, no model weights required):
- Manifest save / load round-trip
- Export directory structure validation
- Matched `.pdf`/`.md` pairs and YAML front matter correctness
- Single-page PDF constraint
- JobRunner streaming and cancellation

---

## Architecture

```
olmocr/finetune_gui/
├── __init__.py
├── __main__.py                 # python -m olmocr.finetune_gui
├── app.py                      # Gradio Blocks UI (5 tabs)
└── services/
    ├── job_runner.py           # Thread-based job + log streaming
    ├── manifest.py             # Dataset manifest (JSON)
    ├── local_extractor.py      # VLM inference → workspace JSONL
    ├── workspace_converter.py  # Wraps prepare_workspace directly
    └── dataset_validator.py    # Validates .pdf/.md pairs
```

### Design decisions

| Decision | Reason |
|---|---|
| Local transformers inference | No vLLM server needed for small-scale dataset work |
| Import `prepare_workspace` directly | Avoid code duplication; stay in sync with upstream changes |
| JSON manifest | Survives GUI restarts; enables iterative dataset building |
| Gradio Blocks | Already a Python dep, works on Windows + Linux without Electron |
| Single-page PDF requirement | Enforced by the olmOCR training dataloader |

---

## Adding new PDFs to an existing dataset

Just run a new session pointing to the **same run directory** with
additional PDFs.  The extractor skips files already processed and the
manifest accumulates new pages without wiping old ones.  Re-export
when ready.
