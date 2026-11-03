"""Build the demonstration of questions that need OCR: the unseen corpus plus six pages of MIL-STD-810H, one of them an
image with no text layer, and one page of a technical order whose answer is a label in a line drawing. One question comes
from XL-DocBench and one from ViDoRe v3. Neither is part of a scored question set.

Usage: TALK2_CORPUS=scanned_demonstration uv run python -m talk2.scanned_demonstration
"""

import csv
import json
import shutil
from pathlib import Path

import pymupdf

from talk2.corpus_assembly import CORPUS_ROOT, EVAL_ROOT, REPOSITORY_ROOT, file_sha256
from talk2.multihop_selection import XL_DOCBENCH

BASE = "pages_unseen"
QUESTION_ID = "adubench_single_000733"
DOCUMENT = "doc_000023"
FIRST_PAGE_INDEX, LAST_PAGE_INDEX = 35, 40
TARGET = "us-dod/mil_std_810h_method_516_8"
TITLE = "MIL-STD-810H Method 516.8 Shock, Transit Drop"
DRAWING_SOURCE = "vidore_v3/industrial/pdfs/TO_00-25-172CL-3.pdf"
DRAWING_PAGE_INDEX = 65
DRAWING_TARGET = "us-dod/to_00_25_172cl_3_figure_7_4"
DRAWING_TITLE = "TO 00-25-172CL-3, Figure 7-4, Aircraft 25MM Gun Safety Pin"
DRAWING_QUESTION = "Where is the 25mm gun safety pin stored when not in use per the Summary of USAF Technical Manual Sections?"


def build() -> None:
    base_root = REPOSITORY_ROOT / "data" / f"talk2_corpus_{BASE}"
    for folder in ("pdf", "text", "markdown"):
        shutil.rmtree(CORPUS_ROOT / folder, ignore_errors=True)
        shutil.copytree(base_root / folder, CORPUS_ROOT / folder, dirs_exist_ok=True, symlinks=folder == "pdf")
    target = CORPUS_ROOT / "pdf" / f"{TARGET}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(XL_DOCBENCH / "pdfs" / f"{DOCUMENT}.pdf") as source:
        extract = pymupdf.open()
        extract.insert_pdf(source, from_page=FIRST_PAGE_INDEX, to_page=LAST_PAGE_INDEX)
        extract.save(target)
        extract.close()
    drawing = CORPUS_ROOT / "pdf" / f"{DRAWING_TARGET}.pdf"
    with pymupdf.open(REPOSITORY_ROOT / "data" / DRAWING_SOURCE) as source:
        extract = pymupdf.open()
        extract.insert_pdf(source, from_page=DRAWING_PAGE_INDEX, to_page=DRAWING_PAGE_INDEX)
        extract.save(drawing)
        extract.close()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    base_manifest = REPOSITORY_ROOT / "eval" / "talk2" / BASE / "corpus_manifest.tsv"
    rows = list(csv.DictReader(base_manifest.open(), delimiter="\t"))
    rows.append({"path": f"{TARGET}.pdf", "source": "xl-docbench", "doc_id": DOCUMENT, "title": TITLE, "language": "english",
                 "pages": str(LAST_PAGE_INDEX - FIRST_PAGE_INDEX + 1), "present": "True", "sha256": file_sha256(target),
                 "original_page_indexes": " ".join(map(str, range(FIRST_PAGE_INDEX, LAST_PAGE_INDEX + 1)))})
    rows.append({"path": f"{DRAWING_TARGET}.pdf", "source": "vidore-industrial", "doc_id": "TO_00-25-172CL-3_figure_7_4",
                 "title": DRAWING_TITLE, "language": "english", "pages": "1", "present": "True",
                 "sha256": file_sha256(drawing), "original_page_indexes": str(DRAWING_PAGE_INDEX)})
    with (EVAL_ROOT / "corpus_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    record = next(r for r in map(json.loads, (XL_DOCBENCH / "qa_single_doc.jsonl").open()) if r["question_id"] == QUESTION_ID)
    question = {"order": 1, "id": QUESTION_ID, "source": "xl-docbench", "set": "demonstration", "question": record["question"],
                "gold": "1.5 m", "match": {"rule": "numeric", "value": 1.5, "tolerance": 0},
                "evidence_docs": [TARGET], "evidence_page_indexes_0_based": {DOCUMENT: [36, 39]}}
    second = {"order": 2, "id": "industrial:239", "source": "vidore-industrial", "set": "demonstration",
              "question": DRAWING_QUESTION, "gold": "Safety Pin Stowage Bracket",
              "match": {"rule": "alias_set", "aliases": ["stowage bracket"]}, "evidence_docs": [DRAWING_TARGET],
              "evidence_page_indexes_0_based": {"TO_00-25-172CL-3": [DRAWING_PAGE_INDEX]}}
    (EVAL_ROOT / "questions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                                       for item in (question, second)))


if __name__ == "__main__":
    build()
    print(f"built {CORPUS_ROOT}")
