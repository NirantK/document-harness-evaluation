"""Baseline text extraction: PyMuPDF page.get_text(), no OCR, one text file per PDF with form-feed page breaks."""

import csv
from pathlib import Path

import pymupdf

from talk2.corpus_assembly import CORPUS_PDF, MANIFEST

CORPUS_TEXT = CORPUS_PDF.parent / "text"


def extract_all() -> tuple[int, int]:
    documents = pages = 0
    with MANIFEST.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["present"] != "True":
                continue
            target = CORPUS_TEXT / Path(row["path"]).with_suffix(".txt")
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with pymupdf.open(CORPUS_PDF / row["path"]) as document:
                target.write_text("\f".join(page.get_text() for page in document))
                pages += document.page_count
            documents += 1
    return documents, pages
