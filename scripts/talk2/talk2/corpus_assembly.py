"""Assemble the talk 2 corpus: one folder of symlinked PDFs that every configuration indexes, plus a manifest."""

import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pymupdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA = REPOSITORY_ROOT / "data"
# TALK2_CORPUS selects the corpus every command works on: "full" (605 PDFs), "pages100" (the selected-page set)
# or "pages_multihop" (pages100 plus the multi-hop questions).
CORPUS_NAME = os.environ.get("TALK2_CORPUS", "full")
EVAL_ROOT = REPOSITORY_ROOT / "eval" / "talk2" / ("" if CORPUS_NAME == "full" else CORPUS_NAME)
CORPUS_ROOT = DATA / ("talk2_corpus" if CORPUS_NAME == "full" else f"talk2_corpus_{CORPUS_NAME}")
CORPUS_PDF = CORPUS_ROOT / "pdf"
MANIFEST = EVAL_ROOT / "corpus_manifest.tsv"
QUESTIONS = EVAL_ROOT / "questions.jsonl"
CACHE_ROOT = REPOSITORY_ROOT / "eval" / "talk2" / "cache"
VIDORE_SETS = ["energy", "finance_fr", "hr", "industrial"]
XL_DOCBENCH_QUESTIONS = ["adubench_cross_000108", "adubench_cross_000018"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_sources() -> list[dict]:
    """Return one record per source PDF: source label, doc_id, title, language, path."""
    records = []
    for pdf in sorted((DATA / "cuad" / "CUAD_v1" / "full_contract_pdf").rglob("*")):
        if pdf.suffix.lower() == ".pdf":
            records.append({"source": "cuad", "doc_id": pdf.stem, "title": pdf.stem, "language": "english", "path": pdf})
    for name in VIDORE_SETS:
        metadata = pd.read_parquet(DATA / "vidore_v3" / name / "documents_metadata.parquet")
        for row in metadata.itertuples():
            pdf = DATA / "vidore_v3" / name / "pdfs" / row.file_name
            records.append({"source": f"vidore-{name}", "doc_id": row.doc_id, "title": row.doc_id,
                            "language": row.doc_language, "path": pdf})
    documents = {json.loads(line)["document_id"]: json.loads(line) for line in open(DATA / "xl-docbench" / "documents.jsonl")}
    questions = [json.loads(line) for line in open(DATA / "xl-docbench" / "qa_cross_doc.jsonl")]
    for question in questions:
        if question["question_id"] not in XL_DOCBENCH_QUESTIONS:
            continue
        for entry in question["documents"]:
            doc_id = entry["document_id"]
            meta = documents[doc_id]
            records.append({"source": "xl-docbench", "doc_id": doc_id, "title": meta["title"] or doc_id,
                            "language": (meta["public_metadata"].get("language") or "").lower(),
                            "path": DATA / "xl-docbench" / "pdfs" / f"{doc_id}.pdf"})
    return records


def assemble() -> list[dict]:
    rows = []
    for record in collect_sources():
        link = CORPUS_PDF / record["source"] / f"{record['doc_id']}.pdf"
        present = record["path"].exists()
        if present:
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(record["path"])
            with pymupdf.open(record["path"]) as document:
                pages = document.page_count
            sha256 = file_sha256(record["path"])
        else:
            pages, sha256 = 0, ""
        rows.append({"path": str(link.relative_to(CORPUS_PDF)), "source": record["source"], "doc_id": record["doc_id"],
                     "title": record["title"], "language": record["language"], "pages": pages,
                     "present": present, "sha256": sha256})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows
