"""Build the `pages100` corpus: for each frozen question, its evidence pages plus hard negatives mined from the full
corpus by a fixed rule, packaged as one subset PDF per source document (original file names kept).

Rule (fixed before any configuration is run):
1. Evidence pages, from the benchmark annotations, located by the rules in EVIDENCE_RULES.
2. Four embedding negatives: the highest-ranked non-evidence pages for the question in the full-corpus RAG index
   (Qwen3-Embedding-0.6B over 512-token chunks; each chunk mapped back to its page).
3. Four keyword negatives: the highest BM25-ranked non-evidence pages for the question text over all full-corpus pages.
Questions are processed in order 1 → 10; a page selected for one question is not counted again for a later one.
"""

import csv
import json
import os
import re
from pathlib import Path

import bm25s
import pymupdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FULL_TEXT = REPOSITORY_ROOT / "data" / "talk2_corpus" / "text"
FULL_PDF = REPOSITORY_ROOT / "data" / "talk2_corpus" / "pdf"
FULL_QUESTIONS = REPOSITORY_ROOT / "eval" / "talk2" / "questions.jsonl"
OUTPUT_EVAL = REPOSITORY_ROOT / "eval" / "talk2" / "pages100"
OUTPUT_PDF = REPOSITORY_ROOT / "data" / "talk2_corpus_pages100" / "pdf"
NEGATIVES_PER_METHOD = 4
ISRAEL_LAW = re.compile(r"laws?\s+of\s+(the\s+)?(state\s+of\s+)?israel", re.IGNORECASE)


def load_pages() -> dict[str, list[str]]:
    """Full-corpus PyMuPDF text per document path (relative, without suffix) as a list of pages."""
    return {str(p.relative_to(FULL_TEXT).with_suffix("")): p.read_text().split("\f")
            for p in sorted(FULL_TEXT.rglob("*.txt"))}


def pages_containing(pages: dict, document: str, pattern) -> list[tuple[str, int]]:
    found = []
    for index, text in enumerate(pages[document]):
        flat = " ".join(text.split())
        hit = pattern.search(flat) if hasattr(pattern, "search") else pattern in flat
        if hit:
            found.append((document, index))
    return found


def evidence_pages(question: dict, pages: dict) -> list[tuple[str, int]]:
    qid = question["id"]
    if qid == "adubench_cross_000108":
        return (pages_containing(pages, "xl-docbench/doc_000116", "Supp.1")
                + pages_containing(pages, "xl-docbench/doc_000254", "Supp.1"))
    if qid == "cuad:israel-governing-law":
        result = []
        for document in question["evidence_docs"]:
            hits = pages_containing(pages, document, ISRAEL_LAW)
            if not hits:
                raise ValueError(f"no governing-law page found in {document}")
            result += hits
        return result
    if qid == "cuad:ultragenyx-cremer-noncompete":
        document = question["evidence_docs"][0]
        return [(document, index) for index in range(len(pages[document]))]
    if qid == "cuad:adaptimmune-mdanderson-19644000":
        return pages_containing(pages, question["evidence_docs"][0], "19,644,000")
    if qid == "adubench_cross_000018":
        return [("xl-docbench/doc_000177", 1), ("xl-docbench/doc_000328", 1), ("xl-docbench/doc_000206", 2)]
    by_stem = {Path(d).name: d for d in question["evidence_docs"]}
    return [(by_stem[stem], index) for stem, indexes in question["evidence_page_indexes_0_based"].items()
            for index in indexes]


def chunk_page(pages: dict, document: str, chunk_text: str) -> int | None:
    probe = " ".join(chunk_text.split())[:80]
    for index, text in enumerate(pages[document]):
        if probe and probe in " ".join(text.split()):
            return index
    probe = probe[:30]
    for index, text in enumerate(pages[document]):
        if probe and probe in " ".join(text.split()):
            return index
    return None


def select() -> list[dict]:
    os.environ.setdefault("TALK2_CORPUS", "full")
    from talk2.retrieval_index import load_retriever

    questions = [json.loads(line) for line in FULL_QUESTIONS.open()]
    pages = load_pages()
    page_keys = [(d, i) for d, texts in pages.items() for i in range(len(texts))]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([pages[d][i] for d, i in page_keys], stopwords="en", show_progress=False),
                    show_progress=False)
    embedding_retriever = load_retriever("text")
    embedding_retriever.similarity_top_k = 80

    selected: dict[tuple[str, int], dict] = {}
    evidence = {q["id"]: evidence_pages(q, pages) for q in questions}
    all_evidence = {page for found in evidence.values() for page in found}
    for question in sorted(questions, key=lambda q: q["order"]):
        for rank, page in enumerate(evidence[question["id"]], 1):
            selected.setdefault(page, {"question": question["id"], "role": "evidence", "rank": rank})
        added = 0
        for rank, node in enumerate(embedding_retriever.retrieve(question["question"]), 1):
            document = node.node.metadata["file_path"]
            index = chunk_page(pages, document, node.node.get_content())
            if index is None or (document, index) in all_evidence or (document, index) in selected:
                continue
            selected[(document, index)] = {"question": question["id"], "role": "embedding_negative", "rank": rank}
            added += 1
            if added == NEGATIVES_PER_METHOD:
                break
        results, _ = retriever.retrieve(bm25s.tokenize([question["question"]], stopwords="en", show_progress=False),
                                        k=200, show_progress=False)
        added = 0
        for rank, position in enumerate(results[0], 1):
            page = page_keys[int(position)]
            if page in all_evidence or page in selected:
                continue
            selected[page] = {"question": question["id"], "role": "keyword_negative", "rank": rank}
            added += 1
            if added == NEGATIVES_PER_METHOD:
                break
    return [{"document": d, "page_index": i, **info} for (d, i), info in selected.items()]


def package(selection: list[dict]) -> None:
    """Write subset PDFs, the selection table, the corpus manifest and the pages100 questions."""
    import hashlib

    OUTPUT_EVAL.mkdir(parents=True, exist_ok=True)
    by_document: dict[str, list[int]] = {}
    for row in selection:
        by_document.setdefault(row["document"], []).append(row["page_index"])
    full_manifest = {r["path"].removesuffix(".pdf"): r for r in
                     csv.DictReader((REPOSITORY_ROOT / "eval/talk2/corpus_manifest.tsv").open(), delimiter="\t")}
    subset_index: dict[tuple[str, int], int] = {}
    manifest_rows = []
    for document, indexes in sorted(by_document.items()):
        indexes = sorted(set(indexes))
        target = OUTPUT_PDF / f"{document}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(FULL_PDF / f"{document}.pdf") as source, pymupdf.open() as subset:
            for position, index in enumerate(indexes):
                subset.insert_pdf(source, from_page=index, to_page=index)
                subset_index[(document, index)] = position
            subset.save(target, garbage=3, deflate=True)
        meta = full_manifest[document]
        manifest_rows.append({"path": f"{document}.pdf", "source": meta["source"], "doc_id": meta["doc_id"],
                              "title": meta["title"], "language": meta["language"], "pages": len(indexes),
                              "present": True, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "original_page_indexes": " ".join(map(str, indexes))})
    with (OUTPUT_EVAL / "corpus_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (OUTPUT_EVAL / "page_selection.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "role", "rank", "document", "page_index",
                                                    "subset_page_index"], delimiter="\t")
        writer.writeheader()
        for row in selection:
            writer.writerow({**{k: row[k] for k in ("question", "role", "rank", "document", "page_index")},
                             "subset_page_index": subset_index[(row["document"], row["page_index"])]})
    with (OUTPUT_EVAL / "questions.jsonl").open("w") as out:
        for line in FULL_QUESTIONS.open():
            question = json.loads(line)
            if question["id"] == "cuad:israel-governing-law":
                question["question"] = "Which of these contracts are governed by the law of the State of Israel?"
            question["evidence_subset_pages"] = [
                {"document": r["document"], "subset_page_index": subset_index[(r["document"], r["page_index"])]}
                for r in selection if r["question"] == question["id"] and r["role"] == "evidence"]
            out.write(json.dumps(question, ensure_ascii=False) + "\n")
