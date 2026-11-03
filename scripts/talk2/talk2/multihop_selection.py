"""Select the multi-hop questions and build the `pages_multihop` corpus: `pages100` plus, per added question, its
evidence pages and hard negatives from the same document, packaged as one subset PDF per document.

Question rule (fixed before any run):
1. Source: XL-DocBench single-document questions with reasoning type `reference_chain`, answerable, human-annotated
   evidence on 1 to 3 distinct pages, an answer of at most 8 words, an English document, the PDF on disk, and a document
   not already in `pages100`. One question per document.
2. Order: shuffled with seed 42 within each domain, then taken round-robin across domains in sorted order.
3. Each question in that order is checked against its PDF; REJECTED records every question passed over and why. The
   first TARGET_COUNT accepted questions are added.

Negatives: the 4 highest-ranked non-evidence pages by embedding and the 4 by BM25, for the question, within the
question's own document. The hop has to be found among pages of one document, so those are the pages that mislead.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
XL_DOCBENCH = REPOSITORY_ROOT / "data" / "xl-docbench"
PAGES100_EVAL = REPOSITORY_ROOT / "eval" / "talk2" / "pages100"
TARGET_COUNT = 10
SELECTION_SEED = 42
MAXIMUM_ANSWER_WORDS = 8
MAXIMUM_EVIDENCE_PAGES = 3


def evidence_page_numbers(record: dict) -> list[int]:
    return sorted({page for item in record["document"]["evidence_items"] for page in item["pages"]})


def eligible_questions() -> list[dict]:
    documents = {json.loads(line)["document_id"]: json.loads(line) for line in (XL_DOCBENCH / "documents.jsonl").open()}
    existing = {row.split("\t")[2] for row in (PAGES100_EVAL / "corpus_manifest.tsv").open()}
    result = []
    for line in (XL_DOCBENCH / "qa_single_doc.jsonl").open():
        record = json.loads(line)
        metadata = record["metadata"]
        document_id = record["document"]["document_id"]
        language = (documents[document_id]["public_metadata"].get("language") or "").lower()
        if (metadata["reasoning_type"] == "reference_chain" and not metadata["is_unanswerable"]
                and 1 <= len(evidence_page_numbers(record)) <= MAXIMUM_EVIDENCE_PAGES
                and len(str(record["answer"]["value"]).split()) <= MAXIMUM_ANSWER_WORDS
                and language.startswith("en") and document_id not in existing
                and (XL_DOCBENCH / "pdfs" / f"{document_id}.pdf").exists()):
            record["title"] = documents[document_id]["title"]
            result.append(record)
    return result


def selection_order() -> list[dict]:
    by_domain = defaultdict(list)
    for record in sorted(eligible_questions(), key=lambda r: r["question_id"]):
        by_domain[record["metadata"]["domain"]].append(record)
    generator = random.Random(SELECTION_SEED)
    for domain in sorted(by_domain):
        generator.shuffle(by_domain[domain])
    ordered, seen_documents = [], set()
    while any(by_domain.values()):
        for domain in sorted(by_domain):
            while by_domain[domain]:
                record = by_domain[domain].pop(0)
                if record["document"]["document_id"] not in seen_documents:
                    seen_documents.add(record["document"]["document_id"])
                    ordered.append(record)
                    break
    return ordered


# Gold check against the PDFs, 1 Nov 2026, before any run on these questions.
ALIASES = "alias_set"
ALL_TERMS = "all_terms"
ACCEPTED = {
    "adubench_single_000697": {"rule": ALIASES, "aliases": ["Financial Promotions and Enforcement Task Force", "FPET"]},
    "adubench_single_000310": {"rule": ALIASES, "aliases": ["twin 3-ft", "twin 3 ft", "twin 3-foot", "twin 3 foot"],
                               "added_evidence_pages": [201]},
    "adubench_single_000875": {"rule": ALIASES, "aliases": ["Morris R. Morgan", "Morris Morgan"]},
    "adubench_single_000329": {"rule": ALIASES, "aliases": ["Latin school drama", "Latin school"]},
    "adubench_single_001200": {"rule": ALIASES, "aliases": ["horny ring", "striated ring"]},
    "adubench_single_000283": {"rule": ALL_TERMS, "terms": [["equation 2", "equations 2", "eq. 2"],
                                                            ["equation 3", "and 3", "eq. 3"], ["chart 8"]]},
    "adubench_single_001107": {"rule": ALIASES, "aliases": ["virtual asset service provider"]},
    "adubench_single_000826": {"rule": ALIASES, "aliases": ["Incidents directly involving biological agents"]},
    "adubench_single_000689": {"rule": ALL_TERMS, "terms": [["further reading"], ["305"]]},
    "adubench_single_000551": {"rule": ALIASES, "aliases": ["economic and geopolitical uncertainty",
                                                            "geopolitical uncertainty"]},
}
REJECTED = {
    "adubench_single_000226": "Gold named in the question (PPSD).",
    "adubench_single_000495": "One fact on one page; no chain.",
    "adubench_single_000284": "Gold not unique: other works are also in Minor Works.",
    "adubench_single_000986": "Gold not determined: nothing links the Editor-in-Chief role to the Executive summary.",
    "adubench_single_001234": "Gold not unique: all the illustrative families appear in both lists.",
    "adubench_single_000132": "One fact on one page; no chain.",
    "adubench_single_000580": "Gold not determined: the match to the competitiveness objective is a judgement.",
    "adubench_single_000942": "Gold guessable without the document (agriculture).",
    "adubench_single_000097": "One fact on one page; the question names the entry to look up.",
}
TITLE_OVERRIDES = {"doc_000219": "The FATF Recommendations", "doc_000067": "Laboratory biosecurity guidance"}


def select_and_package() -> dict:
    """Write eval/talk2/pages_multihop/ and data/talk2_corpus_pages_multihop/pdf/. Requires Ollama for embeddings."""
    import csv
    import hashlib
    import os
    import re

    import bm25s
    import numpy as np
    import pymupdf

    if os.environ.get("TALK2_CORPUS") != "pages_multihop":
        raise ValueError("run with TALK2_CORPUS=pages_multihop")
    from llama_index.core import Settings

    from talk2.corpus_assembly import CORPUS_PDF, EVAL_ROOT
    from talk2.retrieval_index import configure_embedding

    configure_embedding()
    ordered = selection_order()
    positions = {r["question_id"]: i for i, r in enumerate(ordered)}
    last = max(positions[q] for q in ACCEPTED)
    passed_over = {r["question_id"] for r in ordered[:last + 1]} - set(ACCEPTED)
    if passed_over != set(REJECTED):
        raise ValueError(f"rejections do not match the selection order: {passed_over ^ set(REJECTED)}")

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    pages100_pdf = REPOSITORY_ROOT / "data" / "talk2_corpus_pages100" / "pdf"
    for source in pages100_pdf.rglob("*.pdf"):
        target = CORPUS_PDF / source.relative_to(pages100_pdf)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
    manifest = list(csv.DictReader((PAGES100_EVAL / "corpus_manifest.tsv").open(), delimiter="\t"))
    selection = list(csv.DictReader((PAGES100_EVAL / "page_selection.tsv").open(), delimiter="\t"))
    questions = [{**json.loads(line), "set": "rag_misses"} for line in (PAGES100_EVAL / "questions.jsonl").open()]

    for order, record in enumerate(sorted((r for r in ordered if r["question_id"] in ACCEPTED),
                                          key=lambda r: positions[r["question_id"]]), len(questions) + 1):
        question_id, document_id = record["question_id"], record["document"]["document_id"]
        decision = ACCEPTED[question_id]
        document_path = f"xl-docbench/{document_id}"
        evidence = sorted(set(evidence_page_numbers(record)) | set(decision.get("added_evidence_pages", [])))
        evidence_indexes = [page - 1 for page in evidence]    # XL-DocBench pages are 1-based
        with pymupdf.open(XL_DOCBENCH / "pdfs" / f"{document_id}.pdf") as source:
            texts = [page.get_text() for page in source]
            candidates = [i for i, text in enumerate(texts) if text.strip() and i not in evidence_indexes]
            keyword_index = bm25s.BM25()
            keyword_index.index(bm25s.tokenize([texts[i] for i in candidates], stopwords="en", show_progress=False),
                                show_progress=False)
            ranked, _ = keyword_index.retrieve(bm25s.tokenize([record["question"]], stopwords="en", show_progress=False),
                                               k=min(40, len(candidates)), show_progress=False)
            keyword_ranking = [candidates[int(p)] for p in ranked[0]]
            vectors = np.array(Settings.embed_model.get_text_embedding_batch([texts[i][:4000] for i in candidates]))
            query = np.array(Settings.embed_model.get_query_embedding(record["question"]))
            scores = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
            embedding_ranking = [candidates[int(p)] for p in np.argsort(-scores)[:40]]

            chosen: dict[int, tuple[str, int]] = {i: ("evidence", rank) for rank, i in enumerate(evidence_indexes, 1)}
            for role, ranking in (("embedding_negative", embedding_ranking), ("keyword_negative", keyword_ranking)):
                added = 0
                for rank, index in enumerate(ranking, 1):
                    if index not in chosen:
                        chosen[index] = (role, rank)
                        added += 1
                    if added == 4:
                        break
            indexes = sorted(chosen)
            target = CORPUS_PDF / f"{document_path}.pdf"
            target.parent.mkdir(parents=True, exist_ok=True)
            with pymupdf.open() as subset:
                for index in indexes:
                    subset.insert_pdf(source, from_page=index, to_page=index)
                subset.save(target, garbage=3, deflate=True)
        title = TITLE_OVERRIDES.get(document_id) or record["title"]
        manifest.append({"path": f"{document_path}.pdf", "source": "xl-docbench", "doc_id": document_id, "title": title,
                         "language": "english", "pages": len(indexes), "present": True,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "original_page_indexes": " ".join(map(str, indexes))})
        subset_of = {index: position for position, index in enumerate(indexes)}
        selection += [{"question": question_id, "role": role, "rank": rank, "document": document_path,
                       "page_index": index, "subset_page_index": subset_of[index]}
                      for index, (role, rank) in chosen.items()]
        match = {k: v for k, v in decision.items() if k != "added_evidence_pages"}
        questions.append({
            "order": order, "id": question_id, "set": "multi_hop", "source": "xl-docbench",
            "question": f'In the document "{title}": {record["question"]}', "gold": str(record["answer"]["value"]),
            "match": match, "evidence_docs": [document_path], "evidence_page_indexes_0_based": {document_id: evidence_indexes},
            "difficulty": record["metadata"]["difficulty"], "domain": record["metadata"]["domain"],
            "verified_quote": " | ".join(re.sub(r"\s+", " ", item["quote"])[:200] for item in record["document"]["evidence_items"]),
            "verified_by": "Quotes located in the PDF text, 1 Nov 2026",
            "evidence_subset_pages": [{"document": document_path, "subset_page_index": subset_of[i]} for i in evidence_indexes],
        })

    with (EVAL_ROOT / "corpus_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    with (EVAL_ROOT / "page_selection.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selection)
    with (EVAL_ROOT / "questions.jsonl").open("w") as out:
        for question in questions:
            out.write(json.dumps(question, ensure_ascii=False) + "\n")
    with (EVAL_ROOT / "question_selection.tsv").open("w") as out:
        out.write("position\tquestion_id\tdecision\treason\n")
        for record in ordered[:last + 1]:
            question_id = record["question_id"]
            out.write(f"{positions[question_id]}\t{question_id}\t{'accepted' if question_id in ACCEPTED else 'rejected'}"
                      f"\t{REJECTED.get(question_id, '')}\n")
    return {"questions": len(questions), "documents": len(manifest), "pages": sum(int(r["pages"]) for r in manifest)}
