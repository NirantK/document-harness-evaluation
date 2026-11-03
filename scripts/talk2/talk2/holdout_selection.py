"""Build the held-out corpora. `pages_holdout` is `pages_multihop` plus ten questions; `pages_holdout2` is `pages_holdout`
plus fifteen more, fixed after revision 2 of the structured agent was written and before any run on them.

The questions continue the frozen multi-hop selection order from position 19 under the same rejection rules. They were
fixed on 2 Nov 2026 before any run on them, and after the structured agent was first written against the `pages_multihop`
traces. Only `questions.jsonl` here is scored, so a score on this corpus was not available during tuning.
"""

import csv
import hashlib
import json
import os
import re

from talk2.multihop_selection import (ALIASES, ALL_TERMS, REPOSITORY_ROOT, XL_DOCBENCH, evidence_page_numbers,
                                      selection_order)

FIRST_POSITION = 19
ACCEPTED = {
    "adubench_single_001005": {"rule": ALIASES, "aliases": ["HER2 overexpression"]},
    "adubench_single_000040": {"rule": ALIASES, "aliases": ["Garrity"]},
    "adubench_single_001177": {"rule": ALIASES, "aliases": ["International Banking Supervision"]},
    "adubench_single_000173": {"rule": ALIASES, "aliases": ["in-structure response spectra",
                                                            "in-structure response spectrum"]},
    "adubench_single_000106": {"rule": "numeric", "value": 2, "tolerance": 0},
    "adubench_single_000720": {"rule": ALIASES, "aliases": ["evapotranspiration"]},
    "adubench_single_001264": {"rule": ALL_TERMS, "terms": [["figure i-15", "fig. i-15"], ["table 1"]]},
    "adubench_single_000780": {"rule": ALIASES, "aliases": ["cardiology"]},
    "adubench_single_000447": {"rule": ALIASES, "aliases": ["tangent galvanometer"]},
    "adubench_single_000855": {"rule": ALIASES, "aliases": ["state, federal, and local requirements",
                                                            "state, federal and local requirements"]},
}
REJECTED = {
    "adubench_single_000983": "Gold not unique: the clause gives 60, 65 and 75 percent by date.",
    "adubench_single_000265": "Gold not determined: 'most directly contradicted' is a judgement.",
    "adubench_single_000402": "Gold named in the question.",
    "adubench_single_000816": "Gold not determined: 'most directly matches' is a judgement.",
    "adubench_single_000544": "Gold not determined: 'best addresses' is a judgement.",
    "adubench_single_001317": "One fact on one page; no chain.",
    "adubench_single_000204": "Gold (98%) does not answer the question asked.",
    "adubench_single_000266": "One fact on one page; no chain.",
    "adubench_single_000050": "One fact on one page; no chain.",
    "adubench_single_001156": "Gold named in the question.",
    "adubench_single_001210": "Gold named in the question.",
    "adubench_single_000255": "Gold not determined: the question's supersession claim is not stated in the document.",
    "adubench_single_001202": "One fact on one page; no chain.",
    "adubench_single_000076": "One fact on one page; no chain.",
    "adubench_single_001265": "Gold not unique: primary peritoneal cancer appears in both sections.",
    "adubench_single_000706": "Gold named in the question.",
    "adubench_single_000179": "One fact on one page; no chain.",
    "adubench_single_000903": "One fact on one page; no chain.",
}
TITLE_OVERRIDES = {"doc_000024": "Bergey's Manual of Systematic Bacteriology",
                   "doc_000017": "Professional Paper 1770: Groundwater Availability of the Denver Basin Aquifer System",
                   "doc_000297": "Hydraulic Design of Highway Culverts (FHWA-NHI-01-020)",
                   "doc_000218": "Physics Part 1, Class 12"}


SECOND_ACCEPTED = {
    "adubench_single_001140": {"rule": ALIASES, "aliases": ["International cooperation"]},
    "adubench_single_001217": {"rule": ALIASES, "aliases": ["whether these risks are acceptable", "whether the risks are acceptable"]},
    "adubench_single_001274": {"rule": ALIASES, "aliases": ["geophysical log"]},
    "adubench_single_000503": {"rule": ALIASES, "aliases": ["N-1A", "N–1A"]},
    "adubench_single_000646": {"rule": ALL_TERMS, "terms": [["section 5"], ["table 6"]]},
    "adubench_single_000391": {"rule": ALIASES, "aliases": ["11.7.12"]},
    "adubench_single_001029": {"rule": "numeric", "value": 33, "tolerance": 0},
    "adubench_single_000385": {"rule": ALIASES, "aliases": ["principal parties"]},
    "adubench_single_000662": {"rule": ALIASES, "aliases": ["containment"]},
    "adubench_single_000036": {"rule": ALIASES, "aliases": ["articles 29", "article 29"]},
    "adubench_single_000511": {"rule": ALIASES, "aliases": ["Metastatic Synchronous"]},
    "adubench_single_001190": {"rule": ALIASES, "aliases": ["10^-6", "10-6", "10⁻⁶", "10−6"]},
    "adubench_single_000838": {"rule": ALIASES, "aliases": ["4.2.2.3"]},
    "adubench_single_000271": {"rule": ALIASES, "aliases": ["Masonry Society", "TMS"]},
    "adubench_single_001279": {"rule": "numeric", "value": 479, "tolerance": 0},
}
ONE_FACT = "One fact on one page; no chain."
JUDGEMENT = "Gold not determined: the match is a judgement."
SECOND_REJECTED = {
    "adubench_single_000690": "Gold named in the question.",
    "adubench_single_000607": ONE_FACT,
    "adubench_single_000014": JUDGEMENT,
    "adubench_single_000596": ONE_FACT,
    "adubench_single_000727": JUDGEMENT,
    "adubench_single_001194": "Gold is not a term in the document.",
    "adubench_single_000539": ONE_FACT,
    "adubench_single_000672": "Gold named in the question.",
    "adubench_single_000534": ONE_FACT,
    "adubench_single_000638": ONE_FACT,
    "adubench_single_000397": ONE_FACT,
    "adubench_single_000350": ONE_FACT,
    "adubench_single_000998": ONE_FACT,
    "adubench_single_000814": ONE_FACT,
    "adubench_single_000259": ONE_FACT,
    "adubench_single_001346": ONE_FACT,
    "adubench_single_000100": ONE_FACT,
    "adubench_single_000468": JUDGEMENT,
    "adubench_single_000743": JUDGEMENT,
    "adubench_single_000366": "The PDF on disk has one page; the evidence pages are missing.",
    "adubench_single_000733": "Gold not located in the PDF text of the evidence pages.",
    "adubench_single_000212": ONE_FACT,
}
SECOND_TITLE_OVERRIDES = {
    "doc_000272": "Laboratory Biosafety Manual, Fourth Edition",
    "doc_000253": "Federal Register, Vol. 87, No. 226, 25 November 2022",
    "doc_000324": "2023 ESC Guidelines for the management of endocarditis",
    "doc_000191": "Mathematical Methods for Physicists, Instructor's Manual",
    "doc_000179": "EM 385-1-1 Safety and Health Requirements Manual",
    "doc_000190": "IAEA Occupational Exposure Safety Guide, Draft 5.0",
    "doc_000341": "Biosafety in Microbiological and Biomedical Laboratories, 4th Edition",
    "doc_000255": "NCCN Guidelines, Version 2.2019 (COL)",
    "doc_000305": "USGS Professional Paper 1807: Revised Hydrogeologic Framework of the Floridan Aquifer System",
    "doc_000307": "2021 ESC/EACTS Guidelines for the management of valvular heart disease",
}
THIRD_ACCEPTED = {
    "adubench_single_000285": {"rule": ALIASES, "aliases": ["P24,T,O,U", "P_{24,T,O,U}", "P24, T, O, U"]},
    "adubench_single_000384": {"rule": ALIASES, "aliases": ["Amlodipine"]},
    "adubench_single_000448": {"rule": ALIASES, "aliases": ["Federal Highway Administration"]},
    "adubench_single_001073": {"rule": ALIASES, "aliases": ["weight, measure, or count", "weight, measure (including volume, and dimensions), or count"]},
    "adubench_single_000565": {"rule": ALIASES, "aliases": ["POM-501"]},
}
THIRD_REJECTED = {
    "adubench_single_000115": "Gold not determined: 'matches' is a judgement.",
    "adubench_single_000294": "Gold not determined: 'best matches' is a judgement.",
    "adubench_single_000228": "Gold not determined: 'best matches' is a judgement.",
    "adubench_single_001291": "Gold not determined: 'reconciles' with a 'matching clue' is a judgement.",
    "adubench_single_000157": "Gold (Fm) does not answer the either-or question asked.",
    "adubench_single_000625": "Gold not determined: 'best matches' is a judgement.",
}
THIRD_TITLE_OVERRIDES = {
    "doc_000145": "HEC-17, Highways in the River Environment, 2nd Edition",
    "doc_000224": "HEC-25, Volume 2",
}
CORPORA = {
    "pages_holdout": {"first_position": FIRST_POSITION, "accepted": ACCEPTED, "rejected": REJECTED, "base": "pages_multihop",
                      "titles": TITLE_OVERRIDES, "set": "holdout"},
    "pages_holdout2": {"first_position": 47, "accepted": SECOND_ACCEPTED, "rejected": SECOND_REJECTED,
                       "base": "pages_holdout", "titles": SECOND_TITLE_OVERRIDES, "set": "holdout2"},
    "pages_unseen": {"first_position": 84, "accepted": THIRD_ACCEPTED, "rejected": THIRD_REJECTED,
                     "base": "pages_holdout2", "titles": THIRD_TITLE_OVERRIDES, "set": "unseen", "carries_questions": True},
}


def select_and_package() -> dict:
    """Write eval/talk2/<corpus>/ and data/talk2_corpus_<corpus>/pdf/. Requires Ollama for embeddings."""
    import bm25s
    import numpy as np
    import pymupdf

    definition = CORPORA[os.environ.get("TALK2_CORPUS", "")]
    accepted, rejected, title_overrides = definition["accepted"], definition["rejected"], definition["titles"]
    base_eval = REPOSITORY_ROOT / "eval" / "talk2" / definition["base"]
    base_pdf = REPOSITORY_ROOT / "data" / f"talk2_corpus_{definition['base']}" / "pdf"
    from llama_index.core import Settings

    from talk2.corpus_assembly import CORPUS_PDF, EVAL_ROOT
    from talk2.retrieval_index import configure_embedding

    configure_embedding()
    ordered = selection_order()
    positions = {r["question_id"]: i for i, r in enumerate(ordered)}
    last = max(positions[q] for q in accepted)
    examined = ordered[definition["first_position"]:last + 1]
    passed_over = {r["question_id"] for r in examined} - set(accepted)
    if passed_over != set(rejected):
        raise ValueError(f"rejections do not match the selection order: {passed_over ^ set(rejected)}")

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    for source in base_pdf.rglob("*.pdf"):
        target = CORPUS_PDF / source.relative_to(base_pdf)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source.resolve())
    manifest = list(csv.DictReader((base_eval / "corpus_manifest.tsv").open(), delimiter="\t"))
    selection = list(csv.DictReader((base_eval / "page_selection.tsv").open(), delimiter="\t"))
    questions = ([{**json.loads(line), "set": definition["set"]} for line in (base_eval / "questions.jsonl").open()]
                 if definition.get("carries_questions") else [])
    first_order = len(questions) + 1
    gold_on_evidence_pages = {}

    for order, record in enumerate(sorted((r for r in ordered if r["question_id"] in accepted),
                                          key=lambda r: positions[r["question_id"]]), first_order):
        question_id, document_id = record["question_id"], record["document"]["document_id"]
        document_path = f"xl-docbench/{document_id}"
        evidence_indexes = [page - 1 for page in evidence_page_numbers(record)]    # XL-DocBench pages are 1-based
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
            evidence_text = " ".join(re.sub(r"\s+", " ", texts[i]).lower() for i in evidence_indexes)
        decision = accepted[question_id]
        gold_terms = decision.get("aliases") or [t for group in decision.get("terms", []) for t in group]
        gold_on_evidence_pages[question_id] = [t for t in gold_terms if t.lower() in evidence_text]
        title = title_overrides.get(document_id) or record["title"]
        manifest.append({"path": f"{document_path}.pdf", "source": "xl-docbench", "doc_id": document_id, "title": title,
                         "language": "english", "pages": len(indexes), "present": True,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "original_page_indexes": " ".join(map(str, indexes))})
        subset_of = {index: position for position, index in enumerate(indexes)}
        selection += [{"question": question_id, "role": role, "rank": rank, "document": document_path,
                       "page_index": index, "subset_page_index": subset_of[index]}
                      for index, (role, rank) in chosen.items()]
        questions.append({
            "order": order, "id": question_id, "set": definition["set"], "source": "xl-docbench",
            "question": f'In the document "{title}": {record["question"]}', "gold": str(record["answer"]["value"]),
            "match": decision, "evidence_docs": [document_path],
            "evidence_page_indexes_0_based": {document_id: evidence_indexes},
            "difficulty": record["metadata"]["difficulty"], "domain": record["metadata"]["domain"],
            "verified_quote": " | ".join(re.sub(r"\s+", " ", str(item.get("quote")))[:200]
                                         for item in record["document"]["evidence_items"]),
            "verified_by": "Benchmark evidence quotes read and gold terms searched in the PDF text, 2 Nov 2026",
            "evidence_subset_pages": [{"document": document_path, "subset_page_index": subset_of[i]}
                                      for i in evidence_indexes],
        })

    for name, rows in (("corpus_manifest.tsv", manifest), ("page_selection.tsv", selection)):
        with (EVAL_ROOT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    with (EVAL_ROOT / "questions.jsonl").open("w") as out:
        for question in questions:
            out.write(json.dumps(question, ensure_ascii=False) + "\n")
    with (EVAL_ROOT / "question_selection.tsv").open("w") as out:
        out.write("position\tquestion_id\tdecision\treason\n")
        for record in examined:
            question_id = record["question_id"]
            out.write(f"{positions[question_id]}\t{question_id}\t{'accepted' if question_id in accepted else 'rejected'}"
                      f"\t{rejected.get(question_id, '')}\n")
    return {"questions": len(questions), "documents": len(manifest), "pages": sum(int(r["pages"]) for r in manifest),
            "gold_terms_on_evidence_pages": gold_on_evidence_pages}
