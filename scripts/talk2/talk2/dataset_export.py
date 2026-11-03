"""Write the Hugging Face dataset of the talk: questions, pages and answers, in a full and a public variant.

full:   everything, for local use. Not for publication, because some documents do not permit redistribution.
public: holds page images and text only for documents whose licence permits redistribution. XL-DocBench questions
        and restricted page text are replaced by identifiers that resolve_xl_docbench.py fills from the source.
"""

import csv
import json
import re
from pathlib import Path
from urllib.parse import quote

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPOSITORY_ROOT / "eval" / "talk2"
OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "hf_talk2_harness_evaluation"
CORPORA = ("pages_multihop", "pages_holdout", "pages_unseen", "pages_holdout2")
DATASET_REPOSITORY = "nirantk/document-harness-evaluation"
DATASET_FILE_BASE = f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/main/files"
PAGE_RENDER_DPI = 100
PERMITTED, UNCONFIRMED, LINKS_ONLY = "permitted", "unconfirmed", "links_only"
LICENCE_CLASSES = (("cc-by-4.0", PERMITTED), ("cc-by-sa", PERMITTED), ("etalab", PERMITTED), ("insee.fr", PERMITTED),
                   ("distribution statement a", PERMITTED), ("statistiques.developpement-durable", UNCONFIRMED),
                   ("data.inpi.fr", UNCONFIRMED))
RESTRICTED_PATH_PREFIXES = ("xl-docbench/", "vidore-finance_fr/")
OMITTED = "[omitted from the public variant: source licence]"
PAGE_MARKER = re.compile(r"<!-- page (\d+) -->")


def document_licences() -> dict[tuple[str, str], dict]:
    """Licence, source URL and redistribution class per (source, doc_id)."""
    import pyarrow.parquet as parquet

    records: dict[tuple[str, str], dict] = {}
    for line in (REPOSITORY_ROOT / "data" / "xl-docbench" / "documents.jsonl").open():
        row = json.loads(line)
        records["xl-docbench", row["document_id"]] = {
            "licence": "XL-DocBench lists links only and grants no redistribution rights",
            "source_url": row.get("url") or "", "redistribution": LINKS_ONLY}
    for path in (REPOSITORY_ROOT / "data" / "vidore_v3").glob("*/documents_metadata.parquet"):
        for row in parquet.read_table(path).to_pylist():
            licence = str(row.get("license") or "")
            redistribution = next((value for key, value in LICENCE_CLASSES if key in licence.lower()), UNCONFIRMED)
            records[f"vidore-{path.parent.name}", row["doc_id"]] = {
                "licence": licence, "source_url": row.get("url") or "", "redistribution": redistribution}
    return records


CUAD_LICENCE = {"licence": "CC BY 4.0, The Atticus Project, CUAD, DOI 10.5281/zenodo.4595826",
                "source_url": "https://doi.org/10.5281/zenodo.4595826", "redistribution": PERMITTED}


def page_selections() -> dict[tuple[str, int], list[str]]:
    """The roles each page holds, keyed by (document path without suffix, subset page index)."""
    selected: dict[tuple[str, int], list[str]] = {}
    for corpus in CORPORA:
        path = EVAL_ROOT / corpus / "page_selection.tsv"
        if path.exists():
            for row in csv.DictReader(path.open(), delimiter="\t"):
                label = f"{row['question']}:{row['role']}"
                roles = selected.setdefault((row["document"], int(row["subset_page_index"])), [])
                if label not in roles:
                    roles.append(label)
    return selected


def markdown_by_page(path: Path) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    current = 0
    for line in path.read_text(errors="replace").splitlines():
        marker = PAGE_MARKER.search(line)
        if marker:
            current = int(marker.group(1))
            continue
        pages.setdefault(current, []).append(line)
    return {page: "\n".join(lines).strip() for page, lines in pages.items()}


def build_pages(corpus: str, licences: dict) -> list[dict]:
    import pymupdf

    corpus_root = REPOSITORY_ROOT / "data" / f"talk2_corpus_{corpus}"
    selections = page_selections()
    rows = []
    for record in csv.DictReader((EVAL_ROOT / corpus / "corpus_manifest.tsv").open(), delimiter="\t"):
        stem = record["path"].removesuffix(".pdf")
        licence = CUAD_LICENCE if record["source"] == "cuad" else licences[record["source"], record["doc_id"]]
        markdown_path = corpus_root / "markdown" / f"{stem}.md"
        markdown = markdown_by_page(markdown_path) if markdown_path.exists() else {}
        original = [int(value) for value in record["original_page_indexes"].split()]
        with pymupdf.open(corpus_root / "pdf" / record["path"]) as document:
            for index, page in enumerate(document):
                rows.append({
                    "document": stem, "source": record["source"], "title": record["title"],
                    "language": record["language"], "subset_page_index": index,
                    "original_page_index": original[index] if index < len(original) else None,
                    "selected_for": json.dumps(selections.get((stem, index), [])),
                    "markdown": markdown.get(index, ""),
                    "image": {"bytes": page.get_pixmap(dpi=PAGE_RENDER_DPI).tobytes("png"),
                              "path": f"{stem.replace('/', '__')}__{index}.png"},
                    "source_url": licence["source_url"], "licence": licence["licence"],
                    "redistribution": licence["redistribution"],
                    "open_page_url": (hosted_file_url(stem, index) if licence["redistribution"] == PERMITTED else
                                      source_page_url(licence["source_url"], original[index] if index < len(original) else None))})
    return rows


def hosted_file_url(document: str, subset_page_index: int) -> str:
    return f"{DATASET_FILE_BASE}/{quote(document)}.pdf#page={subset_page_index + 1}"


def source_page_url(source_url: str, original_page_index: int | None) -> str:
    """The original document at its page, when the source URL serves a PDF directly."""
    if source_url.lower().split("?")[0].endswith(".pdf") and original_page_index is not None:
        return f"{source_url}#page={original_page_index + 1}"
    return source_url


def write_document_files(corpus: str, licences: dict, root: Path) -> int:
    """Copy each redistributable PDF subset and its Markdown under files/, and index every document with its source."""
    corpus_root = REPOSITORY_ROOT / "data" / f"talk2_corpus_{corpus}"
    index = []
    copied = 0
    for record in csv.DictReader((EVAL_ROOT / corpus / "corpus_manifest.tsv").open(), delimiter="\t"):
        stem = record["path"].removesuffix(".pdf")
        licence = CUAD_LICENCE if record["source"] == "cuad" else licences[record["source"], record["doc_id"]]
        hosted = licence["redistribution"] == PERMITTED
        if hosted:
            for folder, suffix in (("pdf", ".pdf"), ("markdown", ".md"), ("markdown_merged", ".md")):
                source = corpus_root / folder / f"{stem}{suffix}"
                if source.exists():
                    name = f"{stem}.merged.md" if folder == "markdown_merged" else f"{stem}{suffix}"
                    target = root / "files" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
                    copied += 1
        index.append({"document": stem, "source": record["source"], "title": record["title"], "pages_in_subset": record["pages"],
                      "original_page_indexes": record["original_page_indexes"], "hosted_in_this_dataset": hosted,
                      "file_url": f"{DATASET_FILE_BASE}/{quote(stem)}.pdf" if hosted else "",
                      "source_url": licence["source_url"], "licence": licence["licence"],
                      "redistribution": licence["redistribution"]})
    (root / "files").mkdir(parents=True, exist_ok=True)
    with (root / "files" / "document_index.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(index)
    return copied


def build_questions() -> list[dict]:
    rows: dict[str, dict] = {}
    for corpus in CORPORA:
        path = EVAL_ROOT / corpus / "questions.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            question = json.loads(line)
            rows.setdefault(question["id"], {
                "id": question["id"], "set": question["set"], "source": question["source"],
                "question": question["question"], "gold": str(question["gold"]), "match": json.dumps(question["match"]),
                "evidence_documents": json.dumps(question.get("evidence_docs", [])),
                "evidence_pages": json.dumps(question.get("evidence_subset_pages", [])),
                "verified_quote": question.get("verified_quote") or "", "difficulty": question.get("difficulty") or "",
                "domain": question.get("domain") or "", "text_withheld": False})
    return list(rows.values())


def question_records(corpus: str) -> dict[str, dict]:
    return {q["id"]: q for q in map(json.loads, (EVAL_ROOT / corpus / "questions.jsonl").open())}


def build_answers() -> list[dict]:
    from talk2.scoring import score

    rows = []
    for corpus in CORPORA:
        runs_root = EVAL_ROOT / corpus / "runs"
        if not runs_root.exists():
            continue
        questions = question_records(corpus)
        for configuration_directory in sorted(path for path in runs_root.glob("*/*") if path.is_dir()
                                              and not path.parent.name.startswith("gemma")):
            newest: dict[int, list[dict]] = {}
            for path in sorted(configuration_directory.glob("*.jsonl")):
                records = [json.loads(line) for line in path.open()]
                if len(records) == len(questions):
                    newest[records[0]["repeat"]] = records
            for records in newest.values():
                for record in records:
                    response = record.get("response") or {}
                    rows.append({
                        "corpus": corpus, "model": record["model"], "configuration": record["configuration"],
                        "quantization": record.get("quantization") or ("nvfp4" if record["model"] == "qwen3.5:9b" else
                                                                        record["model"].rsplit(":", 1)[-1]),
                        "hosted_provider": record.get("hosted_provider") or "", "seed": record.get("seed") or 42,
                        "repeat": record["repeat"], "question_id": record["id"], "question_source": questions[record["id"]]["source"],
                        "answer": str(response.get("answer", "")), "files": json.dumps(response.get("files", [])),
                        "correct": bool(score(questions[record["id"]], response)["correct"]),
                        "tool_calls": record.get("tool_calls") or 0, "wall_seconds": float(record.get("wall_seconds") or 0),
                        "prompt_tokens": record.get("prompt_tokens") or 0,
                        "completion_tokens": record.get("completion_tokens") or 0,
                        "git_revision": record.get("git_revision") or "", "trace": json.dumps(record.get("trace") or [],
                                                                                              ensure_ascii=False)})
    return rows


def public_question(row: dict) -> dict:
    if row["source"] == "xl-docbench":
        return {**row, "question": "", "gold": "", "match": "", "verified_quote": "", "text_withheld": True}
    if row["source"] == "vidore-finance_fr":
        return {**row, "verified_quote": ""}
    return row


def public_step(step: dict, withheld: bool) -> dict:
    """Keep the shape of a trace step, without text that may reproduce a restricted document."""
    arguments = json.dumps(step.get("arguments"), ensure_ascii=False)
    restricted = withheld or any(prefix in arguments for prefix in RESTRICTED_PATH_PREFIXES)
    kept = {key: step[key] for key in ("tool", "step", "arguments") if key in step}
    if withheld and step.get("tool") == "final_answer":
        kept["arguments"] = OMITTED
    for key in ("output", "reply", "prompt"):
        if key in step:
            text = str(step[key])
            hidden = restricted or any(prefix in text for prefix in RESTRICTED_PATH_PREFIXES)
            kept[key] = OMITTED if hidden else text
    return kept


def public_answer(row: dict) -> dict:
    withheld = row["question_source"] in ("xl-docbench", "vidore-finance_fr")
    trace = [public_step(step, withheld) for step in json.loads(row["trace"])]
    return {**row, "answer": OMITTED if withheld else row["answer"], "trace": json.dumps(trace, ensure_ascii=False)}


def write_parquet(rows: list[dict], path: Path, image_column: str = "") -> None:
    import pyarrow
    import pyarrow.parquet as parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pylist(rows)
    if image_column:
        features = {name: ({"_type": "Image"} if name == image_column else
                           {"dtype": str(table.schema.field(name).type).replace("large_", ""), "_type": "Value"})
                    for name in table.column_names}
        table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    parquet.write_table(table, path, compression="zstd")


CARD = """---
pretty_name: Talk 2 Harness Evaluation
license: other
language: [en, fr]
task_categories: [question-answering]
tags: [document-understanding, rag, agents, evaluation]
configs:
- config_name: questions
  data_files:
{question_splits}
- config_name: pages
  data_files:
  - split: train
    path: pages/train.parquet
- config_name: answers
  data_files:
  - split: train
    path: answers/train.parquet
---

# Talk 2 Harness Evaluation

Data behind the talk "Better Harness, Better Answers". One small model answered the same questions under different
harnesses, and every answer, score and tool call is here. Variant: **{variant}**.

| Config | Rows | Content |
| --- | ---: | --- |
| questions | {question_count} | Question, gold answer, scoring rule, evidence pages. Splits follow the evaluation stages. |
| pages | {page_count} | Page image, converted Markdown, source URL, licence, the questions that selected the page. |
| answers | {answer_count} | Model, quantization, provider, seed, configuration, repeat, answer, score, tool calls, seconds, tokens, full tool trace. |

## Evaluation stages

- `rag_misses`, `multi_hop`: tuning questions. The agent was changed after reading their traces.
- `holdout`: ten questions held back from the first agent, then used to change the harness. Treat them as tuning data
  for any later revision.
- `unseen`: twenty questions selected before their first run. No harness change came from their results or traces.
  Fifteen ran first as `holdout2`; five more were added to match the twenty tuning questions.

## Retrieval and models

- Embedding model: `qwen3-embedding:0.6b` in Ollama (Qwen/Qwen3-Embedding-0.6B, 1024 dimensions), 512-token chunks with
  50 overlap from LlamaIndex SentenceSplitter.
- Vector search: FAISS `IndexFlatIP`, an exact inner-product search over all chunks, top 5.
- BM25: bm25s. Hybrid fuses the top 20 of each ranking by reciprocal rank fusion (k = 60) and keeps 5.
- Page conversion: `maternion/LightOnOCR-3:0.8b` for pages with charts, images or little text, pymupdf4llm otherwise.
- Answer model: Qwen3.5-9B at temperature 0. Final runs use the 4-bit (fp4) weights on Darkbloom through OpenRouter,
  `qwen/qwen3.5-9b`, seeds 42 to 44. Earlier runs use the nvfp4 weights in Ollama, `qwen3.5:9b`, seed 42. Each row of `answers`
  names its `model`, `quantization`, `hosted_provider` and `seed`.

## Files

`files/` holds the PDF subset and Markdown of each document that permits redistribution. `files/document_index.tsv`
lists every document with its source URL, including those not hosted here. `pages.open_page_url` opens a page directly.

## Scoring

`correct` is re-scored with the current rules at export. `match` holds the rule for each question.

{variant_notes}

## Sources and attribution

- CUAD, The Atticus Project, CC BY 4.0, DOI 10.5281/zenodo.4595826. Contracts come from EDGAR.
- ViDoRe v3, annotations CC BY 4.0, arXiv 2601.08620. Each document has its own licence in the `licence` column. INSEE
  documents: "Source : Insee".
- XL-DocBench, Microsoft, arXiv 2608.00036. The benchmark lists links and grants no redistribution rights for documents.

This card is a practical summary and not legal advice.
"""

FULL_NOTES = "## Variant notes\n\nThis full variant is for local use only. It contains documents that do not permit redistribution."
PUBLIC_NOTES = ("## Variant notes\n\nPages with `redistribution` other than `permitted` are excluded. XL-DocBench question text, gold "
                "answers and scoring rules are withheld (`text_withheld`), and `resolve_xl_docbench.py` fills question text "
                "and the benchmark answer by `id` from microsoft/XL-DocBench. The current release lacks five of these questions, "
                "four of them in `unseen`, and the script names them. Answers to questions on restricted documents and "
                "tool outputs that touch them read `" + OMITTED + "`. Scores, call counts, seconds and tokens stay.")

RESOLVER = '''"""Fill withheld XL-DocBench question text from the source benchmark. Usage: python resolve_xl_docbench.py"""
import json
from urllib.request import urlopen

import pyarrow.parquet as parquet

BASE = "https://huggingface.co/datasets/microsoft/XL-DocBench/resolve/main/data/"
source = {}
for name in ("qa_single_doc.jsonl", "qa_cross_doc.jsonl"):
    for line in urlopen(BASE + name):
        row = json.loads(line)
        source[row["question_id"]] = row
for path_name in ("rag_misses", "multi_hop", "holdout", "holdout2", "unseen"):
    try:
        rows = parquet.read_table(f"questions/{path_name}.parquet").to_pylist()
    except FileNotFoundError:
        continue
    missing = [row["id"] for row in rows if row["text_withheld"] and row["id"] not in source]
    if missing:
        print(f"{path_name}: not in the current XL-DocBench release: {', '.join(missing)}")
    for row in rows:
        if row["text_withheld"] and row["id"] in source:
            row["question"], row["gold"] = source[row["id"]]["question"], str(source[row["id"]]["answer"]["value"])
    with open(f"questions_{path_name}_resolved.jsonl", "w") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows)
'''


def write_variant(variant: str, questions: list[dict], pages: list[dict], answers: list[dict]) -> dict:
    root = OUTPUT_ROOT / variant
    splits = sorted({row["set"] for row in questions})
    for split in splits:
        write_parquet([row for row in questions if row["set"] == split], root / "questions" / f"{split}.parquet")
    write_parquet(pages, root / "pages" / "train.parquet", image_column="image")
    write_parquet(answers, root / "answers" / "train.parquet")
    split_lines = "\n".join(f"  - split: {split}\n    path: questions/{split}.parquet" for split in splits)
    (root / "README.md").write_text(CARD.format(
        question_splits=split_lines, variant=variant, question_count=len(questions), page_count=len(pages),
        answer_count=len(answers), variant_notes=FULL_NOTES if variant == "full" else PUBLIC_NOTES))
    if variant == "public":
        (root / "resolve_xl_docbench.py").write_text(RESOLVER)
    return {"questions": len(questions), "pages": len(pages), "answers": len(answers)}


def export_dataset(corpus: str) -> dict:
    """Write both variants under data/hf_talk2_harness_evaluation/. Publishes nothing."""
    licences = document_licences()
    pages = build_pages(corpus, licences)
    questions = build_questions()
    answers = build_answers()
    public_pages = [row for row in pages if row["redistribution"] == PERMITTED]
    summary = {"full": write_variant("full", questions, pages, answers),
               "public": write_variant("public", [public_question(r) for r in questions], public_pages,
                                       [public_answer(r) for r in answers])}
    summary["public"]["files"] = write_document_files(corpus, licences, OUTPUT_ROOT / "public")
    return summary


def publish_public_variant(token: str) -> str:
    """Create or update the public dataset repository from data/hf_talk2_harness_evaluation/public/."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(DATASET_REPOSITORY, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=DATASET_REPOSITORY, repo_type="dataset", folder_path=OUTPUT_ROOT / "public",
                               commit_message="Update questions, pages, answers and document files", delete_patterns=["*"])
    return commit.commit_url
