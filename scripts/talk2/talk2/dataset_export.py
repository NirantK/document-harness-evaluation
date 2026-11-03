"""Write the Hugging Face dataset of the talk: questions, pages and answers, in a full and a public variant.

full:   everything, for local use. Not for publication, because some documents do not permit redistribution.
public: holds every question, gold answer, scoring rule, quote, model answer and score. Page images, page text and tool
        output appear only for documents whose licence permits redistribution.
"""

import csv
import json
import re
from pathlib import Path
from urllib.parse import quote

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPOSITORY_ROOT / "eval" / "talk2"
OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "hf_talk2_harness_evaluation"
CORPORA = ("pages_validation", "pages_unseen", "pages_multihop", "pages_holdout", "pages_holdout2")
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
                "id": question["id"], "set": {"unseen": "test"}.get(question["set"], question["set"]), "source": question["source"],
                "question": question["question"], "gold": str(question["gold"]), "match": json.dumps(question["match"]),
                "evidence_documents": json.dumps(question.get("evidence_docs", [])),
                "evidence_pages": json.dumps(question.get("evidence_subset_pages", [])),
                "verified_quote": question.get("verified_quote") or "", "difficulty": question.get("difficulty") or "",
                "domain": question.get("domain") or ""})
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


def public_step(step: dict) -> dict:
    """Keep each trace step whole, except tool output and prompts that may reproduce a restricted document."""
    arguments = json.dumps(step.get("arguments"), ensure_ascii=False)
    restricted = any(prefix in arguments for prefix in RESTRICTED_PATH_PREFIXES)
    kept = {key: step[key] for key in ("tool", "step", "arguments", "reply") if key in step}
    for key in ("output", "prompt"):
        if key in step:
            text = str(step[key])
            kept[key] = OMITTED if restricted or any(prefix in text for prefix in RESTRICTED_PATH_PREFIXES) else text
    return kept


def public_answer(row: dict) -> dict:
    return {**row, "trace": json.dumps([public_step(step) for step in json.loads(row["trace"])], ensure_ascii=False)}


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

- `rag_misses`, `multi_hop`: validation questions. The agent was changed after reading their traces.
- `holdout`: ten questions held back from the first agent, then used to change the harness. Treat them as validation data
  for any later revision.
- `test`: twenty questions selected before their first run. No harness change came from their results or traces.
  Fifteen ran first as `holdout2`; five more were added to match the twenty validation questions. Validation and test search the same 531 pages. In `answers`, corpus `pages_validation` holds the validation runs and `pages_unseen` the test runs; `pages_multihop` holds earlier validation runs on 222 of those pages, where the harness was tuned.

## Retrieval and models

- Embedding model: `qwen3-embedding:0.6b` in Ollama (Qwen/Qwen3-Embedding-0.6B, 1024 dimensions), 512-token chunks with
  50 overlap from LlamaIndex SentenceSplitter.
- Vector search: FAISS `IndexFlatIP`, an exact inner-product search over all chunks, top 5.
- BM25: bm25s. Hybrid fuses the top 20 of each ranking by reciprocal rank fusion (k = 60) and keeps 5.
- Page conversion: `maternion/LightOnOCR-3:0.8b` for pages with charts, images or little text, pymupdf4llm otherwise.
- Answer model: Qwen3.5-9B at temperature 0. Final runs use the 4-bit (fp4) weights on Darkbloom through OpenRouter,
  `qwen/qwen3.5-9b`, three runs. Earlier runs use the nvfp4 weights in Ollama, `qwen3.5:9b`, one run. Each row of `answers`
  names its `model`, `quantization`, `hosted_provider` and the `seed` sent; at temperature 0 the seed has little effect.
- Iterations: Trace Fixes is configuration `merged_grep_revision2`, Grep Token Reduction is `merged_grep_revision3`, and
  Grep Token Reduction + Thinking is `merged_grep_revision3_thinking`.
- BM25 in the loop (`merged_bm25_loop`), BM25 in the loop + Thinking (`merged_bm25_loop_thinking`) and Hybrid in the loop
  (`merged_fusion_loop`) are the grep agent with a ranked search tool, added after the test scores of the other
  configurations were known. Hybrid means embedding + BM25 with reciprocal rank fusion, top 5. `text_bm25_loop` is BM25 in
  the loop on PDF text.
- BM25 Token Reduction (`merged_bm25_excerpt_fixed_count`) puts BM25 search into the Grep Token Reduction agent, shows an
  excerpt of 600 characters around the matching words and has no turn limit. `merged_bm25_excerpt_selected_count` lets the
  model choose the passage count. `merged_bm25_token_reduction` and `merged_bm25_selected_count` are the first versions of
  these two. All four were tuned on the validation questions and added after the test scores of the others were known.

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
- XL-DocBench, Microsoft, arXiv 2608.00036. Questions and answers come from the benchmark; it lists links to documents and grants no redistribution rights for them.

This card is a practical summary and not legal advice.
"""

FULL_NOTES = "## Variant notes\n\nThis full variant is for local use only. It contains documents that do not permit redistribution."
PUBLIC_NOTES = ("## Variant notes\n\nEvery question, gold answer, scoring rule, evidence quote, model answer and score is here. Five "
                "XL-DocBench questions, four of them in `test`, come from an earlier release of that benchmark. Pages with "
                "`redistribution` other than `permitted` are excluded, because their source licences do not allow copies; "
                "`source_url` and `open_page_url` lead to them. Tool outputs and prompts that reproduce those documents read `"
                + OMITTED + "`.")


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
    return {"questions": len(questions), "pages": len(pages), "answers": len(answers)}


def export_dataset(corpus: str) -> dict:
    """Write both variants under data/hf_talk2_harness_evaluation/. Publishes nothing."""
    licences = document_licences()
    pages = build_pages(corpus, licences)
    questions = build_questions()
    answers = build_answers()
    public_pages = [row for row in pages if row["redistribution"] == PERMITTED]
    summary = {"full": write_variant("full", questions, pages, answers),
               "public": write_variant("public", questions, public_pages,
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
