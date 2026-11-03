"""Place the 20 validation questions on the same documents and pages as the 20 test questions.

The 222 pages of pages_multihop all sit inside the 531 pages of pages_unseen, so pages_unseen is the shared corpus.
pages_validation holds the validation questions and links to the corpus files and indexes of pages_unseen.
Run from the repository root: uv run --project scripts/talk2 python scripts/talk2/assemble_validation_corpus.py
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "talk2"
SHARED, EARLIER, TARGET = "pages_unseen", "pages_multihop", "pages_validation"


def manifest_pages(corpus: str) -> dict[str, list[int]]:
    rows = csv.DictReader((EVAL / corpus / "corpus_manifest.tsv").open(), delimiter="\t")
    return {row["path"].removesuffix(".pdf"): [int(p) for p in row["original_page_indexes"].split()] for row in rows}


def main() -> None:
    earlier, shared = manifest_pages(EARLIER), manifest_pages(SHARED)
    (EVAL / TARGET).mkdir(exist_ok=True)
    for name in ("corpus_manifest.tsv", "page_selection.tsv"):
        shutil.copy(EVAL / SHARED / name, EVAL / TARGET / name)
    lines = []
    for raw in (EVAL / EARLIER / "questions.jsonl").open():
        question = json.loads(raw)
        for entry in question.get("evidence_subset_pages", []):
            original = earlier[entry["document"]][entry["subset_page_index"]]
            entry["subset_page_index"] = shared[entry["document"]].index(original)
        lines.append(json.dumps(question, ensure_ascii=False))
    (EVAL / TARGET / "questions.jsonl").write_text("\n".join(lines) + "\n")
    for link, source in ((ROOT / "data" / f"talk2_corpus_{TARGET}", f"talk2_corpus_{SHARED}"),
                         (EVAL / "cache" / TARGET, SHARED)):
        if not link.is_symlink():
            link.symlink_to(source, target_is_directory=True)
    print(f"{TARGET}: {len(lines)} questions, {sum(map(len, shared.values()))} pages, {len(shared)} documents")


if __name__ == "__main__":
    main()
