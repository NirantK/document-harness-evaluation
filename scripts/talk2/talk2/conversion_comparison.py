"""Compare page-level Markdown variants against the PDF text layer. No model calls.

Usage: TALK2_CORPUS=pages_holdout2 uv run python -m talk2.conversion_comparison [docling_output_folder]
"""

import collections
import json
import re
import sys
from pathlib import Path

import pymupdf
import pymupdf4llm

from talk2.corpus_assembly import CORPUS_PDF
from talk2.markdown_conversion import cache_path, manifest_rows
from talk2.markdown_merging import merged_page, tokens

DIGIT = re.compile(r"\d")


def accumulate(totals: collections.Counter, page_text: str, markdown: str) -> None:
    expected, present = collections.Counter(tokens(page_text)), collections.Counter(tokens(markdown))
    lost = expected - present
    for token, count in expected.items():
        totals["numbers" if DIGIT.search(token) else "words"] += count
    for token, count in lost.items():
        totals["numbers_lost" if DIGIT.search(token) else "words_lost"] += count
    totals["longest_line"] = max(totals["longest_line"], max((len(line) for line in markdown.splitlines()), default=0))
    totals["lines_over_1000"] += sum(len(line) > 1000 for line in markdown.splitlines())


def compare(docling_folder: Path | None) -> dict[str, dict[str, collections.Counter]]:
    scopes = {"all pages": collections.defaultdict(collections.Counter),
              "converter pages": collections.defaultdict(collections.Counter)}
    for row in manifest_rows():
        docling = {}
        if docling_folder:
            path = docling_folder / (row["path"].removesuffix(".pdf").replace("/", "__") + ".json")
            docling = json.loads(path.read_text()) if path.exists() else {}
        with pymupdf.open(CORPUS_PDF / row["path"]) as document:
            for index, page in enumerate(document):
                record = json.loads(cache_path(row, index).read_text())
                body = record["markdown"]
                on_converter = record.get("route") == "converter"
                library = (pymupdf4llm.to_markdown(document, pages=[index], show_progress=False, use_ocr=False)
                           if on_converter else body)
                variants = {"A router": body, "B pymupdf4llm": library, "D router + residue": merged_page(body, page),
                            "E pymupdf4llm + residue": merged_page(library, page)}
                if docling:
                    variants["G docling"] = docling.get(str(index), "")
                    variants["H docling + residue"] = merged_page(docling.get(str(index), ""), page)
                text = page.get_text()
                for name, markdown in variants.items():
                    accumulate(scopes["all pages"][name], text, markdown)
                    if on_converter:
                        accumulate(scopes["converter pages"][name], text, markdown)
    return scopes


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    pymupdf.TOOLS.mupdf_display_errors(False)
    for scope, variants in compare(folder).items():
        print(scope)
        for name, totals in sorted(variants.items()):
            print(f"  {name:26s} words lost {totals['words_lost'] / totals['words']:6.2%}  "
                  f"numbers lost {totals['numbers_lost'] / totals['numbers']:6.2%}  "
                  f"lines over 1000: {totals['lines_over_1000']:4d}  longest {totals['longest_line']}")
