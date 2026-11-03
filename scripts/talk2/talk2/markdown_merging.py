"""Merge the converter's page output with the PDF text layer, so conversion cannot lose text the PDF holds.

The vision converter reads charts and tables that the text layer lacks, and drops words and numbers the text layer
has. Merging keeps both: the converter's Markdown stays as the page body, and every text-layer line holding a word or
number that the body lacks follows it. Long lines are then split at sentence ends, because search tools show lines.
"""

import collections
import re
import unicodedata

import pymupdf

TOKEN = re.compile(r"\d[\d,.]*\d|\d|[^\W_]{2,}", re.UNICODE)
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[\"“(\[]?[A-Z0-9-])")
LONG_LINE = 500
RESIDUE_HEADING = "<!-- text layer lines absent from the converted page -->"


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return TOKEN.findall(re.sub(r"[*_`#|\\]", " ", text))


def text_layer_lines(page: pymupdf.Page) -> list[str]:
    """Lines of the text layer in reading order, with the spans of each line joined."""
    lines = []
    blocks = page.get_text("dict", sort=True)["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
            if text:
                lines.append(text)
    return lines


def absent_lines(body: str, lines: list[str]) -> list[str]:
    """Text-layer lines holding at least one token that the body does not hold, consuming tokens as they match."""
    available = collections.Counter(tokens(body))
    kept = []
    for line in lines:
        line_tokens = tokens(line)
        missing = False
        for token in line_tokens:
            if available[token] > 0:
                available[token] -= 1
            else:
                missing = True
        if missing:
            kept.append(line)
    return kept


def split_long_lines(markdown: str) -> str:
    """Split prose lines over LONG_LINE characters at sentence ends. Table rows stay whole."""
    output = []
    for line in markdown.splitlines():
        if len(line) <= LONG_LINE or line.lstrip().startswith("|"):
            output.append(line)
            continue
        output.extend(part for part in SENTENCE_END.split(line) if part)
    return "\n".join(output)


def merged_page(body: str, page: pymupdf.Page) -> str:
    residue = absent_lines(body, text_layer_lines(page))
    merged = body.strip()
    if residue:
        merged += f"\n\n{RESIDUE_HEADING}\n" + "\n".join(residue)
    return split_long_lines(merged)
