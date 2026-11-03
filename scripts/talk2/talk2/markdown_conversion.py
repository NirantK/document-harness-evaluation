"""DSH step 1: convert every corpus page to Markdown.

A uniform page router decides the method per page:
- converter (LightOnOCR-3-0.8B in Ollama, ParseBench's grounding prompt) when the page has under 200 characters of
  text, images covering more than 10% of its area, or 40+ vector drawings (charts, ruled tables);
- pymupdf4llm otherwise (text-native pages; no model involved).

Each page is cached under eval/talk2/cache/markdown/<file sha256>/<page>.json, so the run resumes after an
interruption. Documents are assembled to data/talk2_corpus/markdown/<source>/<doc_id>.md with page markers.
"""

import asyncio
import base64
import csv
import json
import math
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pymupdf
import pymupdf4llm

from talk2.corpus_assembly import CACHE_ROOT as SHARED_CACHE_ROOT, CORPUS_PDF, MANIFEST

CONVERTER_MODEL = "maternion/LightOnOCR-3:0.8b"
CONVERTER_PROMPT = "grounding"
MAX_PIXELS = 2_000_000
MAX_DPI = 200
MAX_OUTPUT_TOKENS = 8192
OLLAMA_CHAT = "http://localhost:11434/api/chat"
CACHE_ROOT = SHARED_CACHE_ROOT / "markdown"
CORPUS_MARKDOWN = CORPUS_PDF.parent / "markdown"
BLOCK_MARKER = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)[ \t]*")
HTML_TABLE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)


def route_page(page: pymupdf.Page) -> str:
    area = abs(page.rect) or 1.0
    text_length = len(page.get_text().strip())
    image_area = sum(abs(rect) for image in page.get_images(full=True) for rect in page.get_image_rects(image[0]))
    if text_length < 200 or image_area / area > 0.10 or len(page.get_drawings()) >= 40:
        return "converter"
    return "pymupdf4llm"


def render_page(page: pymupdf.Page) -> str:
    width_in, height_in = page.rect.width / 72, page.rect.height / 72
    dpi = min(MAX_DPI, math.floor(math.sqrt(MAX_PIXELS / (width_in * height_in))))
    return base64.b64encode(page.get_pixmap(dpi=dpi).tobytes("png")).decode()


class TableToMarkdown(HTMLParser):
    """Turn one HTML table into a Markdown pipe table (spans are not expanded)."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.rows:
            self.rows[-1].append(" ".join("".join(self.cell).split()).replace("|", "\\|"))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def markdown(self) -> str:
        rows = [r for r in self.rows if r]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)


def html_tables_to_markdown(text: str) -> str:
    def replace(match: re.Match) -> str:
        parser = TableToMarkdown()
        parser.feed(match.group(0))
        return parser.markdown() or match.group(0)

    return HTML_TABLE.sub(replace, text)


def converter_output_to_markdown(raw: str) -> str:
    """Drop the grounding boxes, keep block text; charts and tables become Markdown tables."""
    text = BLOCK_MARKER.sub("", raw)
    return html_tables_to_markdown(text).strip()


async def convert_with_model(client: httpx.AsyncClient, image: str) -> dict:
    started = time.time()
    response = await client.post(OLLAMA_CHAT, timeout=1800, json={
        "model": CONVERTER_MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": CONVERTER_PROMPT, "images": [image]}],
        "options": {"temperature": 0.1, "num_ctx": 16384, "num_predict": MAX_OUTPUT_TOKENS},
    })
    response.raise_for_status()
    data = response.json()
    raw = data["message"]["content"]
    return {"markdown": converter_output_to_markdown(raw), "raw": raw, "output_tokens": data.get("eval_count", 0),
            "truncated": data.get("eval_count", 0) >= MAX_OUTPUT_TOKENS, "seconds": round(time.time() - started, 1)}


def manifest_rows() -> list[dict]:
    with MANIFEST.open() as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["present"] == "True"]


def cache_path(row: dict, page_index: int) -> Path:
    return CACHE_ROOT / row["sha256"] / f"{page_index:05d}.json"


def pending_pages(limit_sources: set[str] | None = None) -> list[tuple[dict, int]]:
    pending = []
    for row in manifest_rows():
        if limit_sources and row["source"] not in limit_sources:
            continue
        for page_index in range(int(row["pages"])):
            if not cache_path(row, page_index).exists():
                pending.append((row, page_index))
    return pending


def write_cache(row: dict, page_index: int, record: dict) -> None:
    path = cache_path(row, page_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False))
    temporary.rename(path)


async def convert_pages(pages: list[tuple[dict, int]], workers: int, progress_every: int = 25,
                        only_route: str = "") -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for item in pages:
        queue.put_nowait(item)
    done = 0
    started = time.time()
    open_documents: dict[str, pymupdf.Document] = {}

    def document_for(row: dict) -> pymupdf.Document:
        if row["path"] not in open_documents:
            open_documents[row["path"]] = pymupdf.open(CORPUS_PDF / row["path"])
        return open_documents[row["path"]]

    async def worker(client: httpx.AsyncClient):
        nonlocal done
        while not queue.empty():
            row, page_index = queue.get_nowait()
            document = document_for(row)
            page = document[page_index]
            route = route_page(page)
            if only_route and route != only_route:
                continue
            if route == "converter":
                try:
                    record = await convert_with_model(client, render_page(page))
                except Exception as error:  # keep going; the page falls back to its text layer
                    record = {"markdown": page.get_text(), "error": f"{type(error).__name__}: {error}"}
            else:
                record = {"markdown": pymupdf4llm.to_markdown(document, pages=[page_index], show_progress=False, use_ocr=False)}
            record["route"] = route
            write_cache(row, page_index, record)
            done += 1
            if done % progress_every == 0:
                elapsed = time.time() - started
                print(f"{done}/{len(pages)} pages, {elapsed / done:.1f} s/page, "
                      f"eta {(len(pages) - done) * elapsed / done / 3600:.1f} h", flush=True)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[worker(client) for _ in range(workers)])


def assemble_documents() -> tuple[int, int]:
    documents = missing = 0
    for row in manifest_rows():
        parts = []
        for page_index in range(int(row["pages"])):
            path = cache_path(row, page_index)
            if not path.exists():
                missing += 1
                continue
            record = json.loads(path.read_text())
            parts.append(f"<!-- page {page_index} -->\n{record['markdown'].strip()}\n")
        target = CORPUS_MARKDOWN / Path(row["path"]).with_suffix(".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(parts))
        documents += 1
    return documents, missing


def assemble_merged_documents() -> tuple[int, int, int]:
    """Write the markdown_merged corpus: each cached page merged with the text-layer lines its Markdown lost."""
    from talk2.markdown_merging import merged_page

    documents = missing = pages_with_residue = 0
    merged_root = CORPUS_MARKDOWN.parent / "markdown_merged"
    for row in manifest_rows():
        parts = []
        with pymupdf.open(CORPUS_MARKDOWN.parent / "pdf" / row["path"]) as pdf:
            for page_index in range(int(row["pages"])):
                path = cache_path(row, page_index)
                if not path.exists():
                    missing += 1
                    continue
                body = merged_page(json.loads(path.read_text())["markdown"], pdf[page_index])
                pages_with_residue += "text layer lines absent" in body
                parts.append(f"<!-- page {page_index} -->\n{body.strip()}\n")
        target = merged_root / Path(row["path"]).with_suffix(".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(parts))
        documents += 1
    return documents, missing, pages_with_residue
