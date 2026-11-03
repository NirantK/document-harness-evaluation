# Conversion comparison

Corpus: the shared 531 pages of 98 documents (`pages_unseen`). No model calls.

Each variant is scored against the text PyMuPDF extracts from the same page. A word or number counts as lost when the variant holds fewer copies than that text. OCR gains on scanned pages go unscored, because this reference has no text for them.

Docling 2.145.0 wrote output for all 531 pages; 2 of them are empty. Run on 3 November 2026.

```
uv run --no-project --python 3.12 --with docling python convert_with_docling.py <pdf_folder> <output_folder>
TALK2_CORPUS=pages_unseen uv run python -m talk2.conversion_comparison <output_folder>
```

## All pages

| Variant | Words lost | Numbers lost | Lines over 1,000 characters | Longest line |
|---|---:|---:|---:|---:|
| Docling, OCR off, table structure on | 6.73% | 16.44% | 313 | 7174 |
| pymupdf4llm + LightOnOCR-3 | 2.05% | 11.96% | 157 | 9866 |
| pymupdf4llm alone | 1.65% | 3.91% | 174 | 9866 |
| Docling + restored lines | 0.54% | 1.65% | 149 | 7174 |
| pymupdf4llm + LightOnOCR-3 + restored lines | 0.45% | 1.37% | 6 | 1349 |
| pymupdf4llm alone + restored lines | 0.10% | 0.95% | 16 | 3681 |

## Converter pages

| Variant | Words lost | Numbers lost | Lines over 1,000 characters | Longest line |
|---|---:|---:|---:|---:|
| Docling, OCR off, table structure on | 12.26% | 20.69% | 124 | 7174 |
| pymupdf4llm + LightOnOCR-3 | 4.78% | 21.94% | 24 | 1691 |
| pymupdf4llm alone | 3.20% | 4.17% | 41 | 6594 |
| pymupdf4llm + LightOnOCR-3 + restored lines | 1.77% | 1.80% | 0 | 965 |
| Docling + restored lines | 0.95% | 1.87% | 93 | 7174 |
| pymupdf4llm alone + restored lines | 0.37% | 0.87% | 10 | 3681 |
