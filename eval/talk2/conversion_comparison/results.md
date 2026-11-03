# Conversion comparison

Corpus: `pages_holdout2`, 481 pages of 93 documents. No model calls.

Each variant is scored against the text PyMuPDF extracts from the same page. A word or number counts as lost when the variant holds fewer copies than that text. OCR gains on scanned pages go unscored, because this reference has no text for them.

Docling 2.145.0 wrote output for all 481 pages; 2 of them are empty. Rerun on 3 November 2026 with the same output.

```
uv run --no-project --python 3.12 --with docling python convert_with_docling.py <pdf_folder> <output_folder>
TALK2_CORPUS=pages_holdout2 uv run python -m talk2.conversion_comparison <output_folder>
```

## All pages

| Variant | Words lost | Numbers lost | Lines over 1,000 characters | Longest line |
|---|---:|---:|---:|---:|
| Docling, OCR off, table structure on | 7.07% | 16.77% | 263 | 7174 |
| pymupdf4llm + LightOnOCR-3 | 2.19% | 12.81% | 150 | 9866 |
| pymupdf4llm alone | 1.73% | 4.06% | 167 | 9866 |
| Docling + restored lines | 0.59% | 1.76% | 104 | 7174 |
| pymupdf4llm + LightOnOCR-3 + restored lines | 0.47% | 1.44% | 5 | 1349 |
| pymupdf4llm alone + restored lines | 0.09% | 0.98% | 15 | 3681 |

## Converter pages

| Variant | Words lost | Numbers lost | Lines over 1,000 characters | Longest line |
|---|---:|---:|---:|---:|
| Docling, OCR off, table structure on | 12.57% | 20.55% | 124 | 7174 |
| pymupdf4llm + LightOnOCR-3 | 5.00% | 22.63% | 24 | 1691 |
| pymupdf4llm alone | 3.22% | 4.20% | 41 | 6594 |
| pymupdf4llm + LightOnOCR-3 + restored lines | 1.80% | 1.81% | 0 | 965 |
| Docling + restored lines | 1.00% | 1.93% | 93 | 7174 |
| pymupdf4llm alone + restored lines | 0.33% | 0.85% | 10 | 3681 |
