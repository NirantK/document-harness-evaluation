# Talk 2 evaluation method (frozen with questions.jsonl)

Plain RAG against the **Deep Seek Harness (DSH)** on the ten rag-misses questions. This file, `questions.jsonl` and
`corpus_manifest.tsv` are frozen together under the git tag `talk2-eval-frozen`. Any change after the tag is listed
in `results.md` under "Post-freeze changes".

## Corpora
Two corpora, each run with all four configurations; select one with `TALK2_CORPUS=pages100|full`.

### Primary: `pages100` (119 pages from 58 documents; `pages100/`)
Converting all 22,253 pages with the local converter takes over 40 hours, so the talk's headline numbers come from a
page-level corpus selected by a fixed rule (`scripts/talk2/talk2/page_selection.py`, output `pages100/page_selection.tsv`):
1. **Evidence pages**, from the benchmark annotations: the ViDoRe qrels, the XL-DocBench evidence pages, the CUAD
   pages that hold the cited clause. #6 uses every page in the six contracts that matches
   "laws of (the) (State of) Israel". #7 uses the whole Ultragenyx contract, because an absence needs the full file.
2. **4 embedding negatives per question:** the highest-ranked non-evidence pages for the question in the
   full-corpus RAG index. These are the pages that mislead retrieval.
3. **4 keyword negatives per question:** the highest BM25-ranked non-evidence pages for the question text over all
   22,253 pages. These are the pages that make grep discriminate.

Questions are processed in order 1 → 10; a page already selected is not selected again. Each source document becomes
one subset PDF with its original file name. In `pages100/questions.jsonl`, #6 reads "Which of these contracts…"
instead of "these 510 contracts"; nothing else changes.

### Scale check: `full` (605 PDFs, 22,253 pages; `corpus_manifest.tsv`)
Plain RAG (`rag`) runs on the full corpus as a scale comparison. The Markdown configurations run on it only if the
full conversion finishes.

| Source | Files | Pages |
|---|---|---|
| CUAD (all contracts) | 510 | 9,348 |
| ViDoRe v3 energy | 41 | 2,225 |
| ViDoRe v3 finance_fr | 5 | 2,384 |
| ViDoRe v3 hr | 14 | 1,110 |
| ViDoRe v3 industrial | 27 | 5,244 |
| XL-DocBench (context documents of questions 1 and 10) | 8 | 1,942 |

Three XL-DocBench context documents for question 1 (INCB 2017 reports in Spanish, Russian and French: doc_000098,
doc_000117, doc_000334) could not be downloaded from digitallibrary.un.org or incb.org. They were distractors only.

## Answer model
- Local, through Ollama's native /api/chat endpoint: `hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`
  (35B total, 3B active, 4-bit). The same model, settings and answer schema for all four configurations.
- Temperature 0, fixed seed 42, thinking disabled, context 49,152 tokens (measured on 29 Oct: 18 GB resident, 48 tok/s decode; 64k left only 11% memory free). Smoke test passed: well-formed tool call, then a correct answer from the tool result. qwen3.8:27b-mlx was also tested and returned empty content because its MLX runner ignores `think: false`.
- Each configuration is run 3 times. Per question, the reported result is the majority of the 3. Totals are given
  as the median, with the min–max range.

## Answer schema (every configuration)
`{"answer": str, "files": [str], "found": "yes" | "no" | "not_found"}`. The prompt allows "not_found".

## Configurations
1. **rag**: PyMuPDF `page.get_text()` → LlamaIndex `SentenceSplitter(chunk_size=512, chunk_overlap=50)` with the
   Qwen3 tokenizer → Qwen3-Embedding-0.6B → FAISS `IndexFlatIP` over normalised vectors → top 5 → answer prompt with
   chunk text and file path. The retrieved chunks and their scores are logged.
2. **markdown_rag** (DSH step 1): identical to `rag`, but the input text is the converted Markdown corpus.
3. **markdown_grep** (DSH steps 1+2): no embeddings. A tool-calling loop over the Markdown corpus:
   - `search(pattern, path_glob, mode=lines|files|count, fixed_string, ignore_case, context)`: ripgrep. Output is
     capped at about 6,000 characters, followed by "N more matches in M files; narrow the pattern". When nothing
     matches, it returns the explicit line "0 matches in N files under <glob>".
   - `read(path, start_line, line_count ≤ 200)` and `list_files(glob)`, which shows title, language and page count.
   - The budget is 20 tool calls, after which a final answer is forced. The system prompt mentions neither
     languages nor memory.
4. **markdown_grep_memory** (DSH steps 1+2+3): `markdown_grep` plus exactly two changes:
   - **Translate:** a pre-step gives the model the corpus language list and asks for the question's search terms in
     each language. They are injected as "suggested search terms".
   - **Memory:** a `record_note(fact, file, line)` tool. Notes stay pinned in the prompt while tool outputs older
     than the last 3 are elided. Notes and "question N was answered from file F" carry across questions in the
     fixed order 1 → 10.

## Engines
Ollama serves every model: the answer model, the embeddings (`qwen3-embedding:0.6b`) and the converter
(`maternion/LightOnOCR-3:0.8b`). No other inference engine is used.

## Converter (DSH step 1)
Higher resolution did not pay on the 30 highest-loss pages: 4 MP at 300 DPI with the grounding prompt lost 25.2% of text against a 25.8% baseline, at 378 s. A plain OCR prompt at 2 MP lost 19.6% at 95 s.
The router was not changed, because that trial covered only the 30 pages it was chosen to help and no scored run has used the plain prompt.

LightOnOCR-3-0.8B (Apache-2.0; ParseBench 74.6 overall, 64.7 on charts) in Ollama, with ParseBench's grounding
prompt so charts come out as tables (converted from HTML to Markdown tables). Pages are rendered at no more than
2M pixels and 200 DPI; ParseBench's reference setting is 5M pixels and 400 DPI. Its chart score, re-measured locally with `parse-bench run … --group chart` on all 568 pages, is 58.3% of data-point rules at this setting; pymupdf4llm alone scores 1.1%. Both are saved in
`parsebench_chart.json`. A uniform page router applies to every corpus. Pages with fewer than 200 characters of
text, images covering more than 10% of the page, or 40 or more vector drawings go to the converter; every other page
goes to `pymupdf4llm` with OCR off. On the full corpus that routes 6,128 of 22,253 pages to the converter, which
measured 20–46 s per page; on `pages100` it routes 30 of 119. All four evidence pages that hold a table, a chart or a
scan (#2, #3, #4, #5) route to the converter.

## Scoring (`match` in questions.jsonl)
- `exact_string`: the normalised gold string appears in `answer`, and no `reject` string appears.
- `numeric`: the first number in `answer` (French and English separators accepted) lies within the tolerance or the
  low–high range, and no `reject` value appears.
- `alias_set`: any alias appears.
- `range_both_bounds`: both bounds appear, in digits or words.
- `file_set`: the reported `files`, mapped to corpus doc_ids, equal the gold set exactly. Precision and recall are
  reported as well.
- `found_no`: `found == "no"`.

Borderline auto-scores go to `review.tsv` for a human decision, which `results.md` records.

## Gold verification notes (28 Oct 2026)
- ViDoRe `page_number_in_doc` is 0-based: the PDF page index equals the qrel number.
- Question 3's gold is read off a bar with no data label (Graph 2.16). The accepted range is 49.5–51.5 EUR because
  the neighbouring ICT bar is about 49.
- Question 5's answer page (TO 00-25-234, page index 67) has 724 characters of PDF text, including the answer sentence,
  which stores "five" with an fi ligature. A plain search for "five" misses that line, and 14 of the document's 226 pages
  hold no PDF text. Corrected 3 Nov 2026; this note first said the page had no text.
- Question 8's distractor is real: LVMH's 2023 report gives 48.60 % for the same Arnault family group.
