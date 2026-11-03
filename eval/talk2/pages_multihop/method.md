# Talk 2 multi-hop extension (frozen under the tag `talk2-multihop-frozen`)

The ten rag-miss questions plus ten multi-hop questions, on a larger corpus. The rest of the method is in
`../method.md`. `pages100` and its results stay as they were: every question here runs on a different corpus, so
scores from the two corpora are not directly comparable.

## Questions (`questions.jsonl`, field `set`)
Rule, fixed before any run (`scripts/talk2/talk2/multihop_selection.py`):
- **Source:** XL-DocBench single-document questions with reasoning type `reference_chain` (human-annotated evidence).
- **Filters:** answerable, evidence on 1 to 3 distinct pages, an answer of at most 8 words, an English document, the
  PDF on disk, a document not already in `pages100`, one question per document. 95 questions pass.
- **Order:** shuffled with seed 42 within each domain, then round-robin across domains.
- **Gold check:** each question in that order was checked against its PDF, and the first ten that passed were taken.
  It was rejected if the gold was not uniquely determined, the question needed one fact only, the gold was named in the
  question, or the gold could be guessed without the document. `question_selection.tsv` lists the 19 questions
  examined: 10 accepted, 9 rejected, with reasons.

Each question names its document (`In the document "<title>": …`), because XL-DocBench asks them of one known document
and they are ambiguous across a corpus. Titles missing from the benchmark were taken from page 1 of the PDF for
doc_000219 and doc_000067.

Scoring is more lenient than XL-DocBench's exact match: the gold or an alias only has to appear in the answer.
`all_terms` needs every part of a multi-part gold. Aliases were written before any run.

## Corpus (`corpus_manifest.tsv`, `page_selection.tsv`)
`pages100` (119 pages), plus per added question its evidence pages and 8 hard negatives from the same document: the 4
highest-ranked non-evidence pages by embedding and the 4 by BM25. That gives 222 pages in 68 documents. The hop has to
be found among pages of one document, so those are the pages that mislead. Negatives come from the same rule for every
question and from no model's answers.

## Answer models
`qwen3.5:9b` and `gemma4:12b-it-q4_K_M` (7.6 GB and 8.0 GB; Ollama has no 9B Gemma, and this is the same size class), with the
settings in `../method.md`. Each runs all seven configurations with 3 repeats.
