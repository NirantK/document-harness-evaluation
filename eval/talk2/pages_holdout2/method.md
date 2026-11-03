# Talk 2 second held-out test (agent frozen under the tag `talk2-revision2-long-lines-frozen`)

Fifteen questions used to test revision 2 of the structured agent. `pages_holdout` stopped being a clean test once its traces changed the harness. The rest of the method is in `../pages_multihop/method.md`.

## Questions
They continue the frozen selection order after `pages_holdout`, under the same rule and rejection reasons (`scripts/talk2/talk2/holdout_selection.py`). `question_selection.tsv` lists the 37 examined: 15 accepted, 22 rejected. Aliases were written before any run. The gold check matches `pages_holdout`: the benchmark's evidence quotes and a search for the gold terms in the PDF text. All fifteen come from XL-DocBench, so this test does not cover French text, contracts or CUAD.

The questions were selected and the corpus packaged before any run on them. The author read these questions and golds while selecting them, which happened before the long-line fix and the final revision 2 tag. Neither change concerns a specific question.

**Rule for this set:** no harness change follows from its results or traces. A later change needs a further fresh set.

## Corpus
`pages_holdout` (324 pages) plus, per question, its evidence pages and hard negatives from the same document: 481 pages in 93 documents.

## Configurations
`rag`, `markdown_hybrid`, `markdown_grep` and `markdown_grep_revision2`, all with `qwen3.5:9b` and the same settings. Revision 2 runs twice.
