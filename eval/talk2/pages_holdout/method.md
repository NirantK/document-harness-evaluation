# Talk 2 held-out test (structured grep agent frozen under the tag `talk2-structured-frozen`)

Ten questions no agent was tuned on, used to test whether `markdown_grep_structured` generalises beyond the 20 questions
whose failures shaped it. The rest of the method is in `../pages_multihop/method.md`.

## Questions
They continue the frozen multi-hop selection order from position 19, under the same rule and the same rejection reasons
(`scripts/talk2/talk2/holdout_selection.py`). The first ten that passed were taken. `question_selection.tsv` lists the 28
examined: 10 accepted, 18 rejected. Aliases were written before any run. The gold check used the benchmark's evidence
quotes and a search for the gold terms in the PDF text, which is lighter than the first ten's check; only the first
question set had every quote located by hand.

The questions were selected and the corpus packaged before any run on them. The agent was committed and tagged before
its first run here. Its author read these questions and golds while selecting them, after the agent's design was set
and before its last fixes, which concern tool arguments only (a `path` alias for `file`, and malformed tool calls).

## Corpus
`pages_multihop` (222 pages) plus, per question, its evidence pages and 8 hard negatives from the same document: 324 pages
in 78 documents. The ten documents use the same negative rule as the multi-hop ten.

## Configurations
`rag`, `markdown_grep` and `markdown_grep_structured`, all with `qwen3.5:9b` and the same settings.
