# pages_unseen: the held-out question set

20 questions, 531 pages, 98 documents. Selected 3 Nov 2026 under the same rule as `pages_holdout2`: the first accepted
XL-DocBench reference-chain questions by position, then the same evidence-page and hard-negative mining.

- It holds the 15 questions of `pages_holdout2`, on which no harness change was ever based, and 5 new questions from
  positions 84 to 94 of the eligible pool. That pool has 95 positions, so it is now exhausted.
- `question_selection.tsv` records each accepted and rejected candidate with its reason.
- Rule: no harness or prompt change follows from results on this set. A later change needs a further fresh set.

## What was seen before the final runs

- Revision 2 was run on the 15 carried questions, and its scores are reported. No harness change came from those traces.
- While designing revision 3 the turn counts per question on the 15 were printed alongside the tuning sets. No prompt
  text or question content was read from them. Revision 3's prompt comes from the tuning traces in `pages_multihop` and
  `pages_holdout`.
- The merge in `markdown_merging.py` came from the conversion loss measurement over the corpus text against the PDF text
  layer, which uses no questions, and from the request to reduce conversion loss. The measurement ran on the `pages_holdout2`
  pages, so those documents were seen; the questions and their answers were not used.

## Question not in this set

No scored question rests on a page with an empty text layer. `scanned_demonstration` runs one such question separately.
