# pages_validation: the validation questions on the shared corpus

20 questions, 531 pages, 98 documents. Validation and test differ only in their questions.

- The questions are those of `pages_multihop`. Their 222 pages all sit inside the 531 pages of `pages_unseen`, so
  `pages_unseen` is the shared corpus, and the test runs there stand unchanged.
- `scripts/talk2/assemble_validation_corpus.py` builds this folder. It links to the corpus files and indexes of
  `pages_unseen` and renumbers each question's evidence pages to that layout.
- Harness decisions were made on the earlier 222-page validation corpus, whose logs stay in `pages_multihop/runs`.
  Validation scores reported from this folder are on the shared 531-page corpus.
- Display names: Validation is this folder, and Test is `pages_unseen`.
