#!/bin/zsh
# Copy the public subset of this work into the working copy of NirantK/document-harness-evaluation.
# Excluded by design: run logs, review sheets, questions.jsonl (they hold XL-DocBench text), corpora and previews.
# Usage: scripts/talk2/synchronise_public_repository.sh /path/to/working/copy
set -eu
REPOSITORY=${0:A:h:h:h}
TARGET=${1:?working copy path}
mkdir -p $TARGET/scripts/talk2 $TARGET/docs/files $TARGET/eval/talk2
rsync -a --delete --exclude '.venv' --exclude '__pycache__' --exclude 'output' --exclude 'run_evaluation_queue.sh' \
  --exclude 'run_structured_evaluation.sh' $REPOSITORY/scripts/talk2/ $TARGET/scripts/talk2/
for corpus in pages_validation pages_multihop pages_holdout pages_holdout2 pages_unseen scanned_demonstration; do
  [[ -d $REPOSITORY/eval/talk2/$corpus ]] || continue
  mkdir -p $TARGET/eval/talk2/$corpus
  for name in method.md results.md results_notes.md question_selection.tsv page_selection.tsv corpus_manifest.tsv; do
    [[ -f $REPOSITORY/eval/talk2/$corpus/$name ]] && cp $REPOSITORY/eval/talk2/$corpus/$name $TARGET/eval/talk2/$corpus/
  done
done
[[ -f $REPOSITORY/eval/talk2/method.md ]] && cp $REPOSITORY/eval/talk2/method.md $TARGET/eval/talk2/
for measurement in vector_search conversion_comparison; do
  [[ -d $REPOSITORY/eval/talk2/$measurement ]] || continue
  mkdir -p $TARGET/eval/talk2/$measurement
  find $REPOSITORY/eval/talk2/$measurement -maxdepth 1 -type f \( -name '*.md' -o -name '*.tsv' \) -exec cp {} $TARGET/eval/talk2/$measurement/ \;
done
# GitHub Pages serves these PDFs inline, so a link ending in #page=N opens at that page.
rsync -a --delete --include '*/' --include '*.pdf' --include 'document_index.tsv' --exclude '*' \
  $REPOSITORY/data/hf_talk2_harness_evaluation/public/files/ $TARGET/docs/files/
# MIL-STD-810H and the technical order are US Department of Defense documents approved for public release.
mkdir -p $TARGET/docs/files/us-dod
cp $REPOSITORY/data/talk2_corpus_scanned_demonstration/pdf/us-dod/*.pdf $TARGET/docs/files/us-dod/
