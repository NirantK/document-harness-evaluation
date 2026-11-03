#!/bin/zsh
# Run jobs against the hosted endpoint, several at once. Line format: corpus configuration repeat. Repeat N uses seed 41+N.
# Usage: scripts/talk2/run_hosted_evaluation.sh jobs.tsv [concurrent_jobs] [questions_at_once]
set -u
cd ${0:A:h}
export TALK2_HOSTED_PROVIDER=${TALK2_HOSTED_PROVIDER:-Darkbloom}
xargs -P ${2:-8} -L 1 zsh -c 'TALK2_SEED=$((41 + $2)) TALK2_CORPUS=$0 uv run talk2 run --configuration $1 --repeats 1 --first-repeat $2 --model qwen/qwen3.5-9b --backend openrouter --workers '${3:-10}' --resume 2>&1 | grep -E "wrote|already complete|Traceback" | cut -c1-200' < $1
