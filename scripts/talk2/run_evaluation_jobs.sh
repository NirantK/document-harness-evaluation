#!/bin/zsh
# Process .state/talk2_evaluation_jobs.tsv one line at a time; lines may be appended while it runs.
# Line format: corpus <TAB> configuration <TAB> repeat <TAB> model. A repeat with a complete log is skipped.
# Log: .state/talk2_evaluation_jobs.log (appended)
set -u
REPOSITORY=${0:A:h:h:h}
JOBS=$REPOSITORY/.state/talk2_evaluation_jobs.tsv
cd $REPOSITORY/scripts/talk2
position=$(( $(wc -l < $REPOSITORY/.state/talk2_evaluation_jobs.done 2>/dev/null || echo 0) + 1 ))
while true; do
  line=$(sed -n "${position}p" $JOBS)
  if [[ -z "$line" ]]; then
    sleep 30
    continue
  fi
  corpus=${line%%$'\t'*}; rest=${line#*$'\t'}
  configuration=${rest%%$'\t'*}; rest=${rest#*$'\t'}
  repeat=${rest%%$'\t'*}; model=${rest#*$'\t'}
  echo "JOB START $position $corpus $configuration r$repeat $model $(date +%H:%M)"
  TALK2_CORPUS=$corpus caffeinate -i uv run talk2 run --configuration $configuration --repeats 1 --first-repeat $repeat \
    --model $model --resume 2>&1 | grep --line-buffered -E "CORRECT|wrong|wrote|already complete|RESOURCE|OLLAMA RETRY|Error|Traceback"
  echo "$line" >> $REPOSITORY/.state/talk2_evaluation_jobs.done
  position=$(( position + 1 ))
done
