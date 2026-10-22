#!/usr/bin/env bash
# Push queued Kaggle jobs whenever a GPU session slot is free (Kaggle allows 2 concurrent).
# Usage: scripts/kaggle_queue.sh "<kaggle_job.py args>" ["<args>" ...]
for job in "$@"; do
  until out=$(uv run python scripts/kaggle_job.py $job --push 2>&1) && echo "$out" | grep -q "successfully pushed"; do
    echo "$out" | grep -q "session count" || { echo "$(date +%T) push failed: $out"; exit 1; }
    sleep 120
  done
  echo "$(date +%T) pushed: $job"
done
