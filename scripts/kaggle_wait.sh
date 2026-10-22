#!/usr/bin/env bash
# Block until a Kaggle kernel finishes; print its final status. Usage: scripts/kaggle_wait.sh <user/kernel> [poll_seconds]
kernel="$1"; poll="${2:-60}"
while true; do
  s=$(uv run kaggle kernels status "$kernel" 2>&1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*|*FAIL*) echo "$(date +%T) $s"; break ;;
    *running*|*queued*|*RUNNING*|*QUEUED*) ;;
    *) echo "$(date +%T) unexpected: $s" ;;
  esac
  sleep "$poll"
done
