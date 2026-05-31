#!/usr/bin/env bash
# loop.sh — gildnn autonomous loop (systemd user service)
#
# Launches opencode, waits for it to exit, relaunches.
# Backoff on fast crashes, normal 10s pause otherwise.
#
# USAGE (via systemd):
#   systemctl --user start gildnn-loop
#   systemctl --user stop gildnn-loop
#   journalctl --user -u gildnn-loop -f

set -euo pipefail
cd "$(dirname "$0")"

# Project archived 2026-05-31. Stop the loop.
echo "Project timebox concluded 2026-05-31. Loop exiting."
echo "To restart: edit loop.sh to remove this guard, then: systemctl --user start gildnn-loop"
exit 0

SESSION_ID="ses_189bc3a17ffe1O7nyZLLcIAQId"
CRASH_COUNT=0
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    PROMPT=$(cat loop-prompt.md)
    START_TIME=$(date +%s)

    echo "[$(date '+%H:%M:%S')] iteration $ITERATION — launching OpenCode"

    set +e
    opencode run \
        --agent arrange \
        --model github-copilot-max/claude-opus-4.6 \
        --variant high \
        --session "$SESSION_ID" \
        "$PROMPT"
    EXIT_CODE=$?
    set -e

    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "[$(date '+%H:%M:%S')] exited (code $EXIT_CODE, ${ELAPSED}s)"

    if [ "$ELAPSED" -lt 30 ]; then
        CRASH_COUNT=$((CRASH_COUNT + 1))
        BACKOFF=$((60 * CRASH_COUNT))
        [ "$BACKOFF" -gt 300 ] && BACKOFF=300
        echo "Fast exit #$CRASH_COUNT — waiting ${BACKOFF}s"
        sleep "$BACKOFF"
    else
        CRASH_COUNT=0
        echo "Waiting 10s..."
        sleep 10
    fi
done
