#!/usr/bin/env bash
# Upload a built submission to Kaggle.
#
#   ./submit.sh <tag> [message]
#   ./submit.sh --status          # past submissions and their public scores
#
# Uploads outputs/submissions/<tag>/submission.csv. Without an explicit message
# it reuses the one recorded when `src/submission.py` built the submission.
#
# This only uploads — it never retrains, so the checked-out branch does not
# matter. Each run spends one of the 10 daily submissions.
set -euo pipefail

COMPETITION="playground-series-s6e7"

# The CLI usually lives in the project venv rather than on PATH.
if command -v kaggle >/dev/null 2>&1; then
    kg() { kaggle "$@"; }
else
    kg() { (cd "$(dirname "$0")" && uv run kaggle "$@"); }
fi

if [ "${1:-}" = "--status" ]; then
    # Read back scores; costs nothing against the daily quota.
    kg competitions submissions -c "$COMPETITION"
    exit 0
fi

if [ $# -lt 1 ]; then
    echo "usage: $0 <tag> [message]" >&2
    echo "       $0 --status        # show past submissions and their scores" >&2
    echo >&2
    echo "available tags:" >&2
    ls -1 "$(dirname "$0")/outputs/submissions" 2>/dev/null | sed 's/^/  /' >&2 \
        || echo "  (none built yet)" >&2
    exit 1
fi

tag="$1"
dir="$(dirname "$0")/outputs/submissions/${tag}"
csv="${dir}/submission.csv"

if [ ! -f "$csv" ]; then
    echo "no submission at ${csv}" >&2
    echo "build one first: uv run python src/submission.py --tag ${tag} --message \"...\"" >&2
    exit 1
fi

if [ $# -ge 2 ]; then
    message="$2"
elif [ -f "${dir}/message.txt" ]; then
    message="$(cat "${dir}/message.txt")"
else
    echo "no message given and no ${dir}/message.txt to fall back on" >&2
    echo "usage: $0 ${tag} \"your message\"" >&2
    exit 1
fi

echo "submitting ${csv}"
echo "message: ${message}"
kg competitions submit -c "$COMPETITION" -f "$csv" -m "$message"

echo
echo "scoring takes a moment; check with: $0 --status"
