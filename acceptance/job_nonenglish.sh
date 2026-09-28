#!/usr/bin/env bash
# Non-English acceptance: a real German video, fetched and built.
#
# The point is not to translate anything. It is that the evidence store stays in
# the source language while the rendered prose changes, and that identifiers,
# URLs, figures and provenance survive the crossing untouched.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VLM_API_KEY="${OPENROUTER_API_KEY:-}"
export SYNTH_API_KEY="$OPENROUTER_API_KEY"
VID="${VID:-5_jcvQ0-vhI}"
URL="https://www.youtube.com/watch?v=$VID"
LOG=acceptance/logs/E_non_english.log
: > "$LOG"

S=$(python3 -c 'import time;print(time.time())')
uv run scripts/yt_fetch.py "$URL" --lang de >>"$LOG" 2>&1
RC=$?
echo "--- fetch rc=$RC $(python3 -c "import time;print(f'{time.time()-$S:.0f}s')")" | tee -a "$LOG"

DIR="$(ls -dt "$HOME/Desktop/yt-reports"/*-"$VID" 2>/dev/null | head -1)"
[ -z "$DIR" ] && { echo "no artifact dir for $VID"; exit 90; }
echo "$DIR" > acceptance/logs/E_non_english.dir
echo "--- artifact: $DIR" | tee -a "$LOG"

S=$(python3 -c 'import time;print(time.time())')
uv run scripts/build_evidence.py "$DIR" --vlm-provider none >>"$LOG" 2>&1
echo "--- build rc=$? $(python3 -c "import time;print(f'{time.time()-$S:.0f}s')")" | tee -a "$LOG"

# The prose changes language; the evidence must not.
S=$(python3 -c 'import time;print(time.time())')
uv run scripts/build_evidence.py "$DIR" --vlm-provider none --synthesis-only \
    --output-language en >>"$LOG" 2>&1
echo "--- output-language en rc=$? $(python3 -c "import time;print(f'{time.time()-$S:.0f}s')")" | tee -a "$LOG"
cp "$DIR/REPORT.md" acceptance/logs/E_report_en.md
cp "$DIR/evidence.json" acceptance/logs/E_evidence_en.json

S=$(python3 -c 'import time;print(time.time())')
uv run scripts/build_evidence.py "$DIR" --vlm-provider none --synthesis-only \
    --output-language ja >>"$LOG" 2>&1
echo "--- output-language ja rc=$? $(python3 -c "import time;print(f'{time.time()-$S:.0f}s')")" | tee -a "$LOG"
cp "$DIR/REPORT.md" acceptance/logs/E_report_ja.md
