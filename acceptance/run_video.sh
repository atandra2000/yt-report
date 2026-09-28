#!/usr/bin/env bash
# Run the whole pipeline over one video, timing every stage.
#
#   acceptance/run_video.sh <label> <video_id> <url> [extra build_evidence flags...]
#
# `yt_fetch.py` derives its own artifact directory from the title and the video
# id, so the directory is resolved by globbing on the id rather than assumed.
# Stage timings land in acceptance/logs/<label>.stages.json; combined output in
# acceptance/logs/<label>.log. The script's exit code is the last stage's, so a
# caller can tell whether REPORT.md was written without reading the log.
set -uo pipefail

LABEL="$1"; VID="$2"; URL="$3"; shift 3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/acceptance/logs"
ART="$HOME/Desktop/yt-reports"
mkdir -p "$LOGS"
LOG="$LOGS/$LABEL.log"
STAGES="$LOGS/$LABEL.stages.json"

: > "$LOG"
: > "$STAGES"
echo "=== $LABEL  $URL  $(date -u +%FT%TZ)" | tee -a "$LOG"

record_stage() {
  python3 - "$STAGES" "$1" "$2" "$3" <<'PY'
import json, os, sys
path, name, secs, rc = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
# The file is truncated at start-up, so "exists" is not "parseable" — an empty
# stages file is the normal state before the first stage lands.
data = {}
if os.path.exists(path) and os.path.getsize(path):
    data = json.load(open(path))
data[name] = {"seconds": round(secs, 2), "exit": rc}
json.dump(data, open(path, "w"), indent=2)
PY
}

run_stage() {
  local name="$1"; shift
  local start end rc
  start=$(python3 -c 'import time; print(time.time())')
  "$@" >>"$LOG" 2>&1
  rc=$?
  end=$(python3 -c 'import time; print(time.time())')
  record_stage "$name" "$(python3 -c "print($end-$start)")" "$rc"
  echo "--- $name exit=$rc $(python3 -c "print(f'{$end-$start:.1f}s')")" | tee -a "$LOG"
  return $rc
}

run_stage fetch uv run "$ROOT/scripts/yt_fetch.py" "$URL"
FETCH_RC=$?

DIR="$(ls -dt "$ART"/*-"$VID" 2>/dev/null | head -1)"
if [ -z "$DIR" ]; then
  echo "=== $LABEL FAILED: no artifact dir for $VID" | tee -a "$LOG"
  exit 90
fi
echo "$DIR" > "$LOGS/$LABEL.dir"
echo "--- artifact: $DIR" | tee -a "$LOG"

run_stage build uv run "$ROOT/scripts/build_evidence.py" "$DIR" "$@"
BUILD_RC=$?

python3 - "$STAGES" "$FETCH_RC" "$BUILD_RC" "$DIR" <<'PY'
import json, sys
path, fetch_rc, build_rc, dir_ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = json.load(open(path))
d["_exit"] = {"fetch": fetch_rc, "build": build_rc}
d["_artifact_dir"] = dir_
json.dump(d, open(path, "w"), indent=2)
PY
echo "=== $LABEL done fetch=$FETCH_RC build=$BUILD_RC" | tee -a "$LOG"
exit $BUILD_RC
