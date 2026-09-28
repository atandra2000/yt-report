#!/usr/bin/env bash
# Visual pass + synthesis + critic over the four real videos.
#
# Re-runs only the second half of the pipeline: the artifacts are already on disk
# from the fetch stage, so nothing is downloaded and no transcript is re-fetched.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VLM_API_KEY="${OPENROUTER_API_KEY:?}"
export SYNTH_API_KEY="$OPENROUTER_API_KEY"
M="${MODEL:-stealth/space-bunny-alpha}"
FRAMES="${FRAMES:-6}"

while read -r L SLUG; do
  [ -n "$L" ] || continue
  D="$HOME/Desktop/yt-reports/$SLUG"
  echo "=== $L $(date -u +%FT%TZ)"
  S=$(python3 -c 'import time;print(time.time())')
  uv run scripts/build_evidence.py "$D" \
      --vlm-provider openrouter --vlm-model "$M" --max-vlm-frames "$FRAMES" \
      --synth-provider openrouter --synth-model "$M" \
      > "acceptance/logs/${L}.model.log" 2>&1
  echo "--- $L rc=$? $(python3 -c "import time;print(f'{time.time()-$S:.0f}s')")"
done <<'VIDEOS'
A_talking_head this-new-jev-claude-os-just-changed-every-ai-workflow-EOdXR6lU5ZA
B_ai_news google-s-project-suncatcher-is-sending-ai-to-space-sonnet-5-5-500-chat-h5QwrHhW28Q
C_screen_tech ai-agents-explained-how-to-create-and-use-ai-agents-in-2026-4TvH-OZhwxI
D_visual_lecture but-what-is-a-neural-network-deep-learning-chapter-1-aircAruvnKk
VIDEOS
