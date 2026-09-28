#!/usr/bin/env bash
set -u
export VLM_API_KEY="$OPENROUTER_API_KEY"
export SYNTH_API_KEY="$OPENROUTER_API_KEY"
export VLM_PROVIDER=openrouter VLM_MODEL=openai/gpt-4o-mini
R=acceptance/run_video.sh
F="--vlm-provider openrouter --vlm-model openai/gpt-4o-mini --max-vlm-frames 8 --synth-provider openrouter --synth-model openai/gpt-4o-mini"
bash $R C_screen_tech    4TvH-OZhwxI https://www.youtube.com/watch?v=4TvH-OZhwxI $F
bash $R D_visual_lecture aircAruvnKk https://www.youtube.com/watch?v=aircAruvnKk $F
