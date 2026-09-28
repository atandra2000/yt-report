#!/usr/bin/env bash
set -u
export VLM_API_KEY="$OPENROUTER_API_KEY"
export SYNTH_API_KEY="$OPENROUTER_API_KEY"
export VLM_PROVIDER=openrouter VLM_MODEL=openai/gpt-4o-mini
R=acceptance/run_video.sh
F="--vlm-provider openrouter --vlm-model openai/gpt-4o-mini --max-vlm-frames 8 --synth-provider openrouter --synth-model openai/gpt-4o-mini"
bash $R A_talking_head   EOdXR6lU5ZA https://www.youtube.com/watch?v=EOdXR6lU5ZA $F
bash $R B_ai_news       h5QwrHhW28Q https://www.youtube.com/watch?v=h5QwrHhW28Q $F
