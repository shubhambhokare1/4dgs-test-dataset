#!/usr/bin/env bash
# Step 3: full render, every camera x every frame. Resumable: re-run the same
# command after a crash and finished images are skipped.
#   bash mpm/steps/03_render.sh 7
#   bash mpm/steps/03_render.sh 7 --samples 32
#   bash mpm/steps/03_render.sh 7 --cams train --gpu 0   # split across GPUs
#   bash mpm/steps/03_render.sh 7 --cams test  --gpu 2   # (--gpu = nvidia-smi index)
# Use the same render settings for every part of a scene.
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1"); shift
D="$OUT/$S"
blender_py --scene-dir "$D" "$@" 2>&1 | grep -E --line-buffered "^frame|TIMING|verified|Cycles on|WARNING|Error|Traceback" || true
PK=$(peak_frame "$D")
"$PY" "$REPO/mpm/preview.py" cams --scene-dir "$D" --frame "$PK"
"$PY" "$REPO/mpm/preview.py" camgif --scene-dir "$D" --cam cam00 --every 2
"$PY" "$REPO/mpm/preview.py" camgif --scene-dir "$D" --cam test00 --every 2
look_at "$D/preview/cams_frame$(printf %05d "$PK").jpg  (all cameras at impact; pink = missing image)
         $D/preview/cam00.gif, test00.gif           (smooth motion, no flicker)"
