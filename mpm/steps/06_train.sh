#!/usr/bin/env bash
# Step 6: train, render and score 4DGaussians on ONE camera subset, then show
# GT vs prediction. Start with a small subset to check everything works.
#   FOURDGS_ROOT=~/4DGaussians bash mpm/steps/06_train.sh 7 04
#   FOURDGS_ROOT=~/4DGaussians bash mpm/steps/06_train.sh 7 50
source "$(dirname "$0")/_common.sh"
: "${FOURDGS_ROOT:?set FOURDGS_ROOT to your 4DGaussians checkout}"
S=$(scene_name "$1"); N=$(printf %02d "$((10#$2))"); shift 2
EXP="${EXP_ROOT:-mpm_study}/$S/cams_$N"
CFG="$REPO/arguments/mpm_$S.py"
cd "$FOURDGS_ROOT"
if [ -f "output/$EXP/results.json" ] && [ -z "${RETRAIN:-}" ]; then
  echo "$EXP already trained (set RETRAIN=1 to train again); making the preview only"
  "$PY" "$REPO/mpm/preview.py" compare --scene-dir "$OUT/$S" --model-path "$FOURDGS_ROOT/output/$EXP"
  exit 0
fi
"$PY" train.py -s "$OUT/$S/4dgs/cams_$N" --port "${PORT:-$((6200 + 10#$N))}" --expname "$EXP" --configs "$CFG" "$@"
"$PY" render.py --model_path "output/$EXP" --skip_train --skip_video --configs "$CFG"
"$PY" metrics.py --model_path "output/$EXP"
"$PY" "$REPO/mpm/preview.py" compare --scene-dir "$OUT/$S" --model-path "$FOURDGS_ROOT/output/$EXP"
look_at "$OUT/$S/preview/compare_cams_$N.jpg  (GT next to prediction on held-out cameras)"
