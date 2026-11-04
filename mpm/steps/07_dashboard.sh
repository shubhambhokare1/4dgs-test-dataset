#!/usr/bin/env bash
# Step 7: build the dashboard from whatever has been trained so far.
#   FOURDGS_ROOT=~/4DGaussians bash mpm/steps/07_dashboard.sh
source "$(dirname "$0")/_common.sh"
DIRS=()
for S in scene3_collision scene7_deformation; do [ -f "$OUT/$S/rig.json" ] && DIRS+=("$OUT/$S"); done
"$PY" "$REPO/mpm/build_dashboard.py" --scene-dirs "${DIRS[@]}" \
  ${FOURDGS_ROOT:+--results-root "$FOURDGS_ROOT/output/${EXP_ROOT:-mpm_study}"} --out "$REPO/dashboard"
look_at "$REPO/dashboard/index.html  (open in a browser; copy the whole dashboard/ folder if viewing elsewhere)"
