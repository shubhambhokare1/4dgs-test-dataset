#!/usr/bin/env bash
# Step 1: run the MPM simulation for one scene and make a preview.
#   bash mpm/steps/01_simulate.sh 7                 # extra args go to sim_mpm.py
#   bash mpm/steps/01_simulate.sh 3 --E 1e5 2e5 4e5 --overwrite
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1"); shift
"$PY" "$REPO/mpm/sim_mpm.py" --scene "$S" --out "$OUT" "$@"
"$PY" "$REPO/mpm/preview.py" sim --scene-dir "$OUT/$S"
look_at "$OUT/$S/preview/sim.gif     (objects move, collide, rebound; nothing explodes or leaves the box)
         $OUT/$S/preview/strain.png  (one clear spike at impact)"
