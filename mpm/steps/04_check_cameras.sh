#!/usr/bin/env bash
# Step 4: verify intrinsics/extrinsics by projecting the simulated meshes into
# the rendered images.
#   bash mpm/steps/04_check_cameras.sh 7
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1"); shift
D="$OUT/$S"
"$PY" "$REPO/mpm/check_projection.py" --scene-dir "$D" "$@" && ok=1 || ok=0
look_at "$D/projection_check/overview.jpg  (black dots must sit on the objects)
         mean/min score above should be ~1.0"
[ "$ok" = 1 ] || { echo "Some images scored below the threshold, see report.json"; exit 1; }
