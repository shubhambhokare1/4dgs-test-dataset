#!/usr/bin/env bash
# Simulate, render, verify and export one or more scenes.
#
#   bash mpm/run_pipeline.sh                     # both scenes, output in ./out
#   SCENES="scene7_deformation" OUT=/data/mpm bash mpm/run_pipeline.sh
#
# Env vars:
#   SCENES       scenes to build (default: scene3_collision scene7_deformation)
#   OUT          output root (default: ./out)
#   BLENDER      Blender executable; if unset, uses the `bpy` pip module via python
#   RENDER_ARGS  extra args for render_blender.py (e.g. "--samples 128 --res 800")
#   EXPORT_ARGS  extra args for export_4dgs.py (e.g. "--frame-step 2 --test-frame-step 2")
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/out}"
SCENES="${SCENES:-scene3_collision scene7_deformation}"

for s in $SCENES; do
  echo "=== $s: simulate"
  python "$REPO/mpm/sim_mpm.py" --scene "$s" --out "$OUT"
  echo "=== $s: render"
  if [ -n "${BLENDER:-}" ]; then
    "$BLENDER" -b -P "$REPO/mpm/render_blender.py" -- --scene-dir "$OUT/$s" ${RENDER_ARGS:-}
  else
    python "$REPO/mpm/render_blender.py" --scene-dir "$OUT/$s" ${RENDER_ARGS:-}
  fi
  echo "=== $s: check intrinsics/extrinsics against the images"
  python "$REPO/mpm/check_projection.py" --scene-dir "$OUT/$s"
  echo "=== $s: export 4DGS datasets"
  python "$REPO/mpm/export_4dgs.py" --scene-dir "$OUT/$s" ${EXPORT_ARGS:-}
done
