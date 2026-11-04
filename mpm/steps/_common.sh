# Shared setup for the step scripts. Source it, don't run it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$REPO/out}"
PY="${PY:-python}"

scene_name() {
  case "$1" in
    3|scene3|scene3_collision) echo scene3_collision ;;
    7|scene7|scene7_deformation) echo scene7_deformation ;;
    *) echo "unknown scene '$1' (use 3 or 7)" >&2; exit 1 ;;
  esac
}

# run render_blender.py with Blender if $BLENDER is set, else with the bpy pip module
blender_py() {
  if [ -n "${BLENDER:-}" ]; then "$BLENDER" -b -P "$REPO/mpm/render_blender.py" -- "$@"
  else "$PY" "$REPO/mpm/render_blender.py" "$@"; fi
}

peak_frame() {
  "$PY" -c "import numpy as np; print(int(np.argmax(np.load('$1/sim.npz')['max_strain'])))"
}

look_at() { echo; echo ">>> Look at: $*"; }
