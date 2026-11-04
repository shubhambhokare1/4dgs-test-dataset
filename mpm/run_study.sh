#!/usr/bin/env bash
# Train, render and score 4DGaussians on every camera subset, then build the dashboard.
#
#   FOURDGS_ROOT=~/4DGaussians bash mpm/run_study.sh
#
# Env vars:
#   FOURDGS_ROOT  4DGaussians checkout (https://github.com/hustvl/4DGaussians), required
#   OUT           dataset root from run_pipeline.sh (default: ./out)
#   SCENES        default: scene3_collision scene7_deformation
#   SUBSETS       default: 04 08 12 25 50
#   EXP_ROOT      experiment folder under $FOURDGS_ROOT/output (default: mpm_study)
# Finished runs (results.json present) are skipped, so the script can be re-run.
# To use several GPUs, run one copy per GPU with different SCENES/SUBSETS and
# CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${FOURDGS_ROOT:?set FOURDGS_ROOT to your 4DGaussians checkout}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/out}"
SCENES="${SCENES:-scene3_collision scene7_deformation}"
SUBSETS="${SUBSETS:-04 08 12 25 50}"
EXP_ROOT="${EXP_ROOT:-mpm_study}"
PORT="${PORT:-6200}"

cd "$FOURDGS_ROOT"
for s in $SCENES; do
  cfg="$REPO/arguments/mpm_$s.py"
  for n in $SUBSETS; do
    data="$OUT/$s/4dgs/cams_$n"
    exp="$EXP_ROOT/$s/cams_$n"
    if [ -f "output/$exp/results.json" ]; then echo "skip $exp (done)"; continue; fi
    echo "=== $s, $n cameras"
    python train.py -s "$data" --port "$PORT" --expname "$exp" --configs "$cfg"
    python render.py --model_path "output/$exp" --skip_train --skip_video --configs "$cfg"
    python metrics.py --model_path "output/$exp"
    PORT=$((PORT + 1))
  done
done

python "$REPO/mpm/build_dashboard.py" \
  --scene-dirs $(for s in $SCENES; do echo "$OUT/$s"; done) \
  --results-root "$FOURDGS_ROOT/output/$EXP_ROOT" --out "$REPO/dashboard"
echo "Dashboard: $REPO/dashboard/index.html"
