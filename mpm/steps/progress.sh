#!/usr/bin/env bash
# Live render progress for a scene: images done, percent, time left.
# Ctrl-C stops the display only; renders keep running.
#   bash mpm/steps/progress.sh 7
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1")
D="$OUT/$S/images"
TOTAL=$("$PY" -c "import json;r=json.load(open('$OUT/$S/rig.json'));print(len(r['cameras'])*json.load(open('$OUT/$S/sim.json'))['solver']['n_frames'])")
count() { find "$D" -name '*.png' 2>/dev/null | wc -l; }
start=$(count); t0=$(date +%s)
while :; do
  n=$(count); dt=$(( $(date +%s) - t0 ))
  jobs=$(pgrep -fc "render_blender.py.*$S" || true)
  if [ "$n" -gt "$start" ]; then
    left=$(( (TOTAL - n) * dt / (n - start) / 60 )); eta="~${left} min left"
  else eta="measuring speed..."; fi
  printf "\r%s  %5d / %d images (%3d%%)  %-20s  render jobs running: %s   " \
    "$(date +%H:%M:%S)" "$n" "$TOTAL" $(( n * 100 / TOTAL )) "$eta" "$jobs"
  if [ "$n" -ge "$TOTAL" ]; then echo; echo "All $TOTAL images rendered."; break; fi
  if [ "$jobs" = 0 ]; then echo; echo "No render jobs running (stopped or crashed). Last log lines:"; tail -n 3 "$REPO"/render*job*.log 2>/dev/null; break; fi
  sleep 15
done
