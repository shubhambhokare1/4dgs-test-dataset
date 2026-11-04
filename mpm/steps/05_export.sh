#!/usr/bin/env bash
# Step 5: write the 4DGS datasets for every camera subset (4/8/12/25/50).
#   bash mpm/steps/05_export.sh 7
#   bash mpm/steps/05_export.sh 7 --frame-step 2 --test-frame-step 2   # halve RAM in training
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1"); shift
D="$OUT/$S"
"$PY" "$REPO/mpm/export_4dgs.py" --scene-dir "$D" "$@"
# re-read one dataset the way 4DGaussians does and confirm the cameras round-trip
"$PY" -c "
import json, numpy as np
import glob; d = sorted(glob.glob('$D/4dgs/cams_*'))[-1]
t = json.load(open(d + '/transforms_train.json')); cams = json.load(open('$D/cameras.json'))['cameras']
S = np.eye(4); S[:3, 3] = -np.array(t['world_offset'])   # exported frame -> Blender world
worst = 0
for fr in t['frames']:
    M = np.linalg.inv(np.array(fr['transform_matrix'])); R = -M[:3,:3].T; R[:,0] = -R[:,0]; T = -M[:3,3]
    W = np.eye(4); W[:3,:3] = R.T; W[:3,3] = T
    worst = max(worst, np.abs(W - np.array(cams[fr['file_path'].split('/')[-2]]['w2c_cv']) @ S).max())
print(f'4DGaussians camera round-trip error: {worst:.1e} (should be below 1e-4)')
"
look_at "the image counts per subset above, and $D/4dgs/"
