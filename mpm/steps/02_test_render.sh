#!/usr/bin/env bash
# Step 2: render a handful of images at full quality to check the look and
# measure speed before committing to the full render. Writes to images_test/,
# so the real render is unaffected.
#   bash mpm/steps/02_test_render.sh 7
#   bash mpm/steps/02_test_render.sh 7 --samples 32      # try other settings
#   bash mpm/steps/02_test_render.sh 7 --engine EEVEE
source "$(dirname "$0")/_common.sh"
S=$(scene_name "$1"); shift
D="$OUT/$S"
PK=$(peak_frame "$D")
for F in 0 "$PK"; do
  blender_py --scene-dir "$D" --images-dir images_test --overwrite \
    --cams cam00,cam04,cam08,cam10,cam30,test00 --frames "$F:$((F + 1))" "$@" 2>&1 \
    | grep -E -i "TIMING|verified|Cycles on|WARNING|error|Traceback|usage|only [0-9]+ .* devices" || true
done
"$PY" -c "
import sys; sys.path.insert(0, '$REPO/mpm')
from pathlib import Path
from PIL import Image
from preview import sheet, label
d = Path('$D') / 'images_test'
tiles = []
for f in (0, $PK):
    for c in ['cam00','cam04','cam08','cam10','cam30','test00']:
        t = Image.open(d / c / f'{f:05d}.png').convert('RGB'); t.thumbnail((300, 300))
        tiles.append(label(t, f'{c} frame {f}', size=12))
(Path('$D') / 'preview').mkdir(exist_ok=True)
sheet(tiles, 6).save(Path('$D') / 'preview' / 'render_test.jpg', quality=92)
print('wrote', Path('$D') / 'preview' / 'render_test.jpg')
"
look_at "$D/preview/render_test.jpg  (top row frame 0, bottom row peak impact frame $PK)
         full-size PNGs in $D/images_test/   and the TIMING line above for the full-render estimate"
