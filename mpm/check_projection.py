"""
Check that the exported intrinsics/extrinsics match the rendered images.

Every object's mesh (deformed, from sim.npz) is projected with K and w2c_cv
from cameras.json and rasterised into a silhouette. The object's pixels are
found in the image by colour. If the cameras are right, almost every object
pixel lies inside the projected silhouette (occlusion can only remove pixels,
so it does not hurt this score).

    python mpm/check_projection.py --scene-dir out/scene3_collision
    python mpm/check_projection.py --scene-dir out/scene7_deformation --overlays 4

Exits non-zero if any checked image scores below --min-score.
"""

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import project  # noqa: E402

BOX_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def colour_mask(img, rgb, hue_tol=0.06, min_sat=0.35):
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    th = colorsys.rgb_to_hsv(*rgb)[0]
    dh = np.minimum(np.abs(h - th), 1 - np.abs(h - th))
    return (sat > min_sat) & (dh < hue_tol)


def silhouette(polys2d, size):
    im = Image.new("L", size, 0)
    dr = ImageDraw.Draw(im)
    for p in polys2d:
        dr.polygon([tuple(q) for q in p], fill=255)
    im = im.filter(ImageFilter.MaxFilter(5))  # allow 2 px of anti-aliasing
    return np.asarray(im) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--min-score", type=float, default=0.98)
    ap.add_argument("--min-pixels", type=int, default=30)
    ap.add_argument("--overlays", type=int, default=3, help="overlay images to save per camera")
    args = ap.parse_args()

    sdir = Path(args.scene_dir)
    cams = json.load(open(sdir / "cameras.json"))["cameras"]
    meta = json.load(open(sdir / "sim.json"))["scene"]
    sim = np.load(sdir / "sim.npz")

    objs = []
    for o in meta["objects"]:
        objs.append((o["name"], o["color"], sim[f"verts_{o['name']}"], sim[f"faces_{o['name']}"]))
    for s in meta["static_objects"]:
        from geometry import box_corners
        v = box_corners(s["center"], s["half"])
        objs.append((s["name"], s["color"], np.broadcast_to(v, (len(sim["times"]),) + v.shape),
                     np.array(BOX_FACES)))

    odir = sdir / "projection_check"
    odir.mkdir(exist_ok=True)
    rows, worst = [], 1.0
    for cname, c in cams.items():
        files = sorted((sdir / "images" / cname).glob("*.png"))
        K, w2c = np.array(c["K"]), np.array(c["w2c_cv"])
        for n_img, fpath in enumerate(files):
            f = int(fpath.stem)
            img = Image.open(fpath)
            over = img.convert("RGB").copy() if n_img < args.overlays else None
            for name, rgb, verts, faces in objs:
                uv, z = project(verts[f], K, w2c)
                if np.any(z <= 0):
                    continue
                sil = silhouette(uv[faces], img.size)
                mask = colour_mask(img, rgb)
                if mask.sum() < args.min_pixels:
                    continue
                score = float((mask & sil).sum() / mask.sum())
                worst = min(worst, score)
                rows.append(dict(camera=cname, frame=f, object=name, pixels=int(mask.sum()), score=score))
                if over is not None:
                    dr = ImageDraw.Draw(over)
                    for p in uv[:: max(1, len(uv) // 400)]:
                        dr.point(tuple(p), fill=(0, 0, 0))
            if over is not None:
                over.save(odir / f"{cname}_{f:05d}.png")

    if not rows:
        print("no object pixels found to check")
        return 1
    scores = np.array([r["score"] for r in rows])
    bad = [r for r in rows if r["score"] < args.min_score]
    summary = dict(checked=len(rows), mean=float(scores.mean()), min=float(scores.min()),
                   below_threshold=len(bad), threshold=args.min_score)
    json.dump(dict(summary=summary, rows=rows), open(odir / "report.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in bad[:10]:
        print("  low:", r)
    print(f"overlays + report in {odir}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
