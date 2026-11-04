"""
Quick-look images for checking each pipeline step by eye. Needs only numpy + Pillow.

    python mpm/preview.py sim      --scene-dir out/scene7_deformation
    python mpm/preview.py cams     --scene-dir out/scene7_deformation --frame 45
    python mpm/preview.py camgif   --scene-dir out/scene7_deformation --cam cam10
    python mpm/preview.py compare  --scene-dir out/scene7_deformation --model-path ~/4DGaussians/output/mpm_study/scene7_deformation/cams_12

Everything is written to <scene-dir>/preview/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def label(im, text, xy=(6, 4), size=14, fill=(0, 0, 0)):
    d = ImageDraw.Draw(im)
    f = font(size)
    box = d.textbbox(xy, text, font=f)
    d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 255, 255))
    d.text(xy, text, font=f, fill=fill)
    return im


def sheet(tiles, cols, pad=4, bg=(235, 235, 235)):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), bg)
    for i, t in enumerate(tiles):
        out.paste(t, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return out


def peak_frame(sdir):
    return int(np.argmax(np.load(sdir / "sim.npz")["max_strain"]))


# ---------------------------------------------------------------------------- sim
def cmd_sim(a):
    sdir = Path(a.scene_dir)
    sim = np.load(sdir / "sim.npz")
    meta = json.load(open(sdir / "sim.json"))
    sc = meta["scene"]
    lo, hi = np.array(sc["region_lo"]), np.array(sc["region_hi"])
    P, obj = sim["particles"], sim["particle_obj"]
    cols = np.array([[int(255 * c) for c in o["color"]] for o in sc["objects"]])[obj]
    strain, fps = sim["max_strain"], sc["fps"]
    S = 360

    def panel(pts, ax0, ax1, title, static_boxes):
        span = max(hi[ax0] - lo[ax0], hi[ax1] - lo[ax1])
        sc_ = (S - 20) / span
        im = Image.new("RGB", (S, S), (255, 255, 255))
        d = ImageDraw.Draw(im)
        to_px = lambda u, v: (10 + (u - lo[ax0]) * sc_, S - 10 - (v - lo[ax1]) * sc_)
        d.rectangle((*to_px(lo[ax0], hi[ax1]), *to_px(hi[ax0], lo[ax1])), outline=(200, 200, 200))
        for b in static_boxes:
            c, h = np.array(b["center"]), np.array(b["half"])
            d.rectangle((*to_px(c[ax0] - h[ax0], c[ax1] + h[ax1]), *to_px(c[ax0] + h[ax0], c[ax1] - h[ax1])),
                        fill=tuple(int(255 * x) for x in b["color"]))
        for p, col in zip(pts, cols):
            x, y = to_px(p[ax0], p[ax1])
            d.point((x, y), fill=tuple(col))
        return label(im, title, size=13)

    frames = []
    for f in range(0, len(P), a.every):
        top = panel(P[f], 0, 1, "top (x-y)", sc["static_objects"])
        side = panel(P[f], 0, 2, "side (x-z)", sc["static_objects"])
        fr = Image.new("RGB", (2 * S + 12, S + 34), (255, 255, 255))
        fr.paste(top, (4, 30))
        fr.paste(side, (S + 8, 30))
        label(fr, f"{sc['name']}  frame {f}  t={f / fps:.2f}s  max strain {strain[f]:.3f}", (8, 6))
        frames.append(fr)
    out = sdir / "preview"
    out.mkdir(exist_ok=True)
    frames[0].save(out / "sim.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 * a.every / fps), loop=0)

    # strain curve + object centres over time
    W, H = 720, 260
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    m = max(float(strain.max()), 1e-6)
    pts = [(40 + i * (W - 60) / (len(strain) - 1), H - 30 - s / m * (H - 60)) for i, s in enumerate(strain)]
    d.line([(40, H - 30), (W - 20, H - 30)], fill=(180, 180, 180))
    d.line(pts, fill=(42, 120, 214), width=2)
    pk = int(np.argmax(strain))
    d.line([(pts[pk][0], 20), (pts[pk][0], H - 30)], fill=(235, 104, 52))
    label(im, f"max |strain| per frame (peak {m:.3f} at frame {pk}, t={pk / fps:.2f}s)", (8, 4))
    label(im, f"0 .. {len(strain) - 1} frames", (W - 140, H - 24), size=12)
    im.save(out / "strain.png")

    # summary
    print(f"{sc['name']}: {len(P)} frames, peak strain {m:.3f} at frame {pk} (t={pk / fps:.2f}s)")
    for o in sc["objects"]:
        v = sim[f"verts_{o['name']}"]
        c0, c1 = v[0].mean(0), v[-1].mean(0)
        ext = (v.max(1) - v.min(1))
        squash = 1 - ext.min() / (2 * o["radius"])
        print(f"  {o['name']}: E={o['E']:.3g} Pa  start {np.round(c0, 2)}  end {np.round(c1, 2)}  "
              f"max squash {100 * squash:.0f}%  in region: {bool(np.all((v >= lo - 1e-3) & (v <= hi + 1e-3)))}")
    print(f"wrote {out / 'sim.gif'} and {out / 'strain.png'}")


# ---------------------------------------------------------------------- cam sheets
def cam_names(sdir, which):
    rig = json.load(open(sdir / "rig.json"))
    names = [c["name"] for c in rig["cameras"]]
    if which in ("all", None):
        return names
    if which in ("train", "test"):
        return [c["name"] for c in rig["cameras"] if c["split"] == which]
    return which.split(",")


def cmd_cams(a):
    sdir = Path(a.scene_dir)
    f = peak_frame(sdir) if a.frame is None else a.frame
    tiles = []
    for n in cam_names(sdir, a.cams):
        p = sdir / "images" / n / f"{f:05d}.png"
        if p.exists():
            t = Image.open(p).convert("RGB")
            t.thumbnail((a.size, a.size))
        else:
            t = Image.new("RGB", (a.size, a.size), (250, 220, 220))
        tiles.append(label(t, n, size=12))
    out = sdir / "preview"
    out.mkdir(exist_ok=True)
    path = out / f"cams_frame{f:05d}.jpg"
    sheet(tiles, a.cols).save(path, quality=90)
    print(f"wrote {path}  (pink tiles = image not rendered yet)")


def cmd_camgif(a):
    sdir = Path(a.scene_dir)
    files = sorted((sdir / "images" / a.cam).glob("*.png"))
    if not files:
        raise SystemExit(f"no images for {a.cam}")
    fps = json.load(open(sdir / "sim.json"))["scene"]["fps"]
    frames = []
    for p in files[:: a.every]:
        t = Image.open(p).convert("RGB")
        t.thumbnail((a.size, a.size))
        frames.append(label(t, f"{a.cam}  frame {int(p.stem)}", size=12))
    out = sdir / "preview"
    out.mkdir(exist_ok=True)
    path = out / f"{a.cam}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * a.every / fps), loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


# ------------------------------------------------------------------------ compare
def cmd_compare(a):
    sdir, mp = Path(a.scene_dir), Path(a.model_path).expanduser()
    subset = mp.name
    index = json.load(open(sdir / "4dgs" / subset / "test_index.json"))["test"]
    method = sorted((mp / "test").glob("ours_*"), key=lambda p: int(p.name.split("_")[-1]))[-1]
    cams = sorted({e["camera"] for e in index})[: a.n_cams]
    frames = sorted({e["frame"] for e in index})
    pk = peak_frame(sdir)
    want = sorted({frames[0], min(frames, key=lambda x: abs(x - pk)), frames[-1]})
    lookup = {(e["camera"], e["frame"]): e["index"] for e in index}
    tiles = []
    for c in cams:
        for f in want:
            i = lookup[(c, f)]
            for kind in ("gt", "renders"):
                t = Image.open(method / kind / f"{i:05d}.png").convert("RGB")
                t.thumbnail((a.size, a.size))
                tiles.append(label(t, f"{c} f{f} {'GT' if kind == 'gt' else 'pred'}", size=11))
    out = sdir / "preview"
    out.mkdir(exist_ok=True)
    path = out / f"compare_{subset}.jpg"
    sheet(tiles, 2 * len(want)).save(path, quality=90)
    res = mp / "results.json"
    if res.exists():
        r = json.load(open(res))[method.name]
        print(f"{subset}: PSNR {r.get('PSNR', float('nan')):.2f}  SSIM {r.get('SSIM', float('nan')):.4f}  "
              f"LPIPS {r.get('LPIPS-vgg', float('nan')):.4f}")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim", help="GIF of the MPM particles + strain curve")
    s.add_argument("--scene-dir", required=True)
    s.add_argument("--every", type=int, default=1)
    s = sub.add_parser("cams", help="contact sheet of every camera at one frame")
    s.add_argument("--scene-dir", required=True)
    s.add_argument("--frame", type=int, default=None, help="default: peak-strain frame")
    s.add_argument("--cams", default="all")
    s.add_argument("--size", type=int, default=200)
    s.add_argument("--cols", type=int, default=10)
    s = sub.add_parser("camgif", help="GIF of one camera over time")
    s.add_argument("--scene-dir", required=True)
    s.add_argument("--cam", required=True)
    s.add_argument("--size", type=int, default=400)
    s.add_argument("--every", type=int, default=1)
    s = sub.add_parser("compare", help="GT vs 4DGS render for a trained subset")
    s.add_argument("--scene-dir", required=True)
    s.add_argument("--model-path", required=True, help="4DGaussians output folder, e.g. .../cams_12")
    s.add_argument("--n-cams", type=int, default=3)
    s.add_argument("--size", type=int, default=240)
    a = ap.parse_args()
    {"sim": cmd_sim, "cams": cmd_cams, "camgif": cmd_camgif, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
