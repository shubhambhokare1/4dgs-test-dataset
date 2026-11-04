"""
Export a rendered scene to the multi-view 4DGS (D-NeRF / Blender-synthetic) format,
once per camera-count subset.

    python mpm/export_4dgs.py --scene-dir out/scene3_collision

Writes out/<scene>/4dgs/cams_<N>/ for N in the rig's subsets (4, 8, 12, 25, 50):
    transforms_train.json   first N training cameras x every frame
    transforms_test.json    the fixed held-out test cameras (same for every N)
    transforms_val.json     a small slice of the test set
    fused.ply               initial point cloud (identical for every N)
    test_index.json         maps 4DGaussians' test render index -> (camera, frame)

Unlike scripts/dnerf_eq.py (one random camera per timestamp), every training
camera is used at every timestamp, which is what a synchronised multi-camera
rig records. Image paths point back to ../../images so nothing is copied;
pass --copy if your loader cannot follow relative paths.

Memory note: 4DGaussians' Blender loader keeps every training image in RAM
resized to 800x800 float32 (~7.7 MB each). 50 cameras x 91 frames is ~35 GB.
Use --frame-step 2 to halve that if needed.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    n = len(xyz)
    normals = np.zeros_like(xyz)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
        rec["n" + k] = normals[:, i]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())


def initial_points(sdir, meta, mode, n_random, bounds, offset, seed=0):
    """Point cloud used to initialise the Gaussians (the same for every subset).

    Object points are moved by ``offset`` into the exported (target-centred)
    frame; the random cube is already centred on the target there.
    """
    rng = np.random.default_rng(seed)
    pts, cols = [], []
    if mode in ("sim", "both"):
        sim = np.load(sdir / "sim.npz")
        for o in meta["objects"]:
            v = sim[f"verts_{o['name']}"][0]
            pts.append(v + offset)
            cols.append(np.tile(np.round(np.asarray(o["color"]) * 255), (len(v), 1)))
        for s in meta["static_objects"]:
            # points on the surface of static boxes
            c, h = np.asarray(s["center"]), np.asarray(s["half"])
            p = rng.uniform(-1, 1, (3000, 3))
            ax = rng.integers(0, 3, 3000)
            p[np.arange(3000), ax] = np.sign(p[np.arange(3000), ax])
            pts.append(c + p * h + offset)
            cols.append(np.tile(np.round(np.asarray(s["color"]) * 255), (3000, 1)))
    if mode in ("sim", "both"):
        # the floor (white, carries the shadows), as a square around the target
        n_floor = 20000
        xy = rng.uniform(-bounds, bounds, (n_floor, 2))
        z = np.full((n_floor, 1), meta["floor_z"] + offset[2])
        pts.append(np.hstack([xy, z]))
        cols.append(np.full((n_floor, 3), 235))
    if mode in ("random", "both"):
        # same recipe as scripts/dnerf_eq.py: uniform cube, random colours
        pts.append(rng.uniform(-bounds, bounds, (n_random, 3)))
        cols.append(rng.integers(0, 256, (n_random, 3)))
    return np.concatenate(pts), np.concatenate(cols)


def frame_entry(img_rel, t, cam, offset):
    K = cam["K"]
    c2w = np.array(cam["c2w_gl"])
    c2w[:3, 3] += offset
    return {
        "file_path": img_rel,
        "time": t,
        "transform_matrix": c2w.tolist(),
        "fl_x": K[0][0], "fl_y": K[1][1], "cx": K[0][2], "cy": K[1][2],
        "w": cam["width"], "h": cam["height"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--subsets", type=int, nargs="+", default=None)
    ap.add_argument("--frame-step", type=int, default=1, help="use every k-th frame for training")
    ap.add_argument("--test-frame-step", type=int, default=1, help="use every k-th frame for testing")
    ap.add_argument("--init", choices=["random", "sim", "both"], default="sim",
                    help="sim: object surfaces + floor at t=0 (default); random: uniform cube "
                         "like scripts/dnerf_eq.py; both: union")
    ap.add_argument("--n-random", type=int, default=50000)
    ap.add_argument("--copy", action="store_true", help="copy images instead of relative links")
    args = ap.parse_args()

    sdir = Path(args.scene_dir)
    rig = json.load(open(sdir / "rig.json"))
    cams = json.load(open(sdir / "cameras.json"))["cameras"]
    meta = json.load(open(sdir / "sim.json"))["scene"]
    n_frames = json.load(open(sdir / "sim.json"))["solver"]["n_frames"]
    times = [i / (n_frames - 1) for i in range(n_frames)]

    train_order = [c["name"] for c in rig["cameras"] if c["split"] == "train"]
    test_cams = [c["name"] for c in rig["cameras"] if c["split"] == "test"]
    subsets = args.subsets or rig["subsets"]

    if args.test_frame_step % args.frame_step:
        raise SystemExit("--test-frame-step must be a multiple of --frame-step "
                         "(4DGaussians only knows timestamps seen in training)")
    train_frames = list(range(0, n_frames, args.frame_step))
    test_frames = list(range(0, n_frames, args.test_frame_step))

    # every image we reference must exist
    need = [(c, f) for c in train_order[:max(subsets)] for f in train_frames]
    need += [(c, f) for c in test_cams for f in test_frames]
    missing = [(c, f) for c, f in need if not (sdir / "images" / c / f"{f:05d}.png").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}; run render_blender.py first")

    fovx = math.radians(cams[train_order[0]]["fovx_deg"])
    # Shift the world so the rig's look-at target is the origin. 4DGaussians'
    # Blender reader computes T = -inv(c2w)[:3, 3], which has the wrong sign on x
    # unless every camera's x-axis is orthogonal to its position (true for rigs
    # looking at the origin, as in D-NeRF). With the target at the origin every
    # look-at camera satisfies that, so the cameras load exactly.
    offset = -np.asarray(rig["target"], float)
    xyz, rgb = initial_points(sdir, meta, args.init, args.n_random, meta["bounds"], offset)

    test_list = [(c, f) for c in test_cams for f in test_frames]

    for N in subsets:
        out = sdir / "4dgs" / f"cams_{N:02d}"
        out.mkdir(parents=True, exist_ok=True)

        def rel(c, f, split):
            if args.copy:
                dst = out / split / c / f"{f:05d}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(sdir / "images" / c / f"{f:05d}.png", dst)
                return f"./{split}/{c}/{f:05d}"
            return f"../../images/{c}/{f:05d}"

        train = [frame_entry(rel(c, f, "train"), times[f], cams[c], offset)
                 for f in train_frames for c in train_order[:N]]
        test = [frame_entry(rel(c, f, "test"), times[f], cams[c], offset) for c, f in test_list]
        val = test[:: max(1, len(test) // 40)]
        for split, frames in (("train", train), ("test", test), ("val", val)):
            with open(out / f"transforms_{split}.json", "w") as fh:
                json.dump({"camera_angle_x": fovx, "world_offset": offset.tolist(), "frames": frames}, fh, indent=1)
        write_ply(out / "fused.ply", xyz, rgb)
        json.dump(dict(cameras=train_order[:N], test=[dict(index=i, camera=c, frame=f)
                                                      for i, (c, f) in enumerate(test_list)]),
                  open(out / "test_index.json", "w"), indent=1)
        print(f"cams_{N:02d}: {len(train)} train / {len(test)} test images -> {out}")


if __name__ == "__main__":
    main()
