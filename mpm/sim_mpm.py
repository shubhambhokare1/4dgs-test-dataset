"""
MLS-MPM simulation of the ported scenes, written with NVIDIA Warp.

Elastic solids use the fixed-corotated model (the PhysGaussian / warp-mpm
default) with a per-particle Young's modulus, so every object can have its own
stiffness. Rigid obstacles (the scene-7 cube) are grid colliders.

Each object's render mesh is carried along as passive tracer vertices that are
advected with the mass-weighted grid velocity, so Blender gets a deformed
surface every frame without any surface reconstruction step.

Usage:
    python mpm/sim_mpm.py --scene 3 --out out            # GPU if available
    python mpm/sim_mpm.py --scene 7 --out out --device cpu --dx 0.06   # quick test

Output (out/<scene>/):
    sim.npz    per-frame tracer vertices for each object, mesh faces, a
               subsample of MPM particles per frame (for point clouds)
    sim.json   the exact scene + solver settings used
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import uv_sphere  # noqa: E402
from scenes import get_scene  # noqa: E402

BOUND = 3  # wall thickness in grid cells


# ----------------------------------------------------------------------- kernels
@wp.func
def w1d(f: float, i: int):
    if i == 0:
        return 0.5 * (1.5 - f) * (1.5 - f)
    elif i == 1:
        return 0.75 - (f - 1.0) * (f - 1.0)
    return 0.5 * (f - 0.5) * (f - 0.5)


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
        C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
        mu: wp.array(dtype=float), lam: wp.array(dtype=float),
        p_mass: wp.array(dtype=float), p_vol: float,
        origin: wp.vec3, dx: float, inv_dx: float, dt: float,
        grid_m: wp.array3d(dtype=float), grid_mv: wp.array3d(dtype=wp.vec3)):
    p = wp.tid()
    xg = (x[p] - origin) * inv_dx
    bx = int(wp.floor(xg[0] - 0.5))
    by = int(wp.floor(xg[1] - 0.5))
    bz = int(wp.floor(xg[2] - 0.5))
    fx = xg - wp.vec3(float(bx), float(by), float(bz))

    Fp = F[p]
    U = wp.mat33()
    s = wp.vec3()
    V = wp.mat33()
    wp.svd3(Fp, U, s, V)
    R = U @ wp.transpose(V)
    if wp.determinant(R) < 0.0:
        U = U @ wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
        R = U @ wp.transpose(V)
    J = wp.determinant(Fp)
    I = wp.identity(n=3, dtype=float)
    # Kirchhoff stress P F^T for fixed corotated elasticity
    PFt = 2.0 * mu[p] * (Fp - R) @ wp.transpose(Fp) + lam[p] * (J - 1.0) * J * I
    stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * PFt
    m = p_mass[p]
    affine = stress + m * C[p]
    vp = v[p]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = w1d(fx[0], i) * w1d(fx[1], j) * w1d(fx[2], k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                wp.atomic_add(grid_mv, bx + i, by + j, bz + k, w * (m * vp + affine @ dpos))
                wp.atomic_add(grid_m, bx + i, by + j, bz + k, w * m)


@wp.kernel
def grid_update(grid_m: wp.array3d(dtype=float), grid_mv: wp.array3d(dtype=wp.vec3),
                grid_v: wp.array3d(dtype=wp.vec3),
                dt: float, gravity: wp.vec3, origin: wp.vec3, dx: float,
                nx: int, ny: int, nz: int, bound: int, floor_friction: float,
                has_box: int, box_lo: wp.vec3, box_hi: wp.vec3, box_friction: float):
    i, j, k = wp.tid()
    m = grid_m[i, j, k]
    vv = wp.vec3(0.0, 0.0, 0.0)
    if m > 0.0:
        vv = grid_mv[i, j, k] / m + dt * gravity
        # side walls and ceiling: free slip, separating
        if i < bound and vv[0] < 0.0:
            vv = wp.vec3(0.0, vv[1], vv[2])
        if i >= nx - bound and vv[0] > 0.0:
            vv = wp.vec3(0.0, vv[1], vv[2])
        if j < bound and vv[1] < 0.0:
            vv = wp.vec3(vv[0], 0.0, vv[2])
        if j >= ny - bound and vv[1] > 0.0:
            vv = wp.vec3(vv[0], 0.0, vv[2])
        if k >= nz - bound and vv[2] > 0.0:
            vv = wp.vec3(vv[0], vv[1], 0.0)
        # floor: separating with Coulomb friction
        if k < bound and vv[2] < 0.0:
            vn = vv[2]
            vt = wp.vec3(vv[0], vv[1], 0.0)
            lt = wp.length(vt)
            if lt > 0.0:
                vt = vt * wp.max(0.0, 1.0 + floor_friction * vn / lt)
            vv = vt
        # rigid box collider
        if has_box == 1:
            pos = origin + wp.vec3(float(i), float(j), float(k)) * dx
            if pos[0] > box_lo[0] and pos[0] < box_hi[0] and pos[1] > box_lo[1] and pos[1] < box_hi[1] and pos[2] > box_lo[2] and pos[2] < box_hi[2]:
                # outward normal of the nearest face
                n = wp.vec3(-1.0, 0.0, 0.0)
                d = pos[0] - box_lo[0]
                if box_hi[0] - pos[0] < d:
                    d = box_hi[0] - pos[0]
                    n = wp.vec3(1.0, 0.0, 0.0)
                if pos[1] - box_lo[1] < d:
                    d = pos[1] - box_lo[1]
                    n = wp.vec3(0.0, -1.0, 0.0)
                if box_hi[1] - pos[1] < d:
                    d = box_hi[1] - pos[1]
                    n = wp.vec3(0.0, 1.0, 0.0)
                if pos[2] - box_lo[2] < d:
                    d = pos[2] - box_lo[2]
                    n = wp.vec3(0.0, 0.0, -1.0)
                if box_hi[2] - pos[2] < d:
                    d = box_hi[2] - pos[2]
                    n = wp.vec3(0.0, 0.0, 1.0)
                vn = wp.dot(vv, n)
                if vn < 0.0:
                    vt = vv - vn * n
                    lt = wp.length(vt)
                    if lt > 0.0:
                        vt = vt * wp.max(0.0, 1.0 + box_friction * vn / lt)
                    vv = vt
    grid_v[i, j, k] = vv


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
        C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
        grid_v: wp.array3d(dtype=wp.vec3),
        origin: wp.vec3, dx: float, inv_dx: float, dt: float,
        lo: wp.vec3, hi: wp.vec3):
    p = wp.tid()
    xg = (x[p] - origin) * inv_dx
    bx = int(wp.floor(xg[0] - 0.5))
    by = int(wp.floor(xg[1] - 0.5))
    bz = int(wp.floor(xg[2] - 0.5))
    fx = xg - wp.vec3(float(bx), float(by), float(bz))
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = w1d(fx[0], i) * w1d(fx[1], j) * w1d(fx[2], k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                g = grid_v[bx + i, by + j, bz + k]
                new_v = new_v + w * g
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(g, dpos)
    v[p] = new_v
    C[p] = new_C
    F[p] = (wp.identity(n=3, dtype=float) + dt * new_C) @ F[p]
    xn = x[p] + dt * new_v
    x[p] = wp.vec3(wp.clamp(xn[0], lo[0], hi[0]), wp.clamp(xn[1], lo[1], hi[1]),
                   wp.clamp(xn[2], lo[2], hi[2]))


@wp.kernel
def advect_tracers(xt: wp.array(dtype=wp.vec3), grid_v: wp.array3d(dtype=wp.vec3),
                   grid_m: wp.array3d(dtype=float),
                   origin: wp.vec3, inv_dx: float, dt: float, lo: wp.vec3, hi: wp.vec3):
    t = wp.tid()
    xg = (xt[t] - origin) * inv_dx
    bx = int(wp.floor(xg[0] - 0.5))
    by = int(wp.floor(xg[1] - 0.5))
    bz = int(wp.floor(xg[2] - 0.5))
    fx = xg - wp.vec3(float(bx), float(by), float(bz))
    mv = wp.vec3(0.0, 0.0, 0.0)
    ms = float(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = w1d(fx[0], i) * w1d(fx[1], j) * w1d(fx[2], k)
                m = grid_m[bx + i, by + j, bz + k]
                mv = mv + (w * m) * grid_v[bx + i, by + j, bz + k]
                ms = ms + w * m
    if ms > 0.0:
        xn = xt[t] + dt * (mv / ms)
        xt[t] = wp.vec3(wp.clamp(xn[0], lo[0], hi[0]), wp.clamp(xn[1], lo[1], hi[1]),
                        wp.clamp(xn[2], lo[2], hi[2]))


# ------------------------------------------------------------------------ setup
def lame(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sample_sphere(center, radius, spacing):
    n = int(math.ceil(radius / spacing))
    g = (np.arange(-n, n + 1) + 0.5) * spacing
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts[np.linalg.norm(pts, axis=1) <= radius]
    return pts + np.asarray(center)


def run(scene, out_dir, device=None, dx=None, ppc_axis=2, cfl=0.3, n_save_particles=4000,
        sphere_segments=64, sphere_rings=32):
    wp.init()
    device = device or ("cuda" if wp.is_cuda_available() else "cpu")
    dx = float(dx or scene["dx"])
    inv_dx = 1.0 / dx

    # Grid covers the physical region plus BOUND+1 cells of padding per side,
    # so the walls sit exactly on the region boundary (floor at floor_z).
    pad = (BOUND + 1) * dx
    origin = np.asarray(scene["region_lo"], float) - pad
    extent = np.asarray(scene["region_hi"], float) + pad - origin
    nx, ny, nz = (int(math.ceil(e / dx)) + 1 for e in extent)
    lo = origin + (BOUND + 0.5) * dx
    hi = origin + (np.array([nx, ny, nz]) - BOUND - 1.5) * dx

    spacing = dx / ppc_axis
    p_vol = spacing ** 3
    xs, vs, mus, lams, masses, obj_id = [], [], [], [], [], []
    tracers, faces, rests = [], [], []
    for oi, obj in enumerate(scene["objects"]):
        assert obj["shape"] == "sphere", "only spheres are deformable for now"
        pts = sample_sphere(obj["center"], obj["radius"], spacing)
        mu, la = lame(obj["E"], obj["nu"])
        xs.append(pts)
        vs.append(np.tile(obj["velocity"], (len(pts), 1)))
        mus.append(np.full(len(pts), mu))
        lams.append(np.full(len(pts), la))
        masses.append(np.full(len(pts), obj["density"] * p_vol))
        obj_id.append(np.full(len(pts), oi))
        vtx, fc = uv_sphere(obj["radius"], sphere_segments, sphere_rings)
        rests.append(vtx)
        tracers.append(vtx + np.asarray(obj["center"]))
        faces.append(fc)
    x0 = np.concatenate(xs)
    obj_id = np.concatenate(obj_id)
    n_p = len(x0)

    # time step from the CFL condition on the stiffest material
    c_max = 0.0
    for obj in scene["objects"]:
        mu, la = lame(obj["E"], obj["nu"])
        c_max = max(c_max, math.sqrt((la + 2 * mu) / obj["density"]))
    v_max = max(np.linalg.norm(o["velocity"]) for o in scene["objects"])
    frame_dt = 1.0 / scene["fps"]
    dt_cfl = cfl * dx / (c_max + 2.0 * v_max)
    substeps = int(math.ceil(frame_dt / dt_cfl))
    dt = frame_dt / substeps
    n_frames = int(round(scene["duration"] * scene["fps"])) + 1

    print(f"[{scene['name']}] device={device} grid={nx}x{ny}x{nz} dx={dx:.4f} "
          f"particles={n_p} frames={n_frames} substeps/frame={substeps} dt={dt:.2e}")

    with wp.ScopedDevice(device):
        x = wp.array(x0, dtype=wp.vec3)
        v = wp.array(np.concatenate(vs), dtype=wp.vec3)
        C = wp.zeros(n_p, dtype=wp.mat33)
        F = wp.array(np.tile(np.eye(3), (n_p, 1, 1)), dtype=wp.mat33)
        mu_a = wp.array(np.concatenate(mus), dtype=float)
        lam_a = wp.array(np.concatenate(lams), dtype=float)
        mass_a = wp.array(np.concatenate(masses), dtype=float)
        tr_counts = [len(t) for t in tracers]
        xt = wp.array(np.concatenate(tracers), dtype=wp.vec3)
        grid_m = wp.zeros((nx, ny, nz), dtype=float)
        grid_mv = wp.zeros((nx, ny, nz), dtype=wp.vec3)
        grid_v = wp.zeros((nx, ny, nz), dtype=wp.vec3)

        box = scene["colliders"][0] if scene["colliders"] else None
        has_box = 1 if box else 0
        box_lo = wp.vec3(*(np.subtract(box["center"], box["half"]) if box else (0, 0, 0)))
        box_hi = wp.vec3(*(np.add(box["center"], box["half"]) if box else (0, 0, 0)))
        box_fr = float(box["friction"]) if box else 0.0
        o_w, lo_w, hi_w = wp.vec3(*origin), wp.vec3(*lo), wp.vec3(*hi)
        g_w = wp.vec3(*scene["gravity"])

        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(n_p, size=min(n_save_particles, n_p), replace=False))
        tracer_frames = np.zeros((n_frames, xt.shape[0], 3), np.float32)
        particle_frames = np.zeros((n_frames, len(keep), 3), np.float32)
        max_strain = np.zeros(n_frames, np.float32)

        def record(f):
            tracer_frames[f] = xt.numpy()
            particle_frames[f] = x.numpy()[keep]
            Fn = F.numpy()
            s = np.linalg.svd(Fn, compute_uv=False)
            max_strain[f] = float(np.max(np.abs(s - 1.0)))

        record(0)
        t0 = time.time()
        for f in range(1, n_frames):
            for _ in range(substeps):
                grid_m.zero_()
                grid_mv.zero_()
                wp.launch(p2g, dim=n_p, inputs=[x, v, C, F, mu_a, lam_a, mass_a, p_vol,
                                                o_w, dx, inv_dx, dt, grid_m, grid_mv])
                wp.launch(grid_update, dim=(nx, ny, nz),
                          inputs=[grid_m, grid_mv, grid_v, dt, g_w, o_w, dx, nx, ny, nz, BOUND,
                                  float(scene["floor_friction"]), has_box, box_lo, box_hi, box_fr])
                wp.launch(advect_tracers, dim=xt.shape[0],
                          inputs=[xt, grid_v, grid_m, o_w, inv_dx, dt, lo_w, hi_w])
                wp.launch(g2p, dim=n_p, inputs=[x, v, C, F, grid_v, o_w, dx, inv_dx, dt, lo_w, hi_w])
            record(f)
            if f % 10 == 0 or f == n_frames - 1:
                el = time.time() - t0
                print(f"  frame {f}/{n_frames - 1}  max|strain|={max_strain[f]:.3f}  "
                      f"{el:.0f}s elapsed, ~{el / f * (n_frames - 1 - f):.0f}s left", flush=True)
        if not np.all(np.isfinite(tracer_frames)):
            raise RuntimeError("simulation blew up (NaN); lower --cfl or --dx")

    out = Path(out_dir) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(times=np.arange(n_frames) / scene["fps"],
                  particles=particle_frames, particle_obj=obj_id[keep].astype(np.int32),
                  max_strain=max_strain)
    off = 0
    for obj, cnt, fc, rest in zip(scene["objects"], tr_counts, faces, rests):
        arrays[f"verts_{obj['name']}"] = tracer_frames[:, off:off + cnt]
        arrays[f"faces_{obj['name']}"] = fc
        arrays[f"rest_{obj['name']}"] = rest.astype(np.float32)
        off += cnt
    np.savez_compressed(out / "sim.npz", **arrays)
    meta = dict(scene=scene, solver=dict(
        method="MLS-MPM (APIC), fixed corotated elasticity", device=device, dx=dx,
        grid=[nx, ny, nz], origin=origin.tolist(), particles=int(n_p),
        particles_per_cell=ppc_axis ** 3, dt=dt, substeps_per_frame=substeps,
        n_frames=n_frames, cfl=cfl, wall_seconds=round(time.time() - t0, 1)))
    with open(out / "sim.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved {out / 'sim.npz'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="3, 7, scene3_collision or scene7_deformation")
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default=None, help="cuda / cuda:1 / cpu (default: cuda if available)")
    ap.add_argument("--dx", type=float, default=None, help="grid spacing override (m)")
    ap.add_argument("--duration", type=float, default=None, help="override duration (s)")
    ap.add_argument("--E-scale", type=float, default=1.0, help="multiply every Young's modulus")
    ap.add_argument("--E", type=float, nargs="+", default=None,
                    help="explicit Young's modulus per deformable object (Pa)")
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-simulate even if sim.npz exists (existing renders would no longer match)")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    if (Path(args.out) / scene["name"] / "sim.npz").exists() and not args.overwrite:
        print(f"{scene['name']}: sim.npz exists, skipping (use --overwrite to re-run)")
        return
    if args.duration:
        scene["duration"] = args.duration
    if args.E:
        assert len(args.E) == len(scene["objects"]), "need one --E per object"
        for o, e in zip(scene["objects"], args.E):
            o["E"] = e
    for o in scene["objects"]:
        o["E"] *= args.E_scale
    run(scene, args.out, device=args.device, dx=args.dx, cfl=args.cfl)


if __name__ == "__main__":
    main()
