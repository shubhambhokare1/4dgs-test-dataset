"""Mesh and camera maths shared by the simulator, the Blender script and the exporters.

Kept free of Blender and Warp imports so it runs anywhere.

Camera conventions
------------------
* ``c2w_gl``  camera-to-world, OpenGL/Blender axes (x right, y up, looking down -z).
  This is what Blender's ``matrix_world`` holds and what D-NeRF / 4DGaussians
  expect in ``transform_matrix``.
* ``w2c_cv``  world-to-camera, OpenCV axes (x right, y down, looking down +z).
  Use with ``K`` for projection: ``uv = K @ (R @ X + t)``.
* ``K`` uses the pixel-corner convention: the principal point is (W/2, H/2) and
  pixel (i, j) covers [i, i+1) x [j, j+1). Subtract 0.5 from cx/cy for the
  strict OpenCV pixel-centre convention.
"""

import math

import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- meshes
def uv_sphere(radius, segments=64, rings=32):
    """Closed triangle mesh of a sphere with unique vertices (no seam duplicates).

    Returns (verts (V,3) float64, faces (T,3) int32), centred at the origin.
    """
    verts = [(0.0, 0.0, radius)]
    for r in range(1, rings):
        theta = math.pi * r / rings
        z = radius * math.cos(theta)
        s = radius * math.sin(theta)
        for k in range(segments):
            phi = 2.0 * math.pi * k / segments
            verts.append((s * math.cos(phi), s * math.sin(phi), z))
    verts.append((0.0, 0.0, -radius))
    verts = np.asarray(verts)

    def ring(r, k):  # r in 1..rings-1
        return 1 + (r - 1) * segments + (k % segments)

    south = len(verts) - 1
    faces = []
    for k in range(segments):
        faces.append((0, ring(1, k), ring(1, k + 1)))
    for r in range(1, rings - 1):
        for k in range(segments):
            a, b = ring(r, k), ring(r, k + 1)
            c, d = ring(r + 1, k + 1), ring(r + 1, k)
            faces.append((a, d, c))
            faces.append((a, c, b))
    for k in range(segments):
        faces.append((south, ring(rings - 1, k + 1), ring(rings - 1, k)))
    return verts, np.asarray(faces, dtype=np.int32)


def box_corners(center, half):
    c, h = np.asarray(center, float), np.asarray(half, float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return c + signs * h


# -------------------------------------------------------------------------- cameras
def look_at_c2w_gl(pos, target, world_up=(0.0, 0.0, 1.0)):
    """Camera-to-world in OpenGL/Blender convention, looking from pos at target."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(world_up, float)
    if abs(np.dot(fwd, up)) > 0.999:  # looking straight down/up
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, cam_up, -fwd, pos
    return m


def c2w_gl_to_w2c_cv(c2w_gl):
    return np.linalg.inv(np.asarray(c2w_gl) @ GL_TO_CV)


def intrinsics_from_fovy(fovy_deg, width, height):
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def project(points, K, w2c_cv):
    """Project world points (N,3) -> pixel coords (N,2) and depth (N,)."""
    p = np.asarray(points, float)
    cam = (w2c_cv[:3, :3] @ p.T + w2c_cv[:3, 3:4]).T
    z = cam[:, 2]
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3], z


# ------------------------------------------------------------------------ camera rig
def fibonacci_sphere(n, min_elev_deg, max_elev_deg):
    """Roughly uniform unit directions in an elevation band (deg above horizon)."""
    zlo, zhi = math.sin(math.radians(min_elev_deg)), math.sin(math.radians(max_elev_deg))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = zlo + (zhi - zlo) * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * i
        out.append((r * math.cos(phi), r * math.sin(phi), z))
    return np.asarray(out)


def build_rig(scene, n_train=50, n_test=8, fovy_deg=50.0, width=800, height=800):
    """Camera rig with nested training subsets and a fixed held-out test set.

    Training order: the 12 original MuJoCo cameras first (cams 0-3 ring, 4-7
    diagonals, 8-11 low/top views, so 4 < 8 < 12 match the old rig), then extra
    cameras chosen by farthest-point sampling on a sphere around the target.
    Any prefix of the training list is therefore a well-spread subset, and
    4 < 8 < 12 < 25 < 50 are nested by construction.
    """
    target = np.asarray(scene["target"], float)
    mj = [(n, np.asarray(p, float)) for n, p in scene["mujoco_cameras"]]
    radius = float(np.median([np.linalg.norm(p - target) for _, p in mj]))

    # Test cameras: two rings at 20 and 40 deg elevation, 45 deg apart in
    # azimuth, offset 22.5 deg so none coincide with the cardinal training views.
    test_dirs = []
    for k in range(n_test):
        az = math.radians(22.5 + 45.0 * k)
        el = math.radians(20.0 if k % 2 == 0 else 40.0)
        test_dirs.append((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    test_dirs = np.asarray(test_dirs)

    train_pos = [p for _, p in mj][:n_train]
    train_names = [f"cam{i:02d}" for i in range(len(train_pos))]
    sources = ["mujoco"] * len(train_pos)

    if n_train > len(train_pos):
        cand = fibonacci_sphere(2000, 8.0, 80.0)
        # keep candidates away from the held-out views (> 12 deg)
        cand = cand[np.max(cand @ test_dirs.T, axis=1) < math.cos(math.radians(12.0))]
        chosen = [(p - target) / np.linalg.norm(p - target) for p in train_pos]
        mind = np.full(len(cand), np.inf)
        for d in chosen:
            mind = np.minimum(mind, np.linalg.norm(cand - d, axis=1))
        while len(train_pos) < n_train:
            j = int(np.argmax(mind))
            d = cand[j]
            train_pos.append(target + radius * d)
            train_names.append(f"cam{len(train_names):02d}")
            sources.append("fps")
            mind = np.minimum(mind, np.linalg.norm(cand - d, axis=1))

    K = intrinsics_from_fovy(fovy_deg, width, height)
    cams = []
    for name, pos, src in zip(train_names, train_pos, sources):
        cams.append(dict(name=name, split="train", source=src, position=pos.tolist()))
    for k, d in enumerate(test_dirs):
        cams.append(dict(name=f"test{k:02d}", split="test", source="fixed",
                         position=(target + radius * d).tolist()))
    for c in cams:
        c2w = look_at_c2w_gl(c["position"], target)
        c["c2w_gl"] = c2w.tolist()
        c["w2c_cv"] = c2w_gl_to_w2c_cv(c2w).tolist()
    return dict(
        scene=scene["name"], target=target.tolist(), radius=radius,
        fovy_deg=fovy_deg, width=width, height=height, K=K.tolist(),
        subsets=[n for n in (4, 8, 12, 25, 50) if n <= n_train],
        cameras=cams,
    )
