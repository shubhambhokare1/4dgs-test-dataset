"""
Render a simulated scene from every camera of the rig with Blender.

Runs either inside Blender or with the `bpy` pip module:

    blender -b -P mpm/render_blender.py -- --scene-dir out/scene3_collision
    python mpm/render_blender.py --scene-dir out/scene3_collision      # pip install bpy

Reads   <scene-dir>/sim.npz, sim.json   (from sim_mpm.py)
Writes  <scene-dir>/rig.json            camera rig (nested train subsets + test cams)
        <scene-dir>/cameras.json        intrinsics/extrinsics read back from Blender
        <scene-dir>/images/<cam>/<frame:05d>.png

Rendering every camera at every frame is ~58 cams x 91 frames per scene. Use
--cams / --frames to split the job across GPUs or machines; existing images are
skipped so runs can resume.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import build_rig, intrinsics_from_fovy, c2w_gl_to_w2c_cv  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--cams", default="all",
                    help="'all', 'train', 'test', or comma list like cam00,cam01,test00")
    ap.add_argument("--frames", default=":", help="python slice over frames, e.g. 0:91:3")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--gpu", type=int, default=None,
                    help="use only this GPU (nvidia-smi index). Leave CUDA_VISIBLE_DEVICES unset")
    ap.add_argument("--texture", default="spots", choices=["spots", "plain"],
                    help="dark spots on the deformable objects make motion and "
                         "deformation observable (plain = flat colour like the old dataset)")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--images-dir", default="images",
                    help="output folder inside scene-dir (use another name for test renders)")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ scene building
def pci_key(dev):
    """Sort key: the PCI address in a Cycles device id, which matches nvidia-smi's order."""
    import re
    m = re.search(r"([0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}(\.[0-9])?", dev.id)
    return m.group(0).lower() if m else dev.id


def setup_render(args):
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"  # keep the white background white
    scene.view_settings.look = "None"
    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "CPU"
        if args.device == "GPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                try:
                    prefs.compute_device_type = backend
                except TypeError:
                    continue
                prefs.get_devices()
                gpus = sorted((d for d in prefs.devices if d.type == backend), key=pci_key)
                if not gpus:
                    continue
                if args.gpu is not None:
                    if not 0 <= args.gpu < len(gpus):
                        raise SystemExit(f"--gpu {args.gpu}: only {len(gpus)} {backend} devices: "
                                         f"{[d.id for d in gpus]}")
                    chosen = [gpus[args.gpu]]
                else:
                    chosen = gpus
                for d in prefs.devices:
                    d.use = any(d.id == c.id for c in chosen)
                scene.cycles.device = "GPU"
                print(f"Cycles on {backend}: using {[d.id for d in chosen]} "
                      f"(of {len(gpus)} GPUs, ordered by PCI bus = nvidia-smi index)")
                break
            else:
                print("WARNING: no GPU found for Cycles, rendering on CPU")
    else:
        for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
    return scene


def setup_world(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    lp = nt.nodes.new("ShaderNodeLightPath")
    seen = nt.nodes.new("ShaderNodeBackground")      # what the camera sees: pure white
    seen.inputs["Color"].default_value = (1, 1, 1, 1)
    seen.inputs["Strength"].default_value = 1.0
    amb = nt.nodes.new("ShaderNodeBackground")       # soft ambient fill for lighting
    amb.inputs["Color"].default_value = (1, 1, 1, 1)
    amb.inputs["Strength"].default_value = 0.5
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(amb.outputs[0], mix.inputs[1])
    nt.links.new(seen.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 3.0
    sun.angle = math.radians(10.0)
    sun_obj = bpy.data.objects.new("Sun", sun)
    sun_obj.rotation_euler = (math.radians(20.0), math.radians(10.0), 0.0)
    scene.collection.objects.link(sun_obj)


def srgb_to_linear(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def make_material(name, color, spots=False, spot_scale=10.0):
    # Scene colours are display (sRGB) values as in MuJoCo; Blender wants linear.
    color = srgb_to_linear(color)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    base = (*color, 1.0)
    if not spots:
        bsdf.inputs["Base Color"].default_value = base
        return mat
    # Spots follow the material: Voronoi on the undeformed (rest) position,
    # stored per vertex, so they move and stretch with the surface.
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "rest_pos"
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.voronoi_dimensions = "3D"
    vor.inputs["Scale"].default_value = spot_scale
    less = nt.nodes.new("ShaderNodeMath")
    less.operation = "LESS_THAN"
    less.inputs[1].default_value = 0.3
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = base                                   # A
    mix.inputs[7].default_value = (0.15 * color[0], 0.15 * color[1], 0.15 * color[2], 1)  # B
    nt.links.new(attr.outputs["Vector"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], less.inputs[0])
    nt.links.new(less.outputs[0], mix.inputs[0])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def add_deformable(scene, name, rest, verts0, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts0.tolist(), [], faces.tolist())
    mesh.update()
    attr = mesh.attributes.new("rest_pos", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", rest.astype(np.float32).ravel())
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return mesh


def add_box(scene, name, center, half, mat):
    c, h = np.asarray(center), np.asarray(half)
    v = [tuple(c + h * np.array(s)) for s in
         [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(v, [], f)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)


def add_floor(scene, z):
    mesh = bpy.data.meshes.new("floor")
    s = 50.0
    mesh.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [(0, 1, 2, 3)])
    mesh.materials.append(make_material("floor", (1.0, 1.0, 1.0)))
    obj = bpy.data.objects.new("floor", mesh)
    scene.collection.objects.link(obj)


def add_cameras(scene, rig):
    """One Blender camera per rig camera; returns {name: object}."""
    sensor = 24.0
    lens = (sensor / 2.0) / math.tan(math.radians(rig["fovy_deg"]) / 2.0)
    out = {}
    for c in rig["cameras"]:
        cam = bpy.data.cameras.new(c["name"])
        cam.sensor_fit = "VERTICAL"
        cam.sensor_width = cam.sensor_height = sensor
        cam.lens = lens
        cam.clip_start, cam.clip_end = 0.01, 200.0
        obj = bpy.data.objects.new(c["name"], cam)
        obj.matrix_world = Matrix(c["c2w_gl"])
        scene.collection.objects.link(obj)
        out[c["name"]] = obj
    return out


def export_cameras(scene, rig, cam_objs, path):
    """Read intrinsics/extrinsics back from Blender and check them against the rig."""
    bpy.context.view_layer.update()
    W, H = scene.render.resolution_x, scene.render.resolution_y
    K_rig = np.asarray(intrinsics_from_fovy(rig["fovy_deg"], W, H))
    cams, worst = {}, 0.0
    for c in rig["cameras"]:
        o = cam_objs[c["name"]]
        d = o.data
        f_px = d.lens / d.sensor_height * H  # vertical sensor fit
        K = np.array([[f_px, 0, W / 2 - d.shift_x * W], [0, f_px, H / 2 + d.shift_y * H], [0, 0, 1]])
        c2w = np.array(o.matrix_world)
        worst = max(worst, float(np.abs(K - K_rig).max() / f_px), float(np.abs(c2w - np.asarray(c["c2w_gl"])).max()))
        cams[c["name"]] = dict(
            split=c["split"], K=K.tolist(), width=W, height=H,
            fovy_deg=math.degrees(2 * math.atan(H / (2 * f_px))),
            fovx_deg=math.degrees(2 * math.atan(W / (2 * f_px))),
            c2w_gl=c2w.tolist(), w2c_cv=c2w_gl_to_w2c_cv(c2w).tolist(),
        )
    if worst > 1e-5:
        raise RuntimeError(f"Blender cameras differ from the rig (max rel. error {worst:.2e})")
    with open(path, "w") as fh:
        json.dump(dict(convention=dict(
            c2w_gl="camera-to-world, OpenGL/Blender axes (x right, y up, -z forward); "
                   "= D-NeRF transform_matrix",
            w2c_cv="world-to-camera, OpenCV axes (x right, y down, +z forward)",
            K="pixel-corner convention: principal point at (W/2, H/2)"),
            cameras=cams), fh, indent=1)
    print(f"cameras verified against rig (max error {worst:.1e}); wrote {path}")


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    sdir = Path(args.scene_dir)
    sim = np.load(sdir / "sim.npz")
    meta = json.load(open(sdir / "sim.json"))
    scene_cfg = meta["scene"]

    rig = build_rig(scene_cfg, n_train=args.n_train, n_test=args.n_test,
                    fovy_deg=50.0, width=args.res, height=args.res)
    with open(sdir / "rig.json", "w") as fh:
        json.dump(rig, fh, indent=1)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = setup_render(args)
    setup_world(scene)
    if not args.no_floor:
        add_floor(scene, scene_cfg["floor_z"])

    meshes = []
    for o in scene_cfg["objects"]:
        n = o["name"]
        mat = make_material(n, o["color"], spots=args.texture == "spots",
                            spot_scale=3.0 / o["radius"])
        verts = sim[f"verts_{n}"]
        meshes.append((add_deformable(scene, n, sim[f"rest_{n}"], verts[0], sim[f"faces_{n}"], mat), verts))
    for s in scene_cfg["static_objects"]:
        add_box(scene, s["name"], s["center"], s["half"], make_material(s["name"], s["color"]))

    cam_objs = add_cameras(scene, rig)
    export_cameras(scene, rig, cam_objs, sdir / "cameras.json")

    if args.cams == "all":
        names = [c["name"] for c in rig["cameras"]]
    elif args.cams in ("train", "test"):
        names = [c["name"] for c in rig["cameras"] if c["split"] == args.cams]
    else:
        names = args.cams.split(",")
    n_frames = len(sim["times"])
    frames = list(range(n_frames))[slice(*[int(p) if p else None for p in args.frames.split(":")])]

    total, done, rendered, t0 = len(frames) * len(names), 0, 0, time.time()
    for f in frames:
        for mesh, verts in meshes:
            mesh.vertices.foreach_set("co", verts[f].astype(np.float32).ravel())
            mesh.update()
        for name in names:
            path = sdir / args.images_dir / name / f"{f:05d}.png"
            done += 1
            if path.exists() and not args.overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            scene.camera = cam_objs[name]
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            rendered += 1
        el = time.time() - t0
        print(f"frame {f}: {done}/{total} images, {el:.0f}s elapsed", flush=True)
    if rendered:
        per = (time.time() - t0) / rendered
        full = per * n_frames * len(rig["cameras"])
        print(f"TIMING: {per:.2f} s/image over {rendered} images; all {len(rig['cameras'])} cams x "
              f"{n_frames} frames would take ~{full / 3600:.1f} h with these settings")


if __name__ == "__main__":
    main()
