# MPM scenes in Blender + camera-count study

This folder ports two scenes from the MuJoCo dataset to physically simulated
versions. It exports them for multi-view 4DGS and measures how reconstruction
quality changes with the number of cameras.

| Scene | What happens | Materials |
|---|---|---|
| `scene3_collision` | Three balls (r = 0.3 m) meet near the centre and rebound | Young's modulus 100 / 200 / 400 kPa (0.5x, 1x, 2x), ν = 0.3, 1000 kg/m³ |
| `scene7_deformation` | A ball (r = 0.4 m) hits a fixed cube and squashes about 30% | Ball E = 20 kPa, ν = 0.3; cube perfectly rigid |

The geometry, colours, camera positions and look-at targets come from
`scenarios/scene{3,7}_*.xml`. What changed from the MuJoCo version:

* **Motion is simulated.** In the old scenes the motion was scripted
  (`trajectories/`) and MuJoCo contacts were disabled. Now the objects start
  with a velocity and MLS-MPM computes the collision and deformation.
* **Duration is 3 s (91 frames at 30 fps).** Simulated rebounds would carry
  the balls out of view in 5–6 s.
* **Gravity is off.** This keeps the floating layout of the originals. Set
  `gravity` in `scenes.py` if you want it on.
* **Every camera uses a 50° FOV.** The old rig mixed 45/50/55°.
  4DGaussians' D-NeRF loader reads a single `camera_angle_x`, so the mixed
  FOVs in the old export silently gave some cameras the wrong intrinsics.
* **Deformable objects have dark spots by default** (`--texture spots`).
  Without them, rotation and surface strain on a flat-coloured ball are
  invisible to the cameras. Use `--texture plain` to match the old look.

## Pipeline

```
sim_mpm.py          MPM (NVIDIA Warp)       -> out/<scene>/sim.npz, sim.json
render_blender.py   Blender Cycles          -> out/<scene>/images/<cam>/<frame>.png,
                                               rig.json, cameras.json
check_projection.py reprojection check      -> out/<scene>/projection_check/
export_4dgs.py      4DGS (D-NeRF format)    -> out/<scene>/4dgs/cams_{04,08,12,25,50}/
run_study.sh        4DGaussians train/render/metrics for every subset
build_dashboard.py  static dashboard        -> dashboard/index.html
```

### Setup (lab server)

```bash
pip install warp-lang numpy pillow plotly   # plotly only to bundle the dashboard charts
pip install bpy                     # or install Blender and set BLENDER=/path/to/blender
git clone https://github.com/hustvl/4DGaussians   # and follow its install steps
```

`bpy` from pip requires the matching Python version (bpy 5.2 needs Python 3.13,
and bpy 4.x needs 3.11). If that clashes with your 4DGaussians environment, keep
two environments, or use a Blender binary through `BLENDER=`.

### Run step by step (recommended)

Each step is a separate script. It saves preview images to
`out/<scene>/preview/` and ends with a `>>> Look at:` line saying what to
check before you move on. Scenes are given as `3` or `7`. Extra arguments are
passed through to the underlying Python script.

```bash
bash mpm/steps/01_simulate.sh 7        # sim.gif, strain.png: objects collide and rebound, nothing explodes
bash mpm/steps/02_test_render.sh 7     # render_test.jpg: 6 cameras at frame 0 and at impact,
                                       #   plus a TIMING line estimating the full render time
bash mpm/steps/03_render.sh 7          # the full render (resumable); all cameras at impact + GIFs
bash mpm/steps/04_check_cameras.sh 7   # projection_check/overview.jpg: projected mesh lands on the objects
bash mpm/steps/05_export.sh 7          # 4DGS folders; re-reads the cameras the way 4DGaussians does
FOURDGS_ROOT=~/4DGaussians bash mpm/steps/06_train.sh 7 04   # one subset: GT vs prediction
FOURDGS_ROOT=~/4DGaussians bash mpm/steps/07_dashboard.sh    # dashboard from whatever has finished
```

Re-running a step is safe. `01` skips a scene that has already been
simulated (pass `--overwrite` to redo it; existing renders would then no
longer match). `03` skips images that already exist. Try render settings in
`02` (for example `--samples 32` or `--engine EEVEE`), then pass the same
settings to `03`. `PY=python3` and `OUT=/data/out` override the Python
binary and the output folder.

You can also look at things by hand: `python mpm/preview.py --help`.

### Run everything in one go

```bash
bash mpm/run_pipeline.sh                                   # simulate, render, check, export
FOURDGS_ROOT=~/4DGaussians bash mpm/run_study.sh           # train, score, build dashboard
```

To run the steps one at a time:

```bash
python mpm/sim_mpm.py --scene 3 --out out                 # ~1-3 min on a GPU
python mpm/render_blender.py --scene-dir out/scene3_collision          # 58 cams x 91 frames
python mpm/check_projection.py --scene-dir out/scene3_collision
python mpm/export_4dgs.py --scene-dir out/scene3_collision
```

Useful options:

* `sim_mpm.py --E 1e5 2e5 4e5` sets a Young's modulus per object.
  `--E-scale` multiplies all of them, and `--dx` sets the grid spacing.
* `render_blender.py --cams train|test|cam00,cam01 --frames 0:91:2` splits
  the render across GPUs or machines. Finished images are skipped.
* `export_4dgs.py --frame-step 2 --test-frame-step 2` halves the frame count
  (see the memory note below). `--init sim|random|both` chooses the starting
  point cloud (default `sim`: object surfaces and the floor at t = 0).

## Cameras

`rig.json` holds 50 training cameras and 8 held-out test cameras, all aimed at
the scene's original target.

* Cameras `cam00`–`cam11` are the original 12 MuJoCo positions. Cams 0–3 are
  the side ring, 4–7 the diagonals, 8–11 the low and top views. So the 4-,
  8- and 12-camera subsets are subsets of the old rig.
* Cameras `cam12`–`cam49` are added by farthest-point sampling on a sphere of
  the rigs' median radius. This keeps every prefix well spread, so the
  subsets are nested: **4 ⊂ 8 ⊂ 12 ⊂ 25 ⊂ 50**.
* `test00`–`test07` sit at 20°/40° elevation, at azimuths offset 22.5° from
  the training views. They are never used for training and are the same for
  every subset.

`cameras.json` is read back from Blender after the scene is built, and is
checked against `rig.json` to 1e-5. For each camera it stores:

* `K`: intrinsics. The principal point is (W/2, H/2), using the pixel-corner
  convention. Subtract 0.5 for OpenCV pixel centres.
* `c2w_gl`: camera-to-world in Blender/OpenGL axes. This is the D-NeRF
  `transform_matrix`.
* `w2c_cv`: world-to-camera in OpenCV axes, for `K @ (R X + t)`.

`check_projection.py` projects each deformed mesh with `K` and `w2c_cv`, and
requires at least 98% of each object's pixels to fall inside the projected
silhouette.

## 4DGS format

`out/<scene>/4dgs/cams_NN/` loads directly with 4DGaussians' Blender/D-NeRF
reader:

* `transforms_train.json` holds all N cameras at every frame (synchronised
  multi-view, unlike `scripts/dnerf_eq.py`, which picks one camera per
  timestamp).
* `transforms_test.json` holds the 8 test cameras.
* `fused.ply` is the initial point cloud: the object surfaces and the floor
  at t = 0. It is the same for every subset. The configs set
  `white_background = True`, so empty space renders white like the sky.
* `test_index.json` maps each render index back to its camera and frame.

Images are referenced as `../../images/...` and are not copied.

**World offset:** the exported datasets are shifted so that the rig's look-at
target sits at the origin. The shift is stored as `world_offset` in each
`transforms_*.json`. This works around a quirk in 4DGaussians' Blender
reader, which computes `T = -inv(c2w)[:3, 3]` and so gets the sign of the
camera x-translation wrong. That is harmless when the cameras look at the
origin (as in D-NeRF), but wrong otherwise. The old scene 7 export (target at
x = -1) was affected. `05_export.sh` checks that the cameras round-trip
exactly.

**Memory:** 4DGaussians loads every training image into RAM at 800×800
float32, about 7.7 MB each. The 50-camera subset with 91 frames needs about
35 GB, plus about 5.6 GB for the test set. If that is too much, export with
`--frame-step 2 --test-frame-step 2`.

Every subset trains for the same number of iterations (20k, from
`arguments/dnerf_default.py`), so the comparison is at equal compute.

## MPM details

* The solver is MLS-MPM (APIC transfer) with fixed-corotated elasticity, the
  PhysGaussian/warp-mpm default. Every particle has its own Young's modulus,
  so mixed materials work without changes.
* There are 8 particles per cell. The time step comes from the CFL condition
  on the stiffest material's wave speed.
* The scene-7 cube and the floor are grid colliders: separating contact, with
  optional Coulomb friction.
* Each ball's render mesh (a UV sphere, 64×32) is advected as passive tracer
  points through the mass-weighted grid velocity. Blender therefore receives
  a deformed surface every frame.
* Known limitation: objects touching in plain MPM interact through a shared
  grid. Separation is therefore slightly sticky, and the balls start pushing
  on each other about one grid cell (dx) before they touch. A smaller `--dx`
  reduces this.
