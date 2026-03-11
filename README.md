# 4DGS Synthetic Test Dataset

A synthetic benchmark dataset for evaluating **4D Gaussian Splatting (4DGS)** methods on controlled dynamic scenes. Each scene is rendered from 12 calibrated camera views using MuJoCo physics simulation and targets a specific known limitation of current 4D scene reconstruction methods.

All scenes use a white floor and white skybox to isolate object appearance from background, minimising confounding factors in evaluation.

---

## Scenes

| # | Scene | File | 4DGS Deficiency Targeted |
|---|-------|------|--------------------------|
| 1 | Close Proximity — Different Colors | `scene1_close_proximity` | Gaussian boundary preservation; identity maintenance when two differently-coloured objects pass within touching distance |
| 2 | Close Proximity — Identical Appearance | `scene2_identical_objects` | Identity tracking without appearance cues; two visually identical spheres cross paths |
| 3 | Three-Body Collision | `scene3_collision` | Multi-object interaction; chaotic bounce trajectories; temporal consistency before/during/after collision |
| 4 | Occlusion & Dis-occlusion | `scene4_occlusion` | Hallucination behind an occluder; correct reconstruction when objects reappear |
| 5 | Rapid Direction Changes | `scene5_rapid_motion` | Motion smoothness assumptions; temporal interpolation failure at sharp 90° turns |
| 6 | Extreme Scale Change | `scene6_scale_change` | Multi-scale representation; level-of-detail adaptation as object moves far-to-near |
| 7 | Deformable vs Rigid Collision | `scene7_deformation` | Non-rigid deformation capture; appearance change at contact; collision dynamics |
| 8 | Thin Structure Tracking | `scene8_thin_structure` | Thin geometry preservation; anisotropic Gaussian scaling for rod-like objects |
| 9 | Topology Change — Split & Merge | `scene9_topology` | Topology change handling; fixed Gaussian count assumption; split and merge events |
| 10 | High-Frequency Texture + Motion | `scene10_texture` | Fine texture preservation under motion; appearance vs geometry entanglement |

---

## Pipeline Overview

```
MuJoCo scenes + trajectories
        │
        ▼
generate_dataset.py   →   dataset/sceneN/   (RGBA PNGs, intrinsics, extrinsics)
        │
        ▼
dnerf_eq.py           →   dnerf/sceneN/     (D-NeRF format: transforms_*.json)
        │
        ▼
4DGS train / render / eval
```

---

## 1. Generating the Raw Dataset

`generate_dataset.py` renders all frames from all 12 cameras and saves RGBA images, camera intrinsics, and per-frame extrinsics.

```bash
# Single scene
python scripts/generate_dataset.py --scene 3

# Specific resolution and FPS
python scripts/generate_dataset.py --scene 3 --resolution 800x800 --fps 30

# All scenes
python scripts/generate_dataset.py --all
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s / --scene` | — | Scene number (1–10) |
| `--all` | — | Generate all scenes |
| `-o / --output` | `dataset` | Output root directory |
| `--resolution` | `800x800` | Render resolution as `WxH` |
| `--fps` | `30` | Frames per second |

**Output structure:**

```
dataset/
└── scene3/
    ├── images/
    │   ├── cam0/
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── cam1/
    │   └── ...
    ├── camera_intrinsics.json
    ├── camera_extrinsics.json
    └── metadata.json
```

---

## 2. Alpha Masking

Images are saved as **RGBA PNG** files. The alpha channel encodes a clean object mask computed directly from the MuJoCo depth buffer — no separate mask generation step is required.

```python
# Background detection via depth buffer (inside generate_dataset.py)
alpha = np.where(depth < depth.max() * 0.999, 255, 0).astype(np.uint8)
rgba  = np.dstack([rgb, alpha])
```

- **`α = 255`** — foreground (any rendered geometry)
- **`α = 0`** — background (skybox, beyond scene bounds)

Because the floor plane is rendered geometry, it is included in the foreground mask. If you need masks that exclude the floor, threshold on per-pixel depth distance instead of using the far-plane heuristic.

`generate_masks.py` provides a scaffold for per-object segmentation masks (separating individual objects). The current implementation writes placeholder zeros; a complete implementation would use MuJoCo's segmentation rendering mode or unique per-object colors.

```bash
python scripts/generate_masks.py --scene 3
python scripts/generate_masks.py --all
```

---

## 3. Converting to D-NeRF Format

`dnerf_eq.py` converts the raw dataset into the **D-NeRF / 4DGS training format**. It replicates the D-NeRF data convention: at every timestamp exactly **one randomly selected camera view** is used as the training observation, simulating the monocular-per-timestamp regime that D-NeRF and 4DGS are designed for.

### What it does

1. **Random camera sampling** — for each available frame index, one camera is drawn uniformly at random from all 12 cameras.
2. **Random frame shuffling** — the (camera, frame) pairs are shuffled so temporal ordering is not preserved in the file list.
3. **70 / 15 / 15 split** — pairs are divided into `train`, `val`, and `test` splits.
4. **Resolution** — images are rendered at the target resolution during dataset generation (default **800 × 800**, matching 4DGS's expected input). No additional resizing is applied by this script.
5. **`transforms_*.json`** — each split receives a NeRF-style transforms file containing `camera_angle_x`, per-frame focal lengths (`fl_x`, `fl_y`), principal point (`cx`, `cy`), image dimensions, normalised timestamp (`time` ∈ [0, 1]), and the 4×4 camera-to-world matrix.
6. **`fused.ply`** — a random point cloud covering `[-bounds, bounds]³` is written to the output directory. 4DGaussians reads this file to initialise Gaussians across the full scene volume, overriding its default narrow `[-1.3, 1.3]` random cloud.

**Single scene:**

```bash
python scripts/dnerf_eq.py \
    --dataset_path  dataset/scene3_collision/images \
    --intrinsics    dataset/scene3_collision/camera_intrinsics.json \
    --extrinsics    dataset/scene3_collision/camera_extrinsics.json \
    --output_path   dnerf/scene3_collision \
    --num_frames    150 \
    --bounds        2.5
```

**All scenes at once:**

```bash
python scripts/dnerf_eq.py --all \
    --dataset_dir dataset \
    --output_dir  dnerf
```

When `--all` is used, per-scene `num_frames` and `bounds` are set automatically from the values in the `arguments/` config files.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | — | Path to the `images/` directory (single-scene mode) |
| `--intrinsics` | — | Path to `camera_intrinsics.json` (single-scene mode) |
| `--extrinsics` | — | Path to `camera_extrinsics.json` (single-scene mode) |
| `--output_path` | — | Destination directory for the D-NeRF dataset (single-scene mode) |
| `--num_frames` | `150` | Number of (camera, frame) pairs to sample (single-scene mode) |
| `--bounds` | `2.5` | Spatial extent for `fused.ply` covering `[-bounds, bounds]³` (single-scene mode) |
| `--all` | `false` | Process all 10 scenes automatically |
| `--dataset_dir` | `dataset` | Root dataset directory (used with `--all`) |
| `--output_dir` | `dnerf` | Root output directory (used with `--all`) |

**Output structure:**

```
dnerf/scene3_collision/
├── fused.ply
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
│   ├── r_0000.png
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

## 4. Arguments Folder

The `arguments/` directory contains per-scene configuration files for 4DGaussians. Each file inherits from `dnerf_default.py` and overrides two parameter groups:

**`ModelHiddenParams.kplanes_config.resolution`** — the 4th element is the **temporal resolution** of the k-planes feature grid, set to match the exact frame count of the scene (`duration × fps`). This prevents under- or over-parameterisation of the temporal dimension.

**`ModelHiddenParams.bounds`** — the spatial extent of the scene in world units. This controls both the k-planes spatial grid and the volume covered by `fused.ply`. It should encompass the full range of object motion.

**`OptimizationParams` densification thresholds** — lowered from the 4DGS defaults so that objects reliably trigger densification. Scene 8 (thin rods, radius 0.02) uses an even lower threshold (`5e-5`) since very thin geometry produces small per-Gaussian gradients.

| File | Temporal frames | Spatial bounds | Notes |
|------|----------------|----------------|-------|
| `scene1_close_proximity.py` | 150 | 2.0 | Two spheres, ±1.5 travel in X |
| `scene2_identical_objects.py` | 150 | 2.0 | Two spheres, ±1.5 travel in X |
| `scene3_collision.py` | 150 | 3.5 | Spheres start at radius 2.5, bounce up to ~3.1 m from origin |
| `scene4_occlusion.py` | 120 | 2.5 | 4 s scene; sphere orbit radius 1.8 |
| `scene5_rapid_motion.py` | 180 | 2.5 | Zigzag clipped at ±2 in XY |
| `scene6_scale_change.py` | 180 | 3.5 | Corners at (~2.83 XY distance, z=3.0) |
| `scene7_deformation.py` | 180 | 3.5 | Sphere starts at x=−3.0 |
| `scene8_thin_structure.py` | 150 | 2.5 | Rods travel ±2.0; grad threshold halved |
| `scene9_topology.py` | 180 | 1.5 | Compact scene; max separation 1.0 m |
| `scene10_texture.py` | 150 | 2.0 | Orbit radius 1.5 + sphere radius 0.4 |

---

## 5. Training, Rendering, and Evaluation with 4DGS

The commands below assume the [4DGaussians](https://github.com/hustvl/4DGaussians) repository is checked out alongside this dataset. Copy or symlink the relevant `arguments/sceneN_*.py` file into `4DGaussians/arguments/dnerf/` before running.

### Train

```bash
cd /path/to/4DGaussians

python train.py \
    -s /path/to/4dgs-test-dataset/dnerf/scene3_collision \
    --port 6017 \
    --expname "scene3_collision" \
    --configs arguments/dnerf/scene3_collision.py \
    --iterations 30000
```

### Render (4DGS output)

```bash
python render.py \
    --model_path output/scene3_collision/ \
    --configs arguments/dnerf/scene3_collision.py
```

Rendered frames are written to `output/scene3_collision/`.

---

## 6. Generating Segmentation Masks

`scripts/generate_masks.py` generates per-pixel object-ID segmentation masks aligned with a D-NeRF formatted dataset. These masks are useful for object-level evaluation — for example, computing per-object PSNR or measuring tracking accuracy independently of background reconstruction.

### How it works

For each frame in the D-NeRF `transforms_*.json` files, the script:

1. **Recovers the simulation time** from the normalised `time` field in the transforms JSON.
2. **Advances the MuJoCo scene** to that timestamp by replaying the trajectory, placing every object at its correct position.
3. **Renders a segmentation pass** using MuJoCo's built-in segmentation renderer, which returns a per-pixel geometry ID buffer.
4. **Maps geometry IDs to integer object labels** — `0` = background/floor, `1, 2, …` = objects in the same order as `trajectory.get_object_ids()`.
5. **Saves a grayscale PNG** alongside the colour image at `{dnerf_dir}/{split}/masks/{image_name}.png`.
6. **Writes `class_mapping.json`** to the scene root, mapping integer labels to object names.

The mask pixel values match the object ordering used by the trajectory, so label `1` always corresponds to the first object in `get_object_ids()`, regardless of scene.

### Usage

```bash
# Single scene (explicit D-NeRF directory)
python scripts/generate_masks.py --scene 1 \
    --dnerf_dir dnerf_dataset/scene1_close_proximity

# Single scene (auto-resolved from dnerf_dataset/)
python scripts/generate_masks.py --scene 1

# All scenes
python scripts/generate_masks.py --all
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s / --scene` | — | Scene number (1–10) |
| `--dnerf_dir` | `dnerf_dataset/<scene_name>` | Path to the D-NeRF scene directory |
| `--dnerf_root` | `dnerf_dataset/` | Root of all D-NeRF scenes (used with `--all`) |
| `--fps` | `30` | FPS used when the dataset was generated |
| `--all` | — | Process all 10 scenes |

**Output structure:**

```
dnerf_dataset/scene1_close_proximity/
├── class_mapping.json        # { "0": "background", "1": "sphere_red", ... }
├── train/
│   ├── r_0000.png
│   └── masks/
│       ├── r_0000.png        # grayscale label image (0=bg, 1=obj1, ...)
│       └── ...
├── val/
│   └── masks/
└── test/
    └── masks/
```

---

## 7. Rendering Orbit Flyarounds

`scripts/render_orbit.py` renders a 360° orbit flyaround of a scene as a PNG sequence (and optionally an MP4). A virtual camera circles the scene centre while the scene animation plays once in sync.

### Basic usage

```bash
# Single orbit (auto radius, 180 frames at 30 fps)
python scripts/render_orbit.py --scene 1

# All 10 scenes
python scripts/render_orbit.py --all

# All scenes → compile each to MP4
python scripts/render_orbit.py --all --video

# Custom radius and elevation
python scripts/render_orbit.py --scene 3 --radius 7.0 --elevation -30

# Full 360 at 60 fps → compile to MP4
python scripts/render_orbit.py --scene 5 --num_frames 360 --fps 60 --video

# Freeze scene at a specific simulation time (static pose)
python scripts/render_orbit.py --scene 2 --freeze_time 2.5
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-s / --scene` | — | Scene number (1–10) |
| `--all` | — | Render all 10 scenes; errors are reported per-scene and do not abort the run |
| `-o / --output` | `renders` | Root output directory |
| `--radius` | auto per scene | Orbit radius in metres |
| `--elevation` | `-20` | Camera elevation in degrees (negative = looking down) |
| `--num_frames` | `180` | Total orbit frames (180 → 6 s at 30 fps) |
| `--resolution` | `800x800` | Frame resolution as `WxH` |
| `--fps` | `30` | Playback frame rate |
| `--freeze_time` | — | Lock scene at this simulation time (seconds) instead of animating |
| `--video` | off | Compile frames to MP4 via ffmpeg after rendering |

### Default orbit radii

Radii are chosen to keep the full object trajectory in frame at 800×800 resolution:

| Scene | Default radius |
|-------|:--------------:|
| 1, 2 | 5.0 m |
| 3 | 7.0 m |
| 4 | 5.5 m |
| 5 | 6.5 m |
| 6 | 6.5 m |
| 7 | 6.0 m |
| 8 | 5.5 m |
| 9 | 3.0 m |
| 10 | 5.0 m |

### Output

```
renders/
└── orbit_scene3/
    ├── 00000.png
    ├── 00001.png
    └── ...
renders/orbit_scene3.mp4   # only if --video is passed
```

---

## Requirements

```bash
pip install mujoco numpy Pillow tqdm
```

MuJoCo rendering backend is selected automatically based on platform:

| Platform | Default backend | Notes |
|----------|----------------|-------|
| macOS | `glfw` | Native OpenGL; no extra install needed |
| Linux + GPU | `egl` | Hardware-accelerated headless rendering |
| Linux, no GPU | set `MUJOCO_GL=osmesa` manually | Software rendering; `apt install libosmesa6` |

Override by setting `MUJOCO_GL` in your environment before running any script.
