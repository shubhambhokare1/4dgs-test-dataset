# 4DGS Synthetic Test Dataset

A synthetic benchmark dataset for evaluating **4D Gaussian Splatting (4DGS)** methods on controlled dynamic scenes. Each scene is rendered from 12 calibrated camera views using MuJoCo physics simulation and targets a specific known limitation of current 4D scene reconstruction methods.

All scenes use a white floor and white skybox to isolate object appearance from background, minimising confounding factors in evaluation.

---

## Scenes

| # | Scene | File | 4DGS Deficiency Targeted | Complexity Score | Rank |
|---|-------|------|--------------------------|:---:|:---:|
| 1 | Close Proximity — Different Colors | `scene1_close_proximity` | Gaussian boundary preservation; identity maintenance when two differently-coloured objects pass within touching distance | 0.166 | 8 |
| 2 | Close Proximity — Identical Appearance | `scene2_identical_objects` | Identity tracking without appearance cues; two visually identical spheres cross paths | 0.176 | 6 |
| 3 | Three-Body Collision | `scene3_collision` | Multi-object interaction; chaotic bounce trajectories; temporal consistency before/during/after collision | 0.354 | 2 |
| 4 | Occlusion & Dis-occlusion | `scene4_occlusion` | Hallucination behind an occluder; correct reconstruction when objects reappear | 0.245 | 4 |
| 5 | Rapid Direction Changes | `scene5_rapid_motion` | Motion smoothness assumptions; temporal interpolation failure at sharp 90° turns | 0.171 | 7 |
| 6 | Extreme Scale Change | `scene6_scale_change` | Multi-scale representation; level-of-detail adaptation as object moves far-to-near | 0.117 | 10 |
| 7 | Deformable vs Rigid Collision | `scene7_deformation` | Non-rigid deformation capture; appearance change at contact; collision dynamics | 0.308 | 3 |
| 8 | Thin Structure Tracking | `scene8_thin_structure` | Thin geometry preservation; anisotropic Gaussian scaling for rod-like objects | 0.226 | 5 |
| 9 | Topology Change — Split & Merge | `scene9_topology` | Topology change handling; fixed Gaussian count assumption; split and merge events | 0.399 | 1 |
| 10 | High-Frequency Texture + Motion | `scene10_texture` | Fine texture preservation under motion; appearance vs geometry entanglement | 0.125 | 9 |

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

```bash
python scripts/dnerf_eq.py \
    --dataset_path  dataset/scene3/images \
    --intrinsics    dataset/scene3/camera_intrinsics.json \
    --extrinsics    dataset/scene3/camera_extrinsics.json \
    --output_path   dnerf/scene3_collision \
    --num_frames    150
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | — | Path to the `images/` directory inside the scene folder |
| `--intrinsics` | — | Path to `camera_intrinsics.json` |
| `--extrinsics` | — | Path to `camera_extrinsics.json` |
| `--output_path` | — | Destination directory for the D-NeRF dataset |
| `--num_frames` | `150` | Number of (camera, frame) pairs to sample |

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
| `scene3_collision.py` | 150 | 2.5 | Spheres start at radius 2.5 from origin |
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

### Render

```bash
python render.py \
    -s /path/to/4dgs-test-dataset/dnerf/scene3_collision \
    --expname "scene3_collision" \
    --configs arguments/dnerf/scene3_collision.py
```

Rendered frames are written to `output/scene3_collision/`.

### Evaluate

```bash
python metrics.py \
    -s /path/to/4dgs-test-dataset/dnerf/scene3_collision \
    --expname "scene3_collision"
```

Reports PSNR, SSIM, and LPIPS on the test split.

---

## 6. Scene Complexity Scoring

Each scene is assigned a complexity score **C ∈ [0, 1]** that quantifies how difficult it is expected to be for 4DGS-class reconstruction methods. The score is a weighted sum of ten scene-level factors, each independently rated on a [0, 1] scale:

### Factors and weights

| Factor | Weight | Description |
|--------|:------:|-------------|
| `motion_speed` | 0.12 | Peak object speed relative to scene bounds. Fast motion violates temporal-continuity priors in k-planes. |
| `num_objects` | 0.08 | Number of independently moving objects (normalised; 3+ = 1.0). More objects require more Gaussians. |
| `occlusion` | 0.13 | Degree of complete or prolonged occlusion. Full occlusion causes hallucination and dis-occlusion artefacts. |
| `proximity` | 0.10 | Closest inter-object separation normalised by object diameter (touching = 1.0). Nearby Gaussians blur across boundaries. |
| `deformation` | 0.13 | Presence and magnitude of non-rigid shape change. 4DGS assumes near-rigid deformation fields. |
| `topology` | 0.18 | Presence of topology change events (split / merge). Fixed Gaussian count cannot represent object count changes. |
| `thin_geometry` | 0.09 | Aspect ratio of the thinnest object (rod radius / half-length). Very thin structures need many tiny anisotropic Gaussians. |
| `scale_variation` | 0.05 | Ratio of apparent object size at closest vs farthest point. Extreme scale changes challenge level-of-detail representation. |
| `direction_changes` | 0.07 | Number and sharpness of direction reversals. Sharp turns violate smooth motion priors in the deformation field. |
| `texture_complexity` | 0.05 | High-frequency surface texture under rotation/translation. Appearance and geometry become entangled in Gaussian features. |

### Formula

$$C = \sum_{i=1}^{10} w_i \cdot f_i$$

where $f_i \in [0,1]$ is the factor score and $w_i$ is the factor weight, with $\sum w_i = 1$.

In plain text:

```
C = 0.12·motion_speed + 0.08·num_objects + 0.13·occlusion + 0.10·proximity
  + 0.13·deformation + 0.18·topology + 0.09·thin_geometry + 0.05·scale_variation
  + 0.07·direction_changes + 0.05·texture_complexity
```

### Scores and ranks

| Rank | Scene | Score | Primary driver(s) |
|:----:|-------|:-----:|-------------------|
| 1 | Scene 9 — Topology Change | 0.399 | topology (0.18 × 1.0) + proximity + deformation |
| 2 | Scene 3 — Three-Body Collision | 0.354 | motion_speed + num_objects + proximity + occlusion |
| 3 | Scene 7 — Deformable Collision | 0.308 | deformation (0.13 × 1.0) + proximity |
| 4 | Scene 4 — Occlusion | 0.245 | occlusion (0.13 × 1.0) |
| 5 | Scene 8 — Thin Structures | 0.226 | thin_geometry (0.09 × 1.0) + motion_speed |
| 6 | Scene 2 — Identical Objects | 0.176 | proximity (0.10 × 1.0) |
| 7 | Scene 5 — Rapid Motion | 0.171 | direction_changes (0.07 × 1.0) + motion_speed |
| 8 | Scene 1 — Close Proximity | 0.166 | proximity + motion_speed |
| 9 | Scene 10 — Texture + Motion | 0.125 | texture_complexity (0.05 × 1.0) |
| 10 | Scene 6 — Scale Change | 0.117 | scale_variation (0.05 × 1.0) |

The complexity scores and ranks are computed programmatically in `scripts/metrics_analysis.py`. Run `--complexity_only` to print the table without generating plots.

---

## 7. Metrics Analysis and Visualisation

`scripts/metrics_analysis.py` ingests per-iteration training metrics from three reconstruction methods and generates two families of plots.

### Input format

Place one JSON file per scene per method under a `metrics/` directory:

```
metrics/
├── 3dgs/        scene1.json  …  scene10.json
├── 4dgs/        scene1.json  …  scene10.json
└── deformable/  scene1.json  …  scene10.json
```

Each JSON file should be a JSON array (or NDJSON) of checkpoint records:

```json
[
    {"iteration": 1000,  "psnr": 25.3, "ssim": 0.85, "lpips": 0.12},
    {"iteration": 7000,  "psnr": 28.1, "ssim": 0.91, "lpips": 0.08},
    {"iteration": 30000, "psnr": 30.5, "ssim": 0.94, "lpips": 0.05}
]
```

If a method uses different field names, fill in the corresponding `parse_3dgs` / `parse_4dgs` / `parse_deformable` placeholder function in the script.

### Plot types

**Type 1 — Training curves** (10 figures × 3 subplots = 30 plots)

One figure per scene, three subplots (PSNR / SSIM / LPIPS), each showing all three methods as lines over training iterations. Saved to `figures/training_curves/scene{N}_training_curves.png`.

**Type 2 — Eval vs complexity** (3 figures)

One figure per metric; x-axis is scene complexity rank (rank 1 = most complex), y-axis is the final recorded metric value. Shows how each method degrades as scene difficulty increases. Saved to `figures/eval_vs_complexity/eval_{metric}.png`.

### Usage

```bash
# Generate all plots (expects metrics/ directory in cwd)
python scripts/metrics_analysis.py

# Custom paths
python scripts/metrics_analysis.py \
    --metrics_dir path/to/metrics \
    --output_dir  path/to/figures

# Print complexity table only — no plots
python scripts/metrics_analysis.py --complexity_only
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--metrics_dir` | `metrics` | Root directory containing `3dgs/`, `4dgs/`, `deformable/` sub-dirs |
| `--output_dir` | `figures` | Directory to write generated PNG files |
| `--complexity_only` | — | Print complexity table and exit without generating plots |

```bash
pip install matplotlib numpy
```

---

## Requirements

```bash
pip install mujoco numpy Pillow tqdm
```

MuJoCo rendering requires either an EGL-capable GPU or OSMesa for headless environments. `generate_dataset.py` sets `MUJOCO_GL=osmesa` automatically.
