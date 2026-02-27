import os
import json
import math
import random
import shutil
import argparse
import numpy as np
from pathlib import Path


def write_fused_ply(output_path, n_points=100000, bounds=2.5):
    """
    Write a fused.ply point cloud covering [-bounds, bounds]^3.
    4DGaussians uses this for Gaussian initialization when the file exists,
    bypassing its default narrow [-1.3, 1.3] random cloud.
    """
    np.random.seed(42)
    xyz = np.random.uniform(-bounds, bounds, (n_points, 3)).astype(np.float32)
    rgb = np.random.randint(0, 256, (n_points, 3), dtype=np.uint8)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    ply_path = output_path / "fused.ply"
    with open(ply_path, 'wb') as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n_points}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))
        float_data = np.hstack([xyz, normals])  # (n, 6) float32
        for i in range(n_points):
            f.write(float_data[i].tobytes() + rgb[i].tobytes())
    print(f"  Generated fused.ply ({n_points} points, bounds=±{bounds})")


def normalize_time(frame_idx, max_frame):
    return frame_idx / max_frame


def main(dataset_path, intrinsics_path, extrinsics_path,
         output_path, num_frames=150, bounds=2.5):

    random.seed(42)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    write_fused_ply(output_path, bounds=bounds)

    # Load metadata
    with open(intrinsics_path) as f:
        intr = json.load(f)   # {cam: {...}}

    with open(extrinsics_path) as f:
        extr = json.load(f)   # {cam: [4x4, 4x4, ...]}

    cams = sorted(intr.keys())

    # ---- Get frame list from first camera ----
    first_cam_path = dataset_path / cams[0]
    frame_files = sorted(
        [f for f in os.listdir(first_cam_path) if f.endswith(".png")]
    )

    num_available_frames = len(frame_files)
    max_frame_index = num_available_frames - 1

    if num_available_frames == 0:
        raise ValueError("No PNG files found in dataset.")

    # ---- Sample timestamps (by index, not filename number) ----
    frame_indices = list(range(num_available_frames))
    selected_indices = random.sample(
        frame_indices,
        min(num_frames, num_available_frames)
    )

    # Pick one random camera per timestamp
    selected_pairs = []
    for idx in selected_indices:
        cam = random.choice(cams)
        selected_pairs.append((cam, idx))

    random.shuffle(selected_pairs)

    # ---- Split 70 / 15 / 15 ----
    n = len(selected_pairs)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "train": selected_pairs[:train_end],
        "val": selected_pairs[train_end:val_end],
        "test": selected_pairs[val_end:]
    }

    # ---- Process splits ----
    for split_name, items in splits.items():

        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        frames_json = []

        for new_idx, (cam, frame_idx) in enumerate(items):

            # Source image (index-aligned)
            src = dataset_path / cam / frame_files[frame_idx]
            new_name = f"r_{new_idx:04d}.png"
            dst = split_dir / new_name

            shutil.copy(src, dst)

            # Intrinsics
            cam_intr = intr[cam]

            # Extrinsics (index-aligned list)
            cam_extr = extr[cam][frame_idx]

            frames_json.append({
                "file_path": f"./{split_name}/r_{new_idx:04d}",
                "rotation": 0.0,
                "time": normalize_time(frame_idx, max_frame_index),

                "fl_x": cam_intr["fx"],
                "fl_y": cam_intr["fy"],
                "cx": cam_intr["cx"],
                "cy": cam_intr["cy"],
                "w": cam_intr["width"],
                "h": cam_intr["height"],

                "transform_matrix": cam_extr
            })

        cam_intr = intr[cams[0]]
        camera_angle_x = 2 * math.atan(cam_intr["width"] / (2 * cam_intr["fx"]))

        transforms = {
            "camera_angle_x": camera_angle_x,
            "frames": frames_json
        }

        with open(output_path / f"transforms_{split_name}.json", "w") as f:
            json.dump(transforms, f, indent=4)

    print("\nDataset successfully created.")
    print(f"Location: {output_path}")
    print("Contains: train/, val/, test/ and transforms_*.json files")


SCENE_NAMES = {
    1: "scene1_close_proximity",
    2: "scene2_identical_objects",
    3: "scene3_collision",
    4: "scene4_occlusion",
    5: "scene5_rapid_motion",
    6: "scene6_scale_change",
    7: "scene7_deformation",
    8: "scene8_thin_structure",
    9: "scene9_topology",
    10: "scene10_texture",
}

SCENE_NUM_FRAMES = {
    1: 150, 2: 150, 3: 150, 4: 150, 5: 180,
    6: 180, 7: 180, 8: 150, 9: 180, 10: 150,
}

SCENE_BOUNDS = {
    1: 2.0, 2: 2.0, 3: 2.5, 4: 2.5, 5: 2.5,
    6: 3.5, 7: 3.5, 8: 2.5, 9: 1.5, 10: 2.0,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Per-scene mode
    parser.add_argument("--dataset_path")
    parser.add_argument("--intrinsics")
    parser.add_argument("--extrinsics")
    parser.add_argument("--output_path")
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--bounds", type=float, default=2.5,
                        help="Spatial bounds for fused.ply (±bounds in each axis)")

    # Batch mode
    parser.add_argument("--all", action="store_true",
                        help="Process all 10 scenes automatically")
    parser.add_argument("--dataset_dir", default="dataset",
                        help="Root directory containing per-scene folders (used with --all)")
    parser.add_argument("--output_dir", default="dnerf",
                        help="Root output directory for D-NeRF data (used with --all)")

    args = parser.parse_args()

    if args.all:
        dataset_dir = Path(args.dataset_dir)
        output_dir = Path(args.output_dir)
        for scene_num, scene_name in SCENE_NAMES.items():
            print(f"\n=== Processing {scene_name} ===")
            scene_dir = dataset_dir / f"scene{scene_num}"
            main(
                dataset_path=scene_dir / "images",
                intrinsics_path=scene_dir / "camera_intrinsics.json",
                extrinsics_path=scene_dir / "camera_extrinsics.json",
                output_path=output_dir / scene_name,
                num_frames=SCENE_NUM_FRAMES[scene_num],
                bounds=SCENE_BOUNDS[scene_num],
            )
    else:
        if not all([args.dataset_path, args.intrinsics, args.extrinsics, args.output_path]):
            parser.error("--dataset_path, --intrinsics, --extrinsics, and --output_path "
                         "are required when not using --all")
        main(
            args.dataset_path,
            args.intrinsics,
            args.extrinsics,
            args.output_path,
            args.num_frames,
            args.bounds,
        )