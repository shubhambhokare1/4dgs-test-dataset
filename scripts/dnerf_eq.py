import os
import json
import random
import shutil
import argparse
from pathlib import Path


def normalize_time(frame_idx, max_frame):
    return frame_idx / max_frame * 5


def main(dataset_path, intrinsics_path, extrinsics_path,
         output_path, num_frames=150):

    random.seed(42)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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

        transforms = {
            "frames": frames_json
        }

        with open(output_path / f"transforms_{split_name}.json", "w") as f:
            json.dump(transforms, f, indent=4)

    print("\nDataset successfully created.")
    print(f"Location: {output_path}")
    print("Contains: train/, val/, test/ and transforms_*.json files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--extrinsics", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_frames", type=int, default=150)

    args = parser.parse_args()

    main(
        args.dataset_path,
        args.intrinsics,
        args.extrinsics,
        args.output_path,
        args.num_frames
    )