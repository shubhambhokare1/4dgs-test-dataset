#!/usr/bin/env python3
"""
Generate per-pixel object-ID segmentation masks for a D-NeRF formatted dataset.

Masks are saved as grayscale PNGs alongside their corresponding images:

    {dnerf_dir}/{split}/masks/{image_name}.png

Pixel values:
    0        = background
    1, 2, … = object labels (same order as trajectory.get_object_ids())

A class_mapping.json is written to {dnerf_dir}/class_mapping.json.

Usage:
    # Single scene
    python scripts/generate_masks.py \\
        --scene 1 \\
        --dnerf_dir dnerf_dataset/scene1_close_proximity

    # All scenes (assumes dnerf_dataset/ at repo root)
    python scripts/generate_masks.py --all
"""

import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "osmesa"

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectories import get_trajectory


# --------------------------------------------------------------------------- #
# Scene metadata (mirrors dnerf_eq.py so camera cycling stays consistent)     #
# --------------------------------------------------------------------------- #
SCENE_XMLS = {
    1: "scene1_close_proximity.xml",
    2: "scene2_identical_objects.xml",
    3: "scene3_collision.xml",
    4: "scene4_occlusion.xml",
    5: "scene5_rapid_motion.xml",
    6: "scene6_scale_change.xml",
    7: "scene7_deformation.xml",
    8: "scene8_thin_structure.xml",
    9: "scene9_topology.xml",
    10: "scene10_texture.xml",
}

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


# --------------------------------------------------------------------------- #
# Generator                                                                    #
# --------------------------------------------------------------------------- #
class DNeRFMaskGenerator:
    """Generates object-ID masks aligned with a D-NeRF formatted dataset."""

    def __init__(self, scene_num: int, dnerf_dir: str, fps: int = 30):
        self.scene_num = scene_num
        self.dnerf_dir = Path(dnerf_dir)
        self.fps = fps

        if not self.dnerf_dir.exists():
            raise FileNotFoundError(f"D-NeRF directory not found: {self.dnerf_dir}")

        # ----- MuJoCo model -----
        xml_path = (
            Path(__file__).parent.parent / "scenarios" / SCENE_XMLS[scene_num]
        )
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # ----- Trajectory -----
        self.trajectory = get_trajectory(scene_num, fps=fps)
        self.duration = self.trajectory.duration

        # Frames & timing (must match dnerf_eq.py's normalize_time logic)
        self.num_frames = int(self.duration * fps)
        self.max_frame_index = max(self.num_frames - 1, 1)

        # ----- Camera names, sorted to match dnerf_eq.py -----
        self.camera_names = sorted(
            self.model.camera(i).name
            for i in range(self.model.ncam)
            if self.model.camera(i).name.startswith("cam")
        )
        self.num_cams = len(self.camera_names)
        if self.num_cams == 0:
            raise RuntimeError("No cameras starting with 'cam' found in model.")

        # ----- Object → qpos mapping (for state updates) -----
        self.object_mappings = {}  # obj_id -> qpos start index
        for obj_id in self.trajectory.get_object_ids():
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_id
                )
                jnt_id = self.model.body_jntadr[body_id]
                qpos_idx = self.model.jnt_qposadr[jnt_id]
                self.object_mappings[obj_id] = qpos_idx
            except Exception:
                print(f"  Warning: object '{obj_id}' not found in model, skipping.")

        # ----- geom_id → integer label (1-indexed) -----
        # label 0 is background; label k corresponds to object_ids[k-1]
        self.geom_to_label: dict[int, int] = {}
        self.label_to_name: dict[int, str] = {0: "background"}

        for label, obj_id in enumerate(self.trajectory.get_object_ids(), start=1):
            self.label_to_name[label] = obj_id
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_id
                )
                for geom_id in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_id] == body_id:
                        self.geom_to_label[geom_id] = label
            except Exception:
                pass

        print(f"Scene {scene_num}: {SCENE_NAMES.get(scene_num, '?')}")
        print(f"  D-NeRF dir : {self.dnerf_dir}")
        print(f"  Cameras    : {self.camera_names}")
        print(f"  Objects    : {self.trajectory.get_object_ids()}")
        print(f"  Duration   : {self.duration}s  |  fps={fps}  |  frames={self.num_frames}")

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #
    def _update_scene_state(self, sim_time: float) -> None:
        """Advance the MuJoCo simulation to *sim_time* seconds."""
        sim_time = float(np.clip(sim_time, 0.0, self.duration))
        for obj_id, qpos_idx in self.object_mappings.items():
            state = self.trajectory.get_object_state(sim_time, obj_id)
            self.data.qpos[qpos_idx : qpos_idx + 3] = state["position"]
            self.data.qpos[qpos_idx + 3 : qpos_idx + 7] = state["quaternion"]
        mujoco.mj_forward(self.model, self.data)

    def _render_mask(self, camera_name: str, H: int, W: int) -> np.ndarray:
        """Render a segmentation mask at the given camera view.

        Returns a uint8 array of shape (H, W) where each pixel value is the
        object label (0 = background).
        """
        renderer = mujoco.Renderer(self.model, height=H, width=W)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data, camera=camera_name)
        seg = renderer.render()  # (H, W, 2): [type, geom_id]; -1 = background
        renderer.close()

        geom_ids = seg[:, :, 0]  # (H, W) int array, -1 for background/sky
        mask = np.zeros((H, W), dtype=np.uint8)
        for geom_id, label in self.geom_to_label.items():
            mask[geom_ids == geom_id] = label
        return mask

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #
    def generate(self) -> None:
        """Generate masks for all splits found in dnerf_dir."""
        print("\n" + "=" * 70)
        print(f"Generating D-NeRF segmentation masks — scene {self.scene_num}")
        print("=" * 70)

        for split in ("train", "test", "val"):
            transforms_path = self.dnerf_dir / f"transforms_{split}.json"
            if not transforms_path.exists():
                print(f"  [skip] {transforms_path.name} not found")
                continue

            with open(transforms_path) as f:
                contents = json.load(f)

            frames = contents["frames"]
            if not frames:
                print(f"  [skip] {split}: no frames")
                continue

            mask_dir = self.dnerf_dir / split / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Split '{split}' — {len(frames)} frames → {mask_dir}")

            for frame in tqdm(frames, desc=f"  {split}"):
                norm_time: float = frame["time"]

                # Recover the global frame index used by dnerf_eq.py
                #   normalize_time(frame_idx, max_frame_index) = frame_idx / max_frame_index
                frame_idx = round(norm_time * self.max_frame_index)

                # Camera that dnerf_eq.py selected for this global index
                camera_name = self.camera_names[frame_idx % self.num_cams]

                # Simulation time: norm_time ∈ [0,1] maps to [0, duration]
                #   (equivalent to frame_idx/fps up to ±1 frame rounding)
                sim_time = norm_time * self.duration

                H = int(frame.get("h", 800))
                W = int(frame.get("w", 800))

                self._update_scene_state(sim_time)
                mask = self._render_mask(camera_name, H, W)

                # Mask name matches the image name (stem of file_path)
                image_stem = Path(frame["file_path"]).stem  # e.g. "r_0000"
                Image.fromarray(mask, mode="L").save(mask_dir / f"{image_stem}.png")

        # Save label → object-name mapping
        class_mapping = {str(k): v for k, v in sorted(self.label_to_name.items())}
        mapping_path = self.dnerf_dir / "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=2)

        print(f"\n  class_mapping.json → {mapping_path}")
        print("=" * 70)
        print("Done.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    repo_root = Path(__file__).parent.parent
    dnerf_root = repo_root / "dnerf_dataset"

    parser = argparse.ArgumentParser(
        description="Generate object-ID masks for D-NeRF formatted scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scene (explicit dnerf dir)
  python scripts/generate_masks.py --scene 1 \\
      --dnerf_dir dnerf_dataset/scene1_close_proximity

  # Single scene (auto-resolve from dnerf_dataset/)
  python scripts/generate_masks.py --scene 1

  # All scenes
  python scripts/generate_masks.py --all
""",
    )
    parser.add_argument("-s", "--scene", type=int, help="Scene number (1-10)")
    parser.add_argument(
        "--dnerf_dir",
        type=str,
        default=None,
        help="Path to the D-NeRF scene directory (default: dnerf_dataset/<scene_name>)",
    )
    parser.add_argument(
        "--dnerf_root",
        type=str,
        default=str(dnerf_root),
        help=f"Root of all D-NeRF scenes when using --all (default: {dnerf_root})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS used when the dataset was generated (default: 30)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all 10 scenes under --dnerf_root",
    )
    args = parser.parse_args()

    if args.all:
        scenes = list(SCENE_NAMES.keys())
    elif args.scene:
        scenes = [args.scene]
    else:
        parser.print_help()
        return

    for scene_num in scenes:
        if args.dnerf_dir and not args.all:
            dnerf_dir = args.dnerf_dir
        else:
            dnerf_dir = str(Path(args.dnerf_root) / SCENE_NAMES[scene_num])

        try:
            gen = DNeRFMaskGenerator(scene_num, dnerf_dir, fps=args.fps)
            gen.generate()
            print()
        except FileNotFoundError as e:
            print(f"  [Error] {e}")
        except Exception as e:
            import traceback

            print(f"  [Error] Scene {scene_num}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
