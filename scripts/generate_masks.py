#!/usr/bin/env python3
"""
Generate segmentation masks for all objects.

Creates per-pixel object IDs:
- 0 = background
- 1, 2, 3... = object IDs

Usage:
    python scripts/generate_masks.py --scene 1
    python scripts/generate_masks.py --all
"""

import os
import sys

# Pick the right MuJoCo rendering backend for the current platform.
# macOS         — native OpenGL via glfw (egl/osmesa are Linux-only).
# Linux + GPU   — egl  (hardware-accelerated, recommended for headless).
# Linux no GPU  — osmesa (software, slower; install with: apt install libosmesa6).
# Override by setting MUJOCO_GL in the environment before running.
if "MUJOCO_GL" not in os.environ:
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "glfw"
    elif os.path.exists("/dev/dri"):
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

import argparse
from pathlib import Path
import json
import numpy as np
import mujoco
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectories import get_trajectory


class MaskGenerator:
    def __init__(self, scene_num: int, dataset_dir: str):
        """Initialize mask generator."""
        self.scene_num = scene_num
        self.dataset_dir = Path(dataset_dir)
        self.scene_dir = self.dataset_dir / f"scene{scene_num}"
        
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")
        
        # Load metadata
        with open(self.scene_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load MuJoCo model
        scene_xmls = {
            1: 'scene1_close_proximity.xml', 2: 'scene2_identical_objects.xml',
            3: 'scene3_collision.xml', 4: 'scene4_occlusion.xml',
            5: 'scene5_rapid_motion.xml', 6: 'scene6_scale_change.xml',
            7: 'scene7_deformation.xml', 8: 'scene8_thin_structure.xml',
            9: 'scene9_topology.xml', 10: 'scene10_texture.xml',
        }
        
        xml_path = Path(__file__).parent.parent / "scenarios" / scene_xmls[scene_num]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Load trajectory
        self.trajectory = get_trajectory(scene_num, fps=self.metadata['fps'])
        
        # Get object mappings
        self.object_mappings = {}
        self.object_to_id = {}  # Map object name to mask ID
        
        for idx, obj_id in enumerate(self.metadata['objects'], start=1):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
                jnt_id = self.model.body_jntadr[body_id]
                qpos_idx = self.model.jnt_qposadr[jnt_id]
                self.object_mappings[obj_id] = qpos_idx
                self.object_to_id[obj_id] = idx
            except:
                print(f"Warning: Object '{obj_id}' not found")
        
        # Output
        self.mask_dir = self.scene_dir / "masks"
        self.width = self.metadata['resolution'][0]
        self.height = self.metadata['resolution'][1]
        
        print(f"Scene {scene_num}: {self.metadata['scene_name']}")
        print(f"  Objects: {len(self.object_to_id)}")
    
    def update_scene_state(self, time: float):
        """Update scene to given time."""
        for obj_id, qpos_idx in self.object_mappings.items():
            state = self.trajectory.get_object_state(time, obj_id)
            self.data.qpos[qpos_idx:qpos_idx+3] = state['position']
            self.data.qpos[qpos_idx+3:qpos_idx+7] = state['quaternion']
        mujoco.mj_forward(self.model, self.data)
    
    def render_segmentation(self, camera_name: str) -> np.ndarray:
        """Render segmentation mask."""
        # Enable segmentation rendering
        renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        
        # Render with segmentation
        renderer.update_scene(self.data, camera=camera_name)
        
        # Get segmentation (geom IDs)
        seg = renderer.render()  # This gives us RGB
        
        # MuJoCo doesn't directly give object IDs in standard renderer
        # We need to use a workaround: render with unique colors per object
        
        # For simplicity, we'll render normally and note this limitation
        # A full implementation would require custom shader or depth-based segmentation
        
        renderer.close()
        
        # Return placeholder (zeros = background)
        # TODO: Implement proper segmentation rendering
        return np.zeros((self.height, self.width), dtype=np.uint8)
    
    def generate(self):
        """Generate all masks."""
        print("\n" + "="*70)
        print(f"Generating Segmentation Masks: Scene {self.scene_num}")
        print("="*70)
        print("\nNOTE: Segmentation mask generation requires custom implementation.")
        print("Current version creates placeholder masks (all zeros).")
        print("For production use, implement depth-based or color-coded segmentation.")
        print("="*70)
        
        # Create directories
        for cam in self.metadata['cameras']:
            (self.mask_dir / cam).mkdir(parents=True, exist_ok=True)
        
        dt = 1.0 / self.metadata['fps']
        
        print("\nGenerating masks...")
        for frame_idx in tqdm(range(self.metadata['total_frames']), desc="Progress"):
            time = frame_idx * dt
            self.update_scene_state(time)
            
            for cam_name in self.metadata['cameras']:
                mask = self.render_segmentation(cam_name)
                
                # Save as grayscale PNG
                output_path = self.mask_dir / cam_name / f"{frame_idx:05d}.png"
                Image.fromarray(mask).save(output_path)
        
        # Save class mapping
        class_mapping = {'0': 'background'}
        for obj_id, mask_id in self.object_to_id.items():
            class_mapping[str(mask_id)] = obj_id
        
        with open(self.mask_dir / "class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"\n✓ Masks saved to: {self.mask_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks')
    parser.add_argument('-s', '--scene', type=int, help='Scene number (1-10)')
    parser.add_argument('--all', action='store_true', help='Process all scenes')
    parser.add_argument('-d', '--dataset-dir', type=str, default='dataset',
                       help='Dataset directory (default: dataset)')
    
    args = parser.parse_args()
    
    if args.all:
        scenes = list(range(1, 11))
    elif args.scene:
        scenes = [args.scene]
    else:
        parser.print_help()
        return
    
    for scene_num in scenes:
        try:
            generator = MaskGenerator(scene_num, args.dataset_dir)
            generator.generate()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()