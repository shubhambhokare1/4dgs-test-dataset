#!/usr/bin/env python3
"""
Generate complete dataset for a scene.

Renders all frames from all cameras and saves:
- Images (PNG)
- Camera intrinsics
- Camera extrinsics per frame
- Metadata

Usage:
    python scripts/generate_dataset.py --scene 1
    python scripts/generate_dataset.py --scene 5 --resolution 1920x1080 --fps 30
    python scripts/generate_dataset.py --all
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Force headless rendering

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import mujoco
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectories import get_trajectory

# Scene metadata
SCENE_INFO = {
    1: {'name': 'scene1_close_proximity', 'xml': 'scene1_close_proximity.xml'},
    2: {'name': 'scene2_identical_objects', 'xml': 'scene2_identical_objects.xml'},
    3: {'name': 'scene3_collision', 'xml': 'scene3_collision.xml'},
    4: {'name': 'scene4_occlusion', 'xml': 'scene4_occlusion.xml'},
    5: {'name': 'scene5_rapid_motion', 'xml': 'scene5_rapid_motion.xml'},
    6: {'name': 'scene6_scale_change', 'xml': 'scene6_scale_change.xml'},
    7: {'name': 'scene7_deformation', 'xml': 'scene7_deformation.xml'},
    8: {'name': 'scene8_thin_structure', 'xml': 'scene8_thin_structure.xml'},
    9: {'name': 'scene9_topology', 'xml': 'scene9_topology.xml'},
    10: {'name': 'scene10_texture', 'xml': 'scene10_texture.xml'},
}


class DatasetGenerator:
    def __init__(self, scene_num: int, output_dir: str, resolution: tuple = (1920, 1080), fps: int = 30):
        """
        Initialize dataset generator.
        
        Args:
            scene_num: Scene number (1-10)
            output_dir: Root output directory
            resolution: (width, height) tuple
            fps: Frames per second
        """
        self.scene_num = scene_num
        self.output_dir = Path(output_dir)
        self.width, self.height = resolution
        self.fps = fps
        
        # Load scene info
        if scene_num not in SCENE_INFO:
            raise ValueError(f"Unknown scene: {scene_num}")
        
        self.scene_info = SCENE_INFO[scene_num]
        self.scene_name = self.scene_info['name']
        
        # Setup paths
        self.scene_dir = self.output_dir / f"scene{scene_num}"
        self.images_dir = self.scene_dir / "images"
        
        # Load MuJoCo model
        xml_path = Path(__file__).parent.parent / "scenarios" / self.scene_info['xml']
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")
        
        print(f"Loading scene {scene_num}: {self.scene_name}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Load trajectory
        self.trajectory = get_trajectory(scene_num, fps=fps)
        self.duration = self.trajectory.duration
        self.num_frames = int(self.duration * fps)
        
        # Get camera names from model
        self.camera_names = []
        for i in range(self.model.ncam):
            cam_name = self.model.camera(i).name
            if cam_name and cam_name.startswith('cam'):
                self.camera_names.append(cam_name)
        
        if not self.camera_names:
            raise ValueError("No cameras found in model")
        
        # Get object mappings
        self.object_mappings = {}
        for obj_id in self.trajectory.get_object_ids():
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
                jnt_id = self.model.body_jntadr[body_id]
                qpos_idx = self.model.jnt_qposadr[jnt_id]
                self.object_mappings[obj_id] = qpos_idx
            except KeyError:
                print(f"Warning: Object '{obj_id}' not found in model")
        
        print(f"  Duration: {self.duration}s")
        print(f"  FPS: {fps}")
        print(f"  Frames: {self.num_frames}")
        print(f"  Cameras: {len(self.camera_names)}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Total images: {self.num_frames * len(self.camera_names)}")
    
    def setup_directories(self):
        """Create directory structure."""
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        
        for cam in self.camera_names:
            (self.images_dir / cam).mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {self.scene_dir}")
    
    def update_scene_state(self, time: float):
        """Update all objects in the scene to given time."""
        for obj_id, qpos_idx in self.object_mappings.items():
            state = self.trajectory.get_object_state(time, obj_id)
            
            # Set position (3 values)
            self.data.qpos[qpos_idx:qpos_idx+3] = state['position']
            
            # Set orientation (4 values: quaternion)
            self.data.qpos[qpos_idx+3:qpos_idx+7] = state['quaternion']
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def render_camera(self, camera_name: str) -> np.ndarray:
        """Render a single camera view."""
        renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        renderer.update_scene(self.data, camera=camera_name)
        pixels = renderer.render()
        renderer.close()
        return pixels
    
    def get_camera_intrinsics(self, camera_name: str) -> dict:
        """Get camera intrinsic parameters."""
        cam_id = self.model.camera(camera_name).id
        fovy = self.model.camera(camera_name).fovy[0]
        
        # Calculate focal length from fovy
        focal_length_y = (self.height / 2) / np.tan(fovy / 2)
        focal_length_x = focal_length_y  # Assuming square pixels
        
        return {
            "width": self.width,
            "height": self.height,
            "fx": float(focal_length_x),
            "fy": float(focal_length_y),
            "cx": float(self.width / 2),
            "cy": float(self.height / 2),
            "fovy": float(fovy),
        }
    
    def get_camera_extrinsics(self, camera_name: str) -> np.ndarray:
        """Get camera-to-world extrinsic matrix."""
        cam_id = self.model.camera(camera_name).id
        
        # Get camera position and orientation from MuJoCo
        cam_pos = self.data.cam_xpos[cam_id].copy()
        cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        
        # MuJoCo gives us camera-to-world transform
        # cam_mat: rotation from camera frame to world frame
        # cam_pos: camera position in world coordinates
        
        # Create 4x4 camera-to-world transformation matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_mat  # Rotation
        extrinsic[:3, 3] = cam_pos   # Translation (camera position)
        
        return extrinsic
    
    def save_camera_intrinsics(self):
        """Save camera intrinsics for all cameras."""
        intrinsics = {}
        
        for cam_name in self.camera_names:
            intrinsics[cam_name] = self.get_camera_intrinsics(cam_name)
        
        intrinsics_path = self.scene_dir / "camera_intrinsics.json"
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics, f, indent=2)
        
        print(f"✓ Saved camera intrinsics: {intrinsics_path}")
    
    def generate(self):
        """Generate complete dataset."""
        print("\n" + "="*70)
        print(f"Generating Dataset: Scene {self.scene_num}")
        print("="*70)
        
        # Setup directories
        self.setup_directories()
        
        # Save camera intrinsics (constant across frames)
        self.save_camera_intrinsics()
        
        # Storage for extrinsics
        all_extrinsics = {cam: [] for cam in self.camera_names}
        
        # Generate frames
        dt = 1.0 / self.fps
        
        print("\nRendering frames...")
        for frame_idx in tqdm(range(self.num_frames), desc="Progress"):
            time = frame_idx * dt
            
            # Update scene state
            self.update_scene_state(time)
            
            # Render all cameras for this frame
            for cam_name in self.camera_names:
                # Render image
                pixels = self.render_camera(cam_name)
                
                # Save image
                img_path = self.images_dir / cam_name / f"{frame_idx:05d}.png"
                Image.fromarray(pixels).save(img_path)
                
                # Get and store extrinsics
                extrinsic = self.get_camera_extrinsics(cam_name)
                all_extrinsics[cam_name].append(extrinsic.tolist())
        
        # Save extrinsics
        extrinsics_path = self.scene_dir / "camera_extrinsics.json"
        with open(extrinsics_path, 'w') as f:
            json.dump(all_extrinsics, f, indent=2)
        
        print(f"\n✓ Saved camera extrinsics: {extrinsics_path}")
        
        # Save metadata
        metadata = {
            "scene_number": self.scene_num,
            "scene_name": self.scene_name,
            "description": self.trajectory.get_description(),
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.num_frames,
            "cameras": self.camera_names,
            "num_cameras": len(self.camera_names),
            "resolution": [self.width, self.height],
            "objects": self.trajectory.get_object_ids(),
            "trajectory_metadata": self.trajectory.get_metadata(),
        }
        
        metadata_path = self.scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Summary
        print("\n" + "="*70)
        print("Dataset Generation Complete!")
        print("="*70)
        print(f"Scene: {self.scene_num} - {self.scene_name}")
        print(f"Output: {self.scene_dir}")
        print(f"Images: {self.num_frames * len(self.camera_names)}")
        print(f"Size: ~{self.estimate_size_mb():.1f} MB")
        print("="*70)
    
    def estimate_size_mb(self) -> float:
        """Estimate dataset size in MB."""
        # Rough estimate: 1920x1080 PNG ≈ 1-2 MB per image
        bytes_per_image = self.width * self.height * 3 * 0.3  # Assuming 30% compression
        total_bytes = bytes_per_image * self.num_frames * len(self.camera_names)
        return total_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset for 4DGS testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_dataset.py --scene 1                    # Generate scene 1
  python scripts/generate_dataset.py --scene 5 --fps 60           # Scene 5 at 60 FPS
  python scripts/generate_dataset.py --scene 1 --resolution 1280x720  # Lower resolution
  python scripts/generate_dataset.py --all                        # Generate all scenes
        """
    )
    
    parser.add_argument('-s', '--scene', type=int,
                       help='Scene number to generate (1-10)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all scenes')
    parser.add_argument('-o', '--output', type=str, default='dataset',
                       help='Output directory (default: dataset)')
    parser.add_argument('--resolution', type=str, default='1920x1080',
                       help='Resolution as WIDTHxHEIGHT (default: 1920x1080)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Error: Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1920x1080)")
        return
    
    # Determine which scenes to generate
    if args.all:
        scenes = list(range(1, 11))
        print(f"Generating all {len(scenes)} scenes...")
    elif args.scene:
        if args.scene not in SCENE_INFO:
            print(f"Error: Unknown scene: {args.scene}")
            print(f"Valid scenes: {list(SCENE_INFO.keys())}")
            return
        scenes = [args.scene]
    else:
        parser.print_help()
        print("\nError: Must specify --scene or --all")
        return
    
    # Generate datasets
    for scene_num in scenes:
        try:
            generator = DatasetGenerator(
                scene_num=scene_num,
                output_dir=args.output,
                resolution=resolution,
                fps=args.fps
            )
            generator.generate()
            print()
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user")
            break
        except Exception as e:
            print(f"\nError generating scene {scene_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("All done!")


if __name__ == "__main__":
    main()