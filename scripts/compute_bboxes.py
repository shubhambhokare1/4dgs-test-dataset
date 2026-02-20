#!/usr/bin/env python3
"""
Compute bounding boxes for all objects in all frames.

Generates:
- 2D bounding boxes (pixel coordinates)
- 3D bounding boxes (world coordinates)
- Per-frame JSON files with bbox data

Usage:
    python scripts/compute_bboxes.py --scene 1
    python scripts/compute_bboxes.py --all
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import mujoco
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectories import get_trajectory


def project_point_to_camera(point_3d, cam_pos, cam_mat, intrinsics):
    """
    Project 3D point to 2D camera coordinates.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        cam_pos: Camera position in world
        cam_mat: Camera rotation matrix (3x3)
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
    
    Returns:
        [u, v] pixel coordinates, or None if behind camera
    """
    # Transform to camera coordinates
    point_cam = cam_mat.T @ (point_3d - cam_pos)
    
    # Check if behind camera
    if point_cam[2] <= 0:
        return None
    
    # Project to image plane
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]
    
    # Apply intrinsics
    u = intrinsics['fx'] * x + intrinsics['cx']
    v = intrinsics['fy'] * y + intrinsics['cy']
    
    return np.array([u, v])


def compute_sphere_bbox_2d(center_3d, radius, cam_pos, cam_mat, intrinsics, img_width, img_height):
    """
    Compute 2D bounding box for a sphere.
    
    Returns:
        [x_min, y_min, x_max, y_max] in pixel coordinates, or None if not visible
    """
    # Project sphere center
    center_2d = project_point_to_camera(center_3d, cam_pos, cam_mat, intrinsics)
    
    if center_2d is None:
        return None
    
    # Compute distance from camera
    cam_to_sphere = np.linalg.norm(center_3d - cam_pos)
    
    # Angular size of sphere
    angular_radius = np.arctan(radius / cam_to_sphere)
    
    # Project to pixel radius (approximate)
    pixel_radius = angular_radius * intrinsics['fx']
    
    # Bounding box
    x_min = max(0, center_2d[0] - pixel_radius)
    y_min = max(0, center_2d[1] - pixel_radius)
    x_max = min(img_width, center_2d[0] + pixel_radius)
    y_max = min(img_height, center_2d[1] + pixel_radius)
    
    # Check if visible
    if x_max <= 0 or y_max <= 0 or x_min >= img_width or y_min >= img_height:
        return None
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def compute_box_bbox_2d(center_3d, half_size, cam_pos, cam_mat, intrinsics, img_width, img_height):
    """
    Compute 2D bounding box for a box (cube).
    
    Args:
        half_size: Half-size of the cube (scalar or [x, y, z])
    """
    if isinstance(half_size, (int, float)):
        half_size = np.array([half_size, half_size, half_size])
    
    # 8 corners of the box
    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corner = center_3d + np.array([dx, dy, dz]) * half_size
                corners.append(corner)
    
    # Project all corners
    projected = []
    for corner in corners:
        p = project_point_to_camera(corner, cam_pos, cam_mat, intrinsics)
        if p is not None:
            projected.append(p)
    
    if len(projected) == 0:
        return None
    
    projected = np.array(projected)
    
    # Compute bounding box
    x_min = max(0, projected[:, 0].min())
    y_min = max(0, projected[:, 1].min())
    x_max = min(img_width, projected[:, 0].max())
    y_max = min(img_height, projected[:, 1].max())
    
    if x_max <= 0 or y_max <= 0 or x_min >= img_width or y_min >= img_height:
        return None
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def compute_cylinder_bbox_2d(center_3d, radius, half_height, quaternion, cam_pos, cam_mat, intrinsics, img_width, img_height):
    """
    Compute 2D bounding box for a cylinder (approximated by sampling points).
    
    Args:
        radius: Cylinder radius
        half_height: Half-height of cylinder
        quaternion: Orientation [qw, qx, qy, qz]
    """
    # Convert quaternion to rotation matrix
    qw, qx, qy, qz = quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    # Sample points on cylinder surface
    points = []
    
    # Top and bottom circles
    for z in [-half_height, half_height]:
        for angle in np.linspace(0, 2*np.pi, 16):
            local_point = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
            world_point = center_3d + R @ local_point
            points.append(world_point)
    
    # Side edges
    for angle in np.linspace(0, 2*np.pi, 8):
        for z in np.linspace(-half_height, half_height, 4):
            local_point = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
            world_point = center_3d + R @ local_point
            points.append(world_point)
    
    # Project all points
    projected = []
    for point in points:
        p = project_point_to_camera(point, cam_pos, cam_mat, intrinsics)
        if p is not None:
            projected.append(p)
    
    if len(projected) == 0:
        return None
    
    projected = np.array(projected)
    
    # Compute bounding box
    x_min = max(0, projected[:, 0].min())
    y_min = max(0, projected[:, 1].min())
    x_max = min(img_width, projected[:, 0].max())
    y_max = min(img_height, projected[:, 1].max())
    
    if x_max <= 0 or y_max <= 0 or x_min >= img_width or y_min >= img_height:
        return None
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


class BBoxComputer:
    def __init__(self, scene_num: int, dataset_dir: str):
        """
        Initialize bounding box computer.
        
        Args:
            scene_num: Scene number (1-10)
            dataset_dir: Root dataset directory
        """
        self.scene_num = scene_num
        self.dataset_dir = Path(dataset_dir)
        self.scene_dir = self.dataset_dir / f"scene{scene_num}"
        
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")
        
        # Load metadata
        with open(self.scene_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load intrinsics
        with open(self.scene_dir / "camera_intrinsics.json") as f:
            self.intrinsics = json.load(f)
        
        # Load extrinsics
        with open(self.scene_dir / "camera_extrinsics.json") as f:
            self.extrinsics = json.load(f)
        
        # Load trajectory
        self.trajectory = get_trajectory(scene_num, fps=self.metadata['fps'])
        
        # Get object info from MuJoCo model
        self._load_object_info()
        
        # Output directory
        self.bbox_dir = self.scene_dir / "bboxes"
        
        print(f"Scene {scene_num}: {self.metadata['scene_name']}")
        print(f"  Frames: {self.metadata['total_frames']}")
        print(f"  Cameras: {len(self.metadata['cameras'])}")
        print(f"  Objects: {len(self.metadata['objects'])}")
    
    def _load_object_info(self):
        """Load object geometry info from MuJoCo model."""
        from trajectories import TRAJECTORY_MAP
        
        # Load model to get geometry info
        scene_info = {
            1: 'scene1_close_proximity.xml',
            2: 'scene2_identical_objects.xml',
            3: 'scene3_collision.xml',
            4: 'scene4_occlusion.xml',
            5: 'scene5_rapid_motion.xml',
            6: 'scene6_scale_change.xml',
            7: 'scene7_deformation.xml',
            8: 'scene8_thin_structure.xml',
            9: 'scene9_topology.xml',
            10: 'scene10_texture.xml',
        }
        
        xml_path = Path(__file__).parent.parent / "scenarios" / scene_info[self.scene_num]
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        
        # Extract geometry info for each object
        self.object_info = {}
        
        for obj_id in self.metadata['objects']:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
                
                # Get first geom associated with this body
                geom_start = model.body_geomadr[body_id]
                geom_num = model.body_geomnum[body_id]
                
                if geom_num > 0:
                    geom_id = geom_start
                    geom_type = model.geom_type[geom_id]
                    geom_size = model.geom_size[geom_id].copy()
                    geom_rgba = model.geom_rgba[geom_id].copy()
                    
                    # Map geom type
                    type_map = {
                        0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule',
                        4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'
                    }
                    
                    self.object_info[obj_id] = {
                        'type': type_map.get(geom_type, 'unknown'),
                        'size': geom_size.tolist(),
                        'color': geom_rgba[:3].tolist(),
                    }
            except:
                print(f"Warning: Could not get geometry info for {obj_id}")
                self.object_info[obj_id] = {'type': 'unknown', 'size': [0.1], 'color': [0.5, 0.5, 0.5]}
    
    def compute_bbox_for_object(self, obj_id, state, camera_name, frame_idx):
        """Compute 2D and 3D bounding box for an object."""
        obj_info = self.object_info[obj_id]
        position = state['position']
        quaternion = state['quaternion']
        
        # Get camera parameters
        intrinsics = self.intrinsics[camera_name]
        extrinsic = np.array(self.extrinsics[camera_name][frame_idx])
        
        # Extract camera position and rotation
        cam_rot = extrinsic[:3, :3].T
        cam_pos = -cam_rot @ extrinsic[:3, 3]
        
        # Compute 2D bbox based on geometry type
        if obj_info['type'] == 'sphere':
            radius = obj_info['size'][0]
            bbox_2d = compute_sphere_bbox_2d(
                position, radius, cam_pos, cam_rot, intrinsics,
                intrinsics['width'], intrinsics['height']
            )
        elif obj_info['type'] == 'box':
            half_size = np.array(obj_info['size'])
            bbox_2d = compute_box_bbox_2d(
                position, half_size, cam_pos, cam_rot, intrinsics,
                intrinsics['width'], intrinsics['height']
            )
        elif obj_info['type'] == 'cylinder':
            radius = obj_info['size'][0]
            half_height = obj_info['size'][1]
            bbox_2d = compute_cylinder_bbox_2d(
                position, radius, half_height, quaternion,
                cam_pos, cam_rot, intrinsics,
                intrinsics['width'], intrinsics['height']
            )
        else:
            bbox_2d = None
        
        # 3D bounding box (in world coordinates)
        if obj_info['type'] == 'sphere':
            radius = obj_info['size'][0]
            bbox_3d = {
                'center': position.tolist(),
                'size': [radius*2, radius*2, radius*2],
                'rotation': quaternion.tolist()
            }
        elif obj_info['type'] == 'box':
            size = (np.array(obj_info['size']) * 2).tolist()
            bbox_3d = {
                'center': position.tolist(),
                'size': size,
                'rotation': quaternion.tolist()
            }
        elif obj_info['type'] == 'cylinder':
            radius = obj_info['size'][0]
            height = obj_info['size'][1] * 2
            bbox_3d = {
                'center': position.tolist(),
                'size': [radius*2, radius*2, height],
                'rotation': quaternion.tolist()
            }
        else:
            bbox_3d = None
        
        return bbox_2d, bbox_3d
    
    def compute_all(self):
        """Compute bounding boxes for all frames and cameras."""
        print("\n" + "="*70)
        print(f"Computing Bounding Boxes: Scene {self.scene_num}")
        print("="*70)
        
        # Create output directories
        for cam in self.metadata['cameras']:
            (self.bbox_dir / cam).mkdir(parents=True, exist_ok=True)
        
        dt = 1.0 / self.metadata['fps']
        
        print("\nProcessing frames...")
        for frame_idx in tqdm(range(self.metadata['total_frames']), desc="Progress"):
            time = frame_idx * dt
            
            # Get all object states at this time
            states = {}
            for obj_id in self.metadata['objects']:
                states[obj_id] = self.trajectory.get_object_state(time, obj_id)
            
            # Compute bboxes for each camera
            for cam_name in self.metadata['cameras']:
                objects_data = []
                
                for obj_id, state in states.items():
                    bbox_2d, bbox_3d = self.compute_bbox_for_object(
                        obj_id, state, cam_name, frame_idx
                    )
                    
                    obj_data = {
                        'id': obj_id,
                        'type': self.object_info[obj_id]['type'],
                        'color': self.object_info[obj_id]['color'],
                        'bbox_2d': bbox_2d,
                        'bbox_3d': bbox_3d,
                        'visible': bbox_2d is not None,
                    }
                    
                    objects_data.append(obj_data)
                
                # Save frame data
                frame_data = {
                    'frame_id': frame_idx,
                    'time': float(time),
                    'camera': cam_name,
                    'objects': objects_data
                }
                
                output_path = self.bbox_dir / cam_name / f"{frame_idx:05d}.json"
                with open(output_path, 'w') as f:
                    json.dump(frame_data, f, indent=2)
        
        # Save metadata
        bbox_metadata = {
            'scene': self.metadata['scene_name'],
            'objects': self.object_info,
        }
        
        with open(self.bbox_dir / "metadata.json", 'w') as f:
            json.dump(bbox_metadata, f, indent=2)
        
        print(f"\n✓ Saved bounding boxes to: {self.bbox_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Compute bounding boxes for dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('-s', '--scene', type=int,
                       help='Scene number (1-10)')
    parser.add_argument('--all', action='store_true',
                       help='Process all scenes')
    parser.add_argument('-d', '--dataset-dir', type=str, default='dataset',
                       help='Dataset directory (default: dataset)')
    
    args = parser.parse_args()
    
    if args.all:
        scenes = list(range(1, 11))
    elif args.scene:
        scenes = [args.scene]
    else:
        parser.print_help()
        print("\nError: Must specify --scene or --all")
        return
    
    for scene_num in scenes:
        try:
            computer = BBoxComputer(scene_num, args.dataset_dir)
            computer.compute_all()
            print()
        except Exception as e:
            print(f"Error processing scene {scene_num}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()