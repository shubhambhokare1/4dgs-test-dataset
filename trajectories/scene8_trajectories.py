"""
Scene 8: Thin Structure Tracking
Two thin rods - one translating, one rotating. Tests preservation of thin features.
Tests: Thin geometry preservation, anisotropic Gaussian scaling
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene8Trajectory(TrajectoryBase):
    """Two thin rods - one moving linearly, one rotating in place."""
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Moving rod trajectory (horizontal movement)
        self.moving_rod_start = np.array([-2.0, -0.5, 1.0])
        self.moving_rod_end = np.array([2.0, -0.5, 1.0])
        
        # Rotating rod position (stays at this position, just rotates)
        self.rotating_rod_pos = np.array([0, 0.5, 1.0])
        
        # Rotation: 3 full revolutions during animation
        self.num_rotations = 3.0
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        t = time / self.duration
        
        if object_id == 'rod_moving':
            # Moving rod - translates left to right, no rotation
            pos = self.moving_rod_start + (self.moving_rod_end - self.moving_rod_start) * t
            
            # Rotate 90 degrees around Y-axis to make it horizontal (lying flat)
            angle = np.pi / 2
            qw = np.cos(angle / 2)
            qx = 0
            qy = np.sin(angle / 2)
            qz = 0
            quat = self.normalize_quaternion(np.array([qw, qx, qy, qz]))
            
            return {
                'position': pos,
                'quaternion': quat,
            }
        
        elif object_id == 'rod_rotating':
            # Rotating rod - stays in place, rotates around its own axis (Z-axis)
            # First rotate to horizontal (Y-axis rotation), then spin around its length (X-axis rotation)
            
            # Base orientation: horizontal (90° around Y)
            base_angle_y = np.pi / 2
            
            # Spin angle around the rod's length axis
            spin_angle = 2 * np.pi * self.num_rotations * t
            
            # Combine rotations: first horizontal orientation, then spin
            # Quaternion for Y rotation (make horizontal)
            qy_w = np.cos(base_angle_y / 2)
            qy_x = 0
            qy_y = np.sin(base_angle_y / 2)
            qy_z = 0
            
            # Quaternion for X rotation (spin around length)
            qx_w = np.cos(spin_angle / 2)
            qx_x = np.sin(spin_angle / 2)
            qx_y = 0
            qx_z = 0
            
            # Combine quaternions (qy * qx)
            qw = qy_w * qx_w - qy_x * qx_x - qy_y * qx_y - qy_z * qx_z
            qx = qy_w * qx_x + qy_x * qx_w + qy_y * qx_z - qy_z * qx_y
            qy = qy_w * qx_y - qy_x * qx_z + qy_y * qx_w + qy_z * qx_x
            qz = qy_w * qx_z + qy_x * qx_y - qy_y * qx_x + qy_z * qx_w
            
            quat = self.normalize_quaternion(np.array([qw, qx, qy, qz]))
            
            return {
                'position': self.rotating_rod_pos,
                'quaternion': quat,
            }
        
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
    
    def get_object_ids(self) -> List[str]:
        return ['rod_moving', 'rod_rotating']
    
    def get_description(self) -> str:
        return "Two thin rods - one translating horizontally, one rotating (3 revolutions)"