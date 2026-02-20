"""
Scene 10: High-Frequency Texture + Motion
Sphere with checkerboard texture rotating while translating.
Tests: Fine texture preservation, appearance vs geometry entanglement
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene10Trajectory(TrajectoryBase):
    """Textured sphere rotating and moving."""
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Circular path
        self.radius = 1.5
        self.height = 0.8
        self.rotations_orbit = 1.0  # Full circle
        self.rotations_spin = 3.0   # 3 full spins while orbiting
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        if object_id != 'textured_sphere':
            raise ValueError(f"Unknown object_id: {object_id}")
        
        time = np.clip(time, 0, self.duration)
        t = time / self.duration
        
        # Orbital motion (circular path)
        angle_orbit = 2 * np.pi * self.rotations_orbit * t
        pos = np.array([
            self.radius * np.cos(angle_orbit),
            self.radius * np.sin(angle_orbit),
            self.height
        ])
        
        # Spin rotation (around its own axis)
        angle_spin = 2 * np.pi * self.rotations_spin * t
        # Rotate around Y-axis
        qw = np.cos(angle_spin / 2)
        qx = 0
        qy = np.sin(angle_spin / 2)
        qz = 0
        quat = self.normalize_quaternion(np.array([qw, qx, qy, qz]))
        
        return {
            'position': pos,
            'quaternion': quat,
        }
    
    def get_object_ids(self) -> List[str]:
        return ['textured_sphere']
    
    def get_description(self) -> str:
        return "Checkerboard-textured sphere orbiting and spinning - tests texture preservation"