"""
Scene 4: Occlusion and Dis-occlusion
Sphere makes a fast circular loop around a narrow wall.
Tests: Hallucination during occlusion, dis-occlusion handling
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene4Trajectory(TrajectoryBase):
    """Sphere orbits quickly in a circle around a narrow occluding wall."""
    
    def __init__(self, duration: float = 4.0, fps: int = 30):  # Faster: 4 seconds
        super().__init__(duration, fps)
        
        # Wall position (static, centered)
        self.wall_pos = np.array([0, 0, 1.0])
        
        # Circular orbit parameters
        self.orbit_radius = 1.8  # Slightly smaller orbit
        self.orbit_center = np.array([0, 0, 0.5])
        self.orbit_height = 0.5
        
        # Speed: complete one full circle in duration
        self.num_loops = 1
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        
        if object_id == 'sphere':
            # Calculate angle based on time (0 to 2π)
            angle = (time / self.duration) * 2 * np.pi * self.num_loops
            
            # Circular path in XY plane
            x = self.orbit_center[0] + self.orbit_radius * np.cos(angle)
            y = self.orbit_center[1] + self.orbit_radius * np.sin(angle)
            z = self.orbit_height
            
            pos = np.array([x, y, z])
            
            return {
                'position': pos,
                'quaternion': self.identity_quaternion(),
            }
        
        elif object_id == 'wall':
            # Static wall
            return {
                'position': self.wall_pos,
                'quaternion': self.identity_quaternion(),
            }
        
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
    
    def get_object_ids(self) -> List[str]:
        return ['sphere', 'wall']
    
    def get_object_bounds(self, time: float, object_id: str) -> dict:
        if object_id == 'sphere':
            return {'type': 'sphere', 'radius': 0.3}
        elif object_id == 'wall':
            # Thin wall: narrow in X, extended in Y and Z
            return {'type': 'obb', 'half_extents': np.array([0.05, 0.4, 1.0])}
        raise ValueError(f"Unknown object_id: {object_id}")

    def get_description(self) -> str:
        return "Sphere orbits quickly around narrow wall - tests occlusion"