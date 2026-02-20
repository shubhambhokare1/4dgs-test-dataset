"""
Scene 1: Close Proximity - Different Colors
Two spheres (red and blue) passing very close to each other.
Tests: Gaussian boundary preservation, identity maintenance
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene1Trajectory(TrajectoryBase):
    """Two spheres passing close to each other with different colors."""
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Define start and end positions
        self.red_start = np.array([-1.5, -0.5, 0.4])
        self.red_end = np.array([1.5, 0.5, 0.4])
        
        self.blue_start = np.array([-1.5, 0.5, 0.7])
        self.blue_end = np.array([1.5, -0.5, 0.7])
        
        # They cross at center (0, 0, 0.5) at t=2.5s
        # Minimum distance: ~0.141m at crossing point
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        # Clamp time
        time = np.clip(time, 0, self.duration)
        t_normalized = time / self.duration
        
        if object_id == 'sphere_red':
            pos = self.red_start + (self.red_end - self.red_start) * t_normalized
        elif object_id == 'sphere_blue':
            pos = self.blue_start + (self.blue_end - self.blue_start) * t_normalized
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
            'velocity': (self.red_end - self.red_start) / self.duration
        }
    
    def get_object_ids(self) -> List[str]:
        return ['sphere_red', 'sphere_blue']
    
    def get_description(self) -> str:
        return "Two different-colored spheres passing within 15cm of each other"
    
    def get_minimum_distance(self) -> float:
        """Calculate minimum distance between objects."""
        # At t=2.5s (center crossing)
        red_pos = self.red_start + (self.red_end - self.red_start) * 0.5
        blue_pos = self.blue_start + (self.blue_end - self.blue_start) * 0.5
        return np.linalg.norm(red_pos - blue_pos)