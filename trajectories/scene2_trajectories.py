"""
Scene 2: Close Proximity - Identical Appearance
Two identical white spheres passing close to each other.
Tests: Identity tracking without appearance cues
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene2Trajectory(TrajectoryBase):
    """Two identical spheres with crossing paths."""
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Crossing paths
        self.sphere1_start = np.array([-1.5, -1.0, 0.4])
        self.sphere1_end = np.array([1.5, 1.0, 0.4])
        
        self.sphere2_start = np.array([-1.5, 1.0, 0.7])
        self.sphere2_end = np.array([1.5, -1.0, 0.7])
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        t = time / self.duration
        
        if object_id == 'sphere_1':
            pos = self.sphere1_start + (self.sphere1_end - self.sphere1_start) * t
        elif object_id == 'sphere_2':
            pos = self.sphere2_start + (self.sphere2_end - self.sphere2_start) * t
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
        }
    
    def get_object_ids(self) -> List[str]:
        return ['sphere_1', 'sphere_2']
    
    def get_description(self) -> str:
        return "Two identical white spheres with crossing paths - tests identity tracking"