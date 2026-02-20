"""
Scene 9: Topology Change - Split and Merge
Blob splits into two droplets, then merges back.
Tests: Topology change handling (major 4DGS limitation)
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene9Trajectory(TrajectoryBase):
    """Single sphere splits into two, then merges back."""
    
    def __init__(self, duration: float = 6.0, fps: int = 30):
        super().__init__(duration, fps)
        
        self.center = np.array([0, 0, 0.5])
        self.separation = 1.0  # Max separation distance
        
        # Timeline
        self.split_start = 0.2 * duration
        self.split_end = 0.4 * duration
        self.merge_start = 0.6 * duration
        self.merge_end = 0.8 * duration
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        
        # Calculate separation amount
        if time < self.split_start:
            # Single sphere
            sep = 0.0
        elif time < self.split_end:
            # Splitting
            t = (time - self.split_start) / (self.split_end - self.split_start)
            sep = self.separation * t
        elif time < self.merge_start:
            # Fully separated
            sep = self.separation
        elif time < self.merge_end:
            # Merging
            t = (time - self.merge_start) / (self.merge_end - self.merge_start)
            sep = self.separation * (1.0 - t)
        else:
            # Merged
            sep = 0.0
        
        if object_id == 'droplet_1':
            offset = np.array([-sep/2, 0, 0])
        elif object_id == 'droplet_2':
            offset = np.array([sep/2, 0, 0])
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
        
        pos = self.center + offset
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
        }
    
    def get_object_ids(self) -> List[str]:
        return ['droplet_1', 'droplet_2']
    
    def get_description(self) -> str:
        return "Blob splits into two droplets, then merges back - tests topology changes"