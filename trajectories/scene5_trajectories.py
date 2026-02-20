"""
Scene 5: Rapid Direction Changes (Zigzag)
Sphere follows zigzag path with 90° turns every 0.4 seconds.
Tests: Motion smoothness assumptions, temporal interpolation
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene5Trajectory(TrajectoryBase):
    """Zigzag motion with sudden direction changes."""
    
    def __init__(self, duration: float = 6.0, fps: int = 30):
        super().__init__(duration, fps)
        
        self.speed = 1.5  # m/s
        self.segment_duration = 0.4  # Change direction every 0.4s
        
        # Direction cycle: +X, +Y, -X, -Y
        self.directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        if object_id != 'sphere':
            raise ValueError(f"Unknown object_id: {object_id}")
        
        time = np.clip(time, 0, self.duration)
        
        # Accumulate position through segments
        pos = np.array([0.0, 0.0, 0.5])
        remaining_time = time
        segment_idx = 0
        
        while remaining_time > 0:
            direction = self.directions[segment_idx % 4]
            time_in_segment = min(remaining_time, self.segment_duration)
            pos += direction * self.speed * time_in_segment
            remaining_time -= self.segment_duration
            segment_idx += 1
        
        # Keep in bounds
        pos[0] = np.clip(pos[0], -2, 2)
        pos[1] = np.clip(pos[1], -2, 2)
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
        }
    
    def get_object_ids(self) -> List[str]:
        return ['sphere']
    
    def get_description(self) -> str:
        return "Zigzag motion with 90° direction changes every 0.4s"