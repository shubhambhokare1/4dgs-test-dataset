"""
Scene 6: Extreme Scale Change
Sphere moves: top-left corner → bottom center → top-right corner
Tests: Multi-scale representation, level-of-detail adaptation
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene6Trajectory(TrajectoryBase):
    """Sphere moving through corners with extreme depth/scale changes."""
    
    def __init__(self, duration: float = 6.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Three key positions
        self.corner_top_left = np.array([-2.0, 2.0, 3.0])   # Far away, top-left, high
        self.center_bottom = np.array([0, 0, 0.3])          # Close, center, low
        self.corner_top_right = np.array([2.0, 2.0, 3.0])  # Far away, top-right, high
        
        # Timing: split into two equal segments
        self.mid_time = 0.5
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        if object_id != 'sphere':
            raise ValueError(f"Unknown object_id: {object_id}")
        
        time = np.clip(time, 0, self.duration)
        t_norm = time / self.duration
        
        # First segment: top-left → bottom center
        if t_norm <= self.mid_time:
            t = t_norm / self.mid_time
            pos = self.corner_top_left + (self.center_bottom - self.corner_top_left) * t
        
        # Second segment: bottom center → top-right
        else:
            t = (t_norm - self.mid_time) / (1.0 - self.mid_time)
            pos = self.center_bottom + (self.corner_top_right - self.center_bottom) * t
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
        }
    
    def get_object_ids(self) -> List[str]:
        return ['sphere']
    
    def get_description(self) -> str:
        return "Sphere moves top-left → bottom center → top-right with extreme scale changes"