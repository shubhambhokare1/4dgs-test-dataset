"""
Scene 7: Deformable vs Rigid Collision
Soft sphere bounces off rigid cube with visible deformation.
Tests: Non-rigid deformation capture, collision dynamics
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene7Trajectory(TrajectoryBase):
    """Sphere bounces off rigid cube with prominent deformation at contact."""
    
    def __init__(self, duration: float = 6.0, fps: int = 30):  # Increased from 5.0 to 6.0
        super().__init__(duration, fps)
        
        # Cube stays centered
        self.cube_pos = np.array([0, 0, 0.5])
        self.cube_half_size = 0.6  # Cube half-size
        self.sphere_radius = 0.4
        
        # Sphere trajectory - approach from left
        self.sphere_start = np.array([-3.0, 0, 0.5])
        
        # Initial contact point: when sphere SURFACE first touches cube
        cube_left_face = self.cube_pos[0] - self.cube_half_size
        self.initial_contact_x = cube_left_face - self.sphere_radius
        self.contact_point = np.array([self.initial_contact_x, 0, 0.5])
        
        # After bounce, sphere travels BACK TO THE LEFT
        self.sphere_end = np.array([-3.0, 0, 0.5])
        
        # Timing - much longer deformation phase
        self.approach_end = 0.30   # 30% approaching (1.8s)
        self.deform_end = 0.65     # 35% deformation phase (2.1s) - MUCH LONGER
        self.bounce_end = 1.0      # 35% bouncing away (2.1s)
        
        # Deformation amount - even more compression
        self.max_deformation = 0.30  # 30cm compression (scaled with doubled object size)
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        t_norm = time / self.duration
        
        if object_id == 'cube':
            # Static cube - no movement
            return {
                'position': self.cube_pos,
                'quaternion': self.identity_quaternion(),
            }
        
        elif object_id == 'sphere':
            # Phase 1: Approaching cube from left
            if t_norm < self.approach_end:
                t = t_norm / self.approach_end
                t_eased = t * t  # Ease in (accelerating)
                pos = self.sphere_start + (self.contact_point - self.sphere_start) * t_eased
            
            # Phase 2: SLOW deformation at contact
            elif t_norm < self.deform_end:
                t = (t_norm - self.approach_end) / (self.deform_end - self.approach_end)
                
                # Slower compression curve - ease in and out for more visible deformation
                # Use a smoother sine curve
                t_smooth = 0.5 - 0.5 * np.cos(np.pi * t)  # Smooth S-curve
                compression = self.max_deformation * np.sin(np.pi * t)
                
                # Sphere moves TOWARD cube (positive X direction from contact point)
                pos_x = self.initial_contact_x + compression
                pos = np.array([pos_x, 0, 0.5])
            
            # Phase 3: Bouncing away BACK TO THE LEFT
            else:
                t = (t_norm - self.deform_end) / (1.0 - self.deform_end)
                t_eased = 1 - (1 - t) * (1 - t)  # Ease out
                
                # Bounce back from contact point to starting position
                pos = self.contact_point + (self.sphere_end - self.contact_point) * t_eased
            
            return {
                'position': pos,
                'quaternion': self.identity_quaternion(),
            }
        
        else:
            raise ValueError(f"Unknown object_id: {object_id}")
    
    def get_object_ids(self) -> List[str]:
        return ['sphere', 'cube']
    
    def get_description(self) -> str:
        return "Sphere bounces off cube with slow, prominent deformation (30cm compression)"