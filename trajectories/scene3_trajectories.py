"""
Scene 3: Three-Body Collision
Three spheres colliding with immediate bounce on contact.
Tests: Multi-object interaction, chaotic motion handling
"""

import numpy as np
from typing import Dict, List
from .trajectory_base import TrajectoryBase

class Scene3Trajectory(TrajectoryBase):
    """Three spheres with immediate collision - no waiting."""
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        super().__init__(duration, fps)
        
        # Starting positions - farther out
        radius = 2.5
        height = 0.5
        
        # 120° apart
        self.positions = {
            'sphere_1': {
                'start': np.array([radius, 0, height]),
                'collision_point': np.array([0.15, 0, height]),
                'arrival_time': 0.35,  # When this sphere reaches collision point
            },
            'sphere_2': {
                'start': np.array([radius * np.cos(2*np.pi/3), radius * np.sin(2*np.pi/3), height]),
                'collision_point': np.array([-0.075, 0.13, height]),
                'arrival_time': 0.38,  # Arrives 0.03s later
            },
            'sphere_3': {
                'start': np.array([radius * np.cos(4*np.pi/3), radius * np.sin(4*np.pi/3), height]),
                'collision_point': np.array([-0.075, -0.13, height]),
                'arrival_time': 0.41,  # Arrives 0.06s later than first
            },
        }
        
        # Pre-compute bounce trajectories
        for sphere_id, data in self.positions.items():
            incoming_dir = data['collision_point'] - data['start']
            incoming_dir = incoming_dir / np.linalg.norm(incoming_dir)
            
            # Bounce back with deflection
            bounce_dir = -incoming_dir
            perp = np.array([-incoming_dir[1], incoming_dir[0], 0])
            
            if sphere_id == 'sphere_1':
                deflection = 0.4
            elif sphere_id == 'sphere_2':
                deflection = -0.6
            else:
                deflection = 0.2
            
            bounce_dir = bounce_dir + perp * deflection
            bounce_dir = bounce_dir / np.linalg.norm(bounce_dir)
            
            data['bounce_dir'] = bounce_dir
        
        self.bounce_distance = 2.5
    
    def get_object_state(self, time: float, object_id: str) -> Dict:
        time = np.clip(time, 0, self.duration)
        
        data = self.positions[object_id]
        start_pos = data['start']
        collision_point = data['collision_point']
        arrival_time = data['arrival_time']
        bounce_dir = data['bounce_dir']
        
        # Phase 1: Approaching collision point
        if time < arrival_time:
            t = time / arrival_time
            pos = start_pos + (collision_point - start_pos) * t
        
        # Phase 2: Bouncing away (starts IMMEDIATELY after arrival)
        else:
            t = (time - arrival_time) / (self.duration - arrival_time)
            t_eased = 1 - (1 - t) * (1 - t)  # Ease out
            pos = collision_point + bounce_dir * self.bounce_distance * t_eased
        
        return {
            'position': pos,
            'quaternion': self.identity_quaternion(),
        }
    
    def get_object_ids(self) -> List[str]:
        return ['sphere_1', 'sphere_2', 'sphere_3']
    
    def get_description(self) -> str:
        return "Three spheres with staggered arrival and immediate bounce"