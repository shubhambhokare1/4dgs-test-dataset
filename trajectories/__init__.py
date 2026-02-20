"""
Trajectory definitions for all scenarios.
"""

from .trajectory_base import TrajectoryBase
from .scene1_trajectories import Scene1Trajectory
from .scene2_trajectories import Scene2Trajectory
from .scene3_trajectories import Scene3Trajectory
from .scene4_trajectories import Scene4Trajectory
from .scene5_trajectories import Scene5Trajectory
from .scene6_trajectories import Scene6Trajectory
from .scene7_trajectories import Scene7Trajectory
from .scene8_trajectories import Scene8Trajectory
from .scene9_trajectories import Scene9Trajectory
from .scene10_trajectories import Scene10Trajectory

# Map scene numbers to trajectory classes
TRAJECTORY_MAP = {
    1: Scene1Trajectory,
    2: Scene2Trajectory,
    3: Scene3Trajectory,
    4: Scene4Trajectory,
    5: Scene5Trajectory,
    6: Scene6Trajectory,
    7: Scene7Trajectory,
    8: Scene8Trajectory,
    9: Scene9Trajectory,
    10: Scene10Trajectory,
}

def get_trajectory(scene_num: int, **kwargs) -> TrajectoryBase:
    """
    Factory function to get trajectory for a scene.
    
    Args:
        scene_num: Scene number (1-10)
        **kwargs: Additional arguments passed to trajectory constructor
        
    Returns:
        TrajectoryBase instance
    """
    if scene_num not in TRAJECTORY_MAP:
        raise ValueError(f"Unknown scene number: {scene_num}. Valid: 1-10")
    
    return TRAJECTORY_MAP[scene_num](**kwargs)

__all__ = [
    'TrajectoryBase',
    'get_trajectory',
    'TRAJECTORY_MAP',
]