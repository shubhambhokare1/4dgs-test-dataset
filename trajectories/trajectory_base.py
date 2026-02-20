"""
Base class for trajectory definitions.
All scenario trajectories inherit from this.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

class TrajectoryBase(ABC):
    """
    Base class for defining object trajectories in synthetic scenes.
    
    Each trajectory defines:
    - Object positions over time
    - Object orientations (quaternions)
    - Metadata about the motion
    """
    
    def __init__(self, duration: float = 5.0, fps: int = 30):
        """
        Initialize trajectory.
        
        Args:
            duration: Total duration of the trajectory in seconds
            fps: Frames per second (for discrete time steps)
        """
        self.duration = duration
        self.fps = fps
        self.dt = 1.0 / fps
        self.num_frames = int(duration * fps)
    
    @abstractmethod
    def get_object_state(self, time: float, object_id: str) -> Dict:
        """
        Get object state at a specific time.
        
        Args:
            time: Time in seconds (0 to duration)
            object_id: Identifier for the object
            
        Returns:
            Dictionary with:
                - 'position': np.array([x, y, z])
                - 'quaternion': np.array([qw, qx, qy, qz])
                - 'velocity': np.array([vx, vy, vz]) (optional)
        """
        pass
    
    @abstractmethod
    def get_object_ids(self) -> List[str]:
        """Return list of object IDs in this trajectory."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of what this trajectory tests."""
        pass
    
    def get_metadata(self) -> Dict:
        """Return metadata about the trajectory."""
        return {
            'duration': self.duration,
            'fps': self.fps,
            'num_frames': self.num_frames,
            'object_ids': self.get_object_ids(),
            'description': self.get_description()
        }
    
    @staticmethod
    def identity_quaternion() -> np.ndarray:
        """Return identity quaternion (no rotation)."""
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        """Normalize a quaternion."""
        return q / np.linalg.norm(q)