"""
Interactive viewer for all scenarios.

Usage:
    python src/viewer.py --scene 1
    python src/viewer.py --scene 5 --duration 10
    python src/viewer.py --list
"""

import os
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

import argparse
import sys
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectories import get_trajectory, TRAJECTORY_MAP


SCENE_INFO = {
    1: {
        'name': 'Close Proximity - Different Colors',
        'xml': 'scene1_close_proximity.xml',
        'tests': 'Gaussian boundary preservation, identity maintenance'
    },
    2: {
        'name': 'Close Proximity - Identical Objects',
        'xml': 'scene2_identical_objects.xml',
        'tests': 'Identity tracking without appearance cues'
    },
    3: {
        'name': 'Three-Body Collision',
        'xml': 'scene3_collision.xml',
        'tests': 'Multi-object interaction, chaotic motion'
    },
    4: {
        'name': 'Occlusion and Dis-occlusion',
        'xml': 'scene4_occlusion.xml',
        'tests': 'Occlusion handling, dis-occlusion quality'
    },
    5: {
        'name': 'Rapid Direction Changes',
        'xml': 'scene5_rapid_motion.xml',
        'tests': 'Motion smoothness, temporal interpolation'
    },
    6: {
        'name': 'Extreme Scale Change',
        'xml': 'scene6_scale_change.xml',
        'tests': 'Multi-scale representation, LOD adaptation'
    },
    7: {
        'name': 'Deformable vs Rigid',
        'xml': 'scene7_deformation.xml',
        'tests': 'Non-rigid deformation capture'
    },
    8: {
        'name': 'Thin Structure Tracking',
        'xml': 'scene8_thin_structure.xml',
        'tests': 'Thin geometry preservation'
    },
    9: {
        'name': 'Topology Change',
        'xml': 'scene9_topology.xml',
        'tests': 'Split/merge topology changes'
    },
    10: {
        'name': 'High-Frequency Texture',
        'xml': 'scene10_texture.xml',
        'tests': 'Texture detail preservation'
    },
}


def list_scenes():
    """Print all available scenes."""
    print("\n" + "="*70)
    print("Available Scenes for 4DGS Testing")
    print("="*70)
    for scene_num, info in SCENE_INFO.items():
        print(f"\nScene {scene_num}: {info['name']}")
        print(f"  Tests: {info['tests']}")
    print("\n" + "="*70)


def view_scene(scene_num: int, duration: float = None, fps: int = 30):
    """
    Launch interactive viewer for a scene.
    
    Args:
        scene_num: Scene number (1-10)
        duration: Override default duration
        fps: Frames per second for rendering
    """
    
    if scene_num not in SCENE_INFO:
        print(f"Error: Unknown scene {scene_num}")
        print("Use --list to see available scenes")
        return
    
    info = SCENE_INFO[scene_num]
    
    # Load MuJoCo model
    xml_path = Path(__file__).parent.parent / "scenarios" / info['xml']
    if not xml_path.exists():
        print(f"Error: Scene XML not found: {xml_path}")
        return
    
    print(f"\nLoading Scene {scene_num}: {info['name']}")
    print(f"  Tests: {info['tests']}")
    print(f"  XML: {xml_path}")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Load trajectory
    traj_kwargs = {'fps': fps}
    if duration is not None:
        traj_kwargs['duration'] = duration
    
    trajectory = get_trajectory(scene_num, **traj_kwargs)
    
    print(f"  Duration: {trajectory.duration}s")
    print(f"  Objects: {', '.join(trajectory.get_object_ids())}")
    
    # Get body → joint → qpos mappings for all objects
    object_mappings = {}
    for obj_id in trajectory.get_object_ids():
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
            jnt_id = model.body_jntadr[body_id]
            qpos_idx = model.jnt_qposadr[jnt_id]
            object_mappings[obj_id] = qpos_idx
        except KeyError:
            print(f"Warning: Object '{obj_id}' not found in model")
    
    print("\nControls:")
    print("  Mouse drag    : Rotate view")
    print("  Scroll        : Zoom")
    print("  Double-click  : Cycle cameras")
    print("  ESC           : Exit")
    print("\nStarting viewer...\n")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            elapsed = time.time() - start_time
            current_time = elapsed % trajectory.duration
            
            # Update all objects
            for obj_id, qpos_idx in object_mappings.items():
                state = trajectory.get_object_state(current_time, obj_id)
                
                # Set position (3 values)
                data.qpos[qpos_idx:qpos_idx+3] = state['position']
                
                # Set orientation (4 values: quaternion)
                data.qpos[qpos_idx+3:qpos_idx+7] = state['quaternion']
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Status message
            print(f"t={current_time:5.2f}s / {trajectory.duration:.1f}s", end='\r')
            
            time.sleep(1.0 / fps)


def main():
    parser = argparse.ArgumentParser(
        description='Interactive viewer for 4DGS synthetic scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/viewer.py --list              # List all scenes
  python src/viewer.py --scene 1           # View scene 1
  python src/viewer.py --scene 5 -d 10     # View scene 5 for 10 seconds
  python src/viewer.py -s 3 --fps 60       # View scene 3 at 60 FPS
        """
    )
    
    parser.add_argument('-s', '--scene', type=int, 
                       help='Scene number to view (1-10)')
    parser.add_argument('-l', '--list', action='store_true',
                       help='List all available scenes')
    parser.add_argument('-d', '--duration', type=float,
                       help='Override scene duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    if args.list:
        list_scenes()
        return
    
    if args.scene is None:
        parser.print_help()
        print("\nError: Must specify --scene or --list")
        return
    
    try:
        view_scene(args.scene, args.duration, args.fps)
    except KeyboardInterrupt:
        print("\n\nViewer closed by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()