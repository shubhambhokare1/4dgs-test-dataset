#!/usr/bin/env python3
"""
Render a 360° orbit flyaround of a scene.

A virtual camera circles the scene centre while the scene animation plays,
producing a PNG frame sequence (and optionally an MP4 via ffmpeg).

Usage:
    python scripts/render_orbit.py --scene 1
    python scripts/render_orbit.py --scene 3 --radius 5.0 --elevation -30
    python scripts/render_orbit.py --scene 5 --num_frames 360 --video
    python scripts/render_orbit.py --scene 2 --freeze_time 1.5   # frozen pose
"""

import os
import sys

if "MUJOCO_GL" not in os.environ:
    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "glfw"
    elif os.path.exists("/dev/dri"):
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import subprocess
from pathlib import Path

import numpy as np
import mujoco
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectories import get_trajectory

# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------

SCENE_INFO = {
    1:  {"name": "scene1_close_proximity",  "xml": "scene1_close_proximity.xml"},
    2:  {"name": "scene2_identical_objects", "xml": "scene2_identical_objects.xml"},
    3:  {"name": "scene3_collision",         "xml": "scene3_collision.xml"},
    4:  {"name": "scene4_occlusion",         "xml": "scene4_occlusion.xml"},
    5:  {"name": "scene5_rapid_motion",      "xml": "scene5_rapid_motion.xml"},
    6:  {"name": "scene6_scale_change",      "xml": "scene6_scale_change.xml"},
    7:  {"name": "scene7_deformation",       "xml": "scene7_deformation.xml"},
    8:  {"name": "scene8_thin_structure",    "xml": "scene8_thin_structure.xml"},
    9:  {"name": "scene9_topology",          "xml": "scene9_topology.xml"},
    10: {"name": "scene10_texture",          "xml": "scene10_texture.xml"},
}

# Default orbit radius per scene — matched to updated camera distances (scaled for 800x800)
DEFAULT_RADIUS = {
    1: 5.0, 2: 5.0, 3: 7.0, 4: 5.5, 5: 6.5,
    6: 6.5, 7: 6.0, 8: 5.5, 9: 3.0, 10: 5.0,
}


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_orbit(
    scene_num:   int,
    output_dir:  Path,
    radius:      float  = None,
    elevation:   float  = -20.0,
    num_frames:  int    = 180,
    resolution:  tuple  = (800, 800),
    fps:         int    = 30,
    freeze_time: float  = None,
    make_video:  bool   = False,
) -> Path:
    """
    Render ``num_frames`` frames of a 360° orbit around scene ``scene_num``.

    Camera azimuth sweeps 0 → 360° over all frames.  By default the scene
    animation plays exactly once in sync with the orbit; pass ``freeze_time``
    to lock the scene at a fixed simulation time instead.

    Returns the output directory containing the PNG sequence.
    """
    if scene_num not in SCENE_INFO:
        raise ValueError(f"Unknown scene: {scene_num}. Valid: {list(SCENE_INFO)}")

    info     = SCENE_INFO[scene_num]
    xml_path = Path(__file__).parent.parent / "scenarios" / info["xml"]
    if not xml_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    if radius is None:
        radius = DEFAULT_RADIUS.get(scene_num, 4.0)

    print(f"Scene {scene_num}: {info['name']}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    # ---- trajectory --------------------------------------------------------
    trajectory = get_trajectory(scene_num, fps=fps)
    object_mappings: dict = {}
    for obj_id in trajectory.get_object_ids():
        try:
            body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_id)
            jnt_id   = model.body_jntadr[body_id]
            qpos_idx = model.jnt_qposadr[jnt_id]
            object_mappings[obj_id] = qpos_idx
        except (KeyError, IndexError):
            pass

    # ---- look-at target ----------------------------------------------------
    # Use the "target" body defined in the XML (all scenes have one at ~0,0,0.5).
    # Fall back to origin if not present.
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    # ---- output directory --------------------------------------------------
    out_dir = output_dir / f"orbit_scene{scene_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height = resolution
    renderer = mujoco.Renderer(model, height=height, width=width)

    # ---- orbit camera ------------------------------------------------------
    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = radius
    cam.elevation = elevation

    print(f"  Radius {radius} m  |  Elevation {elevation}°  |  "
          f"{num_frames} frames  |  {width}×{height}")
    print(f"  Output: {out_dir}")
    if freeze_time is not None:
        print(f"  Scene frozen at t={freeze_time:.3f}s")
    else:
        print(f"  Scene animated: 1 full loop over {num_frames} orbit frames")

    # ---- render loop -------------------------------------------------------
    for frame_idx in tqdm(range(num_frames), desc="Rendering"):
        # Simulation time
        if freeze_time is not None:
            sim_time = freeze_time
        else:
            sim_time = (frame_idx / num_frames) * trajectory.duration

        # Update scene state
        for obj_id, qpos_idx in object_mappings.items():
            state = trajectory.get_object_state(sim_time, obj_id)
            data.qpos[qpos_idx:qpos_idx + 3] = state["position"]
            data.qpos[qpos_idx + 3:qpos_idx + 7] = state["quaternion"]
        mujoco.mj_forward(model, data)

        # Orbit camera azimuth (0 → 360°)
        cam.azimuth = (frame_idx / num_frames) * 360.0

        # Look-at: track target body if available
        if target_id >= 0:
            cam.lookat[:] = data.xpos[target_id]
        else:
            cam.lookat[:] = [0.0, 0.0, 0.5]

        # Render
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()

        Image.fromarray(rgb).save(out_dir / f"{frame_idx:05d}.png")

    renderer.close()
    print(f"  {num_frames} frames saved to {out_dir}/")

    # ---- optional video ----------------------------------------------------
    if make_video:
        _compile_video(out_dir, output_dir / f"orbit_scene{scene_num}.mp4", fps)

    return out_dir


def _compile_video(frames_dir: Path, video_path: Path, fps: int) -> None:
    """Compile PNG sequence to MP4 via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Video saved: {video_path}")
    except FileNotFoundError:
        print("  [warn] ffmpeg not found — install it to compile MP4.")
    except subprocess.CalledProcessError as e:
        print(f"  [warn] ffmpeg failed: {e.stderr.decode()[:200]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 360° orbit flyaround of a scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--scene", type=int, default=None,
                        help="Scene number (1–10)")
    parser.add_argument("--all", action="store_true",
                        help="Render all 10 scenes")
    parser.add_argument("-o", "--output", type=str, default="renders",
                        help="Root output directory (default: renders)")
    parser.add_argument("--radius", type=float, default=None,
                        help="Orbit radius in metres (default: auto per scene)")
    parser.add_argument("--elevation", type=float, default=-20.0,
                        help="Camera elevation in degrees, negative = looking "
                             "down from above (default: -20)")
    parser.add_argument("--num_frames", type=int, default=180,
                        help="Total orbit frames (default: 180 → 2 s at 30 fps)")
    parser.add_argument("--resolution", type=str, default="800x800",
                        help="WIDTHxHEIGHT (default: 800x800)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate (default: 30)")
    parser.add_argument("--freeze_time", type=float, default=None,
                        help="Lock scene at this simulation time in seconds "
                             "(default: scene animates in sync with orbit)")
    parser.add_argument("--video", action="store_true",
                        help="Compile frames to MP4 with ffmpeg after rendering")

    args = parser.parse_args()

    if not args.all and args.scene is None:
        parser.print_help()
        print("\nError: must specify --scene or --all")
        sys.exit(1)

    try:
        w, h = map(int, args.resolution.split("x"))
    except ValueError:
        print(f"Error: invalid resolution '{args.resolution}' — use WIDTHxHEIGHT")
        sys.exit(1)

    scenes = list(SCENE_INFO.keys()) if args.all else [args.scene]

    for scene_num in scenes:
        try:
            render_orbit(
                scene_num=scene_num,
                output_dir=Path(args.output),
                radius=args.radius,
                elevation=args.elevation,
                num_frames=args.num_frames,
                resolution=(w, h),
                fps=args.fps,
                freeze_time=args.freeze_time,
                make_video=args.video,
            )
        except Exception as e:
            print(f"Error (scene {scene_num}): {e}")
            import traceback
            traceback.print_exc()
            if not args.all:
                sys.exit(1)


if __name__ == "__main__":
    main()
