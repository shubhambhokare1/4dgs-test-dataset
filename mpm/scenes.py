"""
Scene definitions for the MPM ports of scene 3 (three-body collision) and
scene 7 (deformable sphere vs. rigid cube).

Geometry, colours and camera positions come from the original MuJoCo XML files
in ../scenarios so the new data lines up with the old dataset. The motion is no
longer scripted: objects get initial velocities and MPM produces everything
after that.

Units: metres, seconds, kilograms, pascals.
"""

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Uniform vertical FOV for every camera. The original rig mixed 45/50/55 deg,
# but 4DGaussians' D-NeRF loader reads a single camera_angle_x per dataset, so
# mixed FOVs silently give wrong intrinsics there. One FOV avoids that.
FOVY_DEG = 50.0


def _parse_mujoco_cameras(xml_path):
    """Return (target_xyz, [(name, pos_xyz), ...]) from a MuJoCo scene file."""
    root = ET.parse(xml_path).getroot()
    target = None
    for body in root.iter("body"):
        if body.get("name") == "target":
            target = np.array([float(v) for v in body.get("pos").split()])
    cams = []
    for cam in root.iter("camera"):
        pos = np.array([float(v) for v in cam.get("pos").split()])
        cams.append((cam.get("name"), pos))
    cams.sort(key=lambda c: int(c[0].replace("cam", "")))
    return target, cams


def _scene3():
    xml = REPO_ROOT / "scenarios" / "scene3_collision.xml"
    target, cams = _parse_mujoco_cameras(xml)

    height = 0.5
    speed = 2.0  # m/s; first contact at ~1.05 s
    base_E = 2.0e5
    # Slightly different stiffness per ball (0.5x, 1x, 2x of base_E).
    E_scale = [0.5, 1.0, 2.0]
    colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0)]
    # Aim points and staggered start radii follow the original scripted
    # trajectory (collision points / arrival times in scene3_trajectories.py),
    # so the impact is slightly off-centre and not perfectly simultaneous.
    aim = [np.array([0.15, 0.0, height]),
           np.array([-0.075, 0.13, height]),
           np.array([-0.075, -0.13, height])]
    start_r = [2.5, 2.56, 2.62]
    objects = []
    for i in range(3):
        ang = 2.0 * math.pi * i / 3.0
        start = np.array([start_r[i] * math.cos(ang), start_r[i] * math.sin(ang), height])
        d = aim[i] - start
        vel = speed * d / np.linalg.norm(d)
        objects.append(dict(
            name=f"sphere_{i + 1}", shape="sphere", radius=0.3,
            center=start.tolist(), velocity=vel.tolist(),
            E=base_E * E_scale[i], nu=0.3, density=1000.0,
            color=colors[i],
        ))

    return dict(
        name="scene3_collision",
        duration=3.0, fps=30,
        gravity=(0.0, 0.0, 0.0),  # objects float, as in the original layout
        # Physical simulation region (walls are added outside it).
        region_lo=(-4.0, -4.0, -0.5), region_hi=(4.0, 4.0, 1.5),
        dx=0.04,
        floor_z=-0.5, floor_friction=0.3,
        objects=objects,
        colliders=[],
        static_objects=[],
        target=target.tolist(),
        mujoco_cameras=[(n, p.tolist()) for n, p in cams],
        bounds=3.5,
    )


def _scene7():
    xml = REPO_ROOT / "scenarios" / "scene7_deformation.xml"
    target, cams = _parse_mujoco_cameras(xml)

    cube_center = (0.0, 0.0, 0.5)
    cube_half = (0.6, 0.6, 0.6)
    objects = [dict(
        name="sphere", shape="sphere", radius=0.4,
        center=[-3.0, 0.0, 0.5], velocity=[1.5, 0.0, 0.0],  # contact at ~1.33 s
        E=2.0e4, nu=0.3, density=1000.0,  # ~30% squash at impact
        color=(0.8, 0.3, 0.3),
    )]
    return dict(
        name="scene7_deformation",
        duration=3.0, fps=30,
        gravity=(0.0, 0.0, 0.0),
        region_lo=(-5.5, -1.5, -0.5), region_hi=(1.0, 1.5, 1.5),
        dx=0.03,
        floor_z=-0.5, floor_friction=0.3,
        objects=objects,
        # The cube is treated as perfectly rigid (infinite Young's modulus):
        # a grid-level collider, so only the ball deforms.
        colliders=[dict(type="box", center=cube_center, half=cube_half, friction=0.0)],
        static_objects=[dict(name="cube", shape="box", center=cube_center,
                             half=cube_half, color=(0.3, 0.3, 0.8))],
        target=target.tolist(),
        mujoco_cameras=[(n, p.tolist()) for n, p in cams],
        bounds=3.5,
    )


_BUILDERS = {"scene3_collision": _scene3, "scene7_deformation": _scene7}
ALIASES = {"3": "scene3_collision", "scene3": "scene3_collision",
           "7": "scene7_deformation", "scene7": "scene7_deformation"}


def get_scene(name):
    name = ALIASES.get(str(name), str(name))
    if name not in _BUILDERS:
        raise KeyError(f"unknown scene {name!r}; choose from {sorted(_BUILDERS)}")
    return deepcopy(_BUILDERS[name]())


def scene_names():
    return sorted(_BUILDERS)
