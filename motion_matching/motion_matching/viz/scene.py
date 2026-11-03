"""Shared Viser scene setup for the interactive viewers.

``run.py`` (interactive motion matching) and ``play_database.py`` (database
playback) built the same world frame + grid + robot-root frame + robot ``ViserUrdf``
by hand. These helpers hold that boilerplate in one place. They only affect the
interactive Viser scene; they produce no saved-file output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import viser
from viser.extras import ViserUrdf

from motion_matching.kinematics.robot_spec import RobotSpec, get_robot_spec


def add_world_scene(server: viser.ViserServer) -> Any:
    """Add the world origin frame, an XY grid, and the robot-root frame.

    Returns the ``/robot_root`` frame handle (callers attach the robot to it and
    drive its pose).
    """
    server.scene.add_frame(
        "/world",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    server.scene.add_grid(
        "/xy_grid", width=100.0, height=100.0, position=np.array([0.0, 0.0, 0.0], dtype=np.float32), plane="xy"
    )
    return server.scene.add_frame("/robot_root", show_axes=True)


def load_robot_viser_urdf(server: viser.ViserServer, robot: str | RobotSpec | None = None) -> ViserUrdf:
    """Attach the robot's URDF (default: G1) to ``/robot_root`` for visualization."""
    spec = get_robot_spec(robot)
    return ViserUrdf(
        server,
        urdf_or_path=Path(spec.urdf_path),
        root_node_name="/robot_root",
        load_meshes=True,
        load_collision_meshes=False,
    )


def add_world_scene_with_robot(
    server: viser.ViserServer, robot: str | RobotSpec | None = None
) -> Tuple[Any, ViserUrdf]:
    """Convenience: :func:`add_world_scene` + :func:`load_robot_viser_urdf`.

    Returns ``(root_frame, viser_urdf)``.
    """
    root_frame = add_world_scene(server)
    viser_urdf = load_robot_viser_urdf(server, robot)
    return root_frame, viser_urdf


def load_g1_viser_urdf(server: viser.ViserServer) -> ViserUrdf:
    """G1 shortcut for :func:`load_robot_viser_urdf`."""
    return load_robot_viser_urdf(server, "g1")


def add_world_scene_with_g1(server: viser.ViserServer) -> Tuple[Any, ViserUrdf]:
    """G1 shortcut for :func:`add_world_scene_with_robot`."""
    return add_world_scene_with_robot(server, "g1")
