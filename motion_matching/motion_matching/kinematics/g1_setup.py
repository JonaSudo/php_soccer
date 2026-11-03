"""Shared boilerplate for entry points that need a configured :class:`URDFKinematics`.

Kept for backward compatibility; new code should use
:meth:`motion_matching.kinematics.robot_spec.RobotSpec.make_kinematics` and
:attr:`RobotSpec.joint_axes` directly.
"""

from __future__ import annotations

import numpy as np

from motion_matching.kinematics.robot_spec import G1_SPEC, RobotSpec
from motion_matching.kinematics.urdf_kinematics import URDFKinematics


def init_robot_kinematics(spec: RobotSpec, *, verbose: bool = False) -> tuple[URDFKinematics, np.ndarray]:
    """Construct a :class:`URDFKinematics` for ``spec`` and return its joint rotation axes.

    Returns:
        ``(kin, joint_axes)`` where ``joint_axes`` is the ``(num_dofs, 3)``
        float32 array from :attr:`RobotSpec.joint_axes`. Meshes are not loaded.
    """
    kin = spec.make_kinematics(verbose=verbose)
    joint_axes = spec.joint_axes
    if verbose:
        print("\nComputing joint rotation axes...")
        for i, joint_name in enumerate(kin.joint_names):
            ax = joint_axes[i]
            print(f"{i:2d}: {joint_name:30s} -> axis: [{ax[0]:6.3f}, {ax[1]:6.3f}, {ax[2]:6.3f}]")
    return kin, joint_axes


def init_g1_kinematics(*, verbose: bool = False) -> tuple[URDFKinematics, np.ndarray]:
    """G1 shortcut for :func:`init_robot_kinematics`."""
    return init_robot_kinematics(G1_SPEC, verbose=verbose)
