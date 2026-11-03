"""Package-wide numeric constants for the motion-matching pipeline.

Robot-specific values (bone indices, joint axes, DoF count, URDF) live in
:class:`motion_matching.kinematics.robot_spec.RobotSpec`; the G1 defaults are
:data:`motion_matching.kinematics.robot_spec.G1_SPEC`.
"""

# Number of randomized terrain variants generated per scenario.
NUM_TERRAIN_VARIANTS = 10
