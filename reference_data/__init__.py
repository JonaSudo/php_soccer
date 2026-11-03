"""Phase 1 of the LUPUS walk-kick plan: reference motions for motion matching.

Pipeline (see reference_data/README.md):

    source clip (LAFAN1 BVH, AMASS SMPL-X, OmniContact BVH + ball, CMU via AMASS)
      -> human_clip.npz        neutral format: world joint positions, z-up, metres
      -> retarget_lupus.py     holosoma interaction-mesh retargeter, LUPUS + ball
      -> postprocess.py        PHP qpos order, time scaling, mirroring,
                               feasibility check, kick annotation, export
      -> verify.py             acceptance: no ground penetration, kick foot hits ball

Everything here runs on the CPU. The retargeting step needs the
holosoma_retargeting environment (numpy 2.3), the rest needs the
motion_matching environment.
"""
