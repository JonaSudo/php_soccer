"""Packaged G1 assets and the user-configurable motion database cache."""

import os
from pathlib import Path

# Wheels installed by pip expose these resources as ordinary files. Both model
# loaders need the complete relative mesh hierarchy alongside the model file.
# ``files()`` returns a MultiplexedPath (not a plain directory) when the
# repository root is on sys.path, because the outer ``motion_matching/`` folder
# then counts as a namespace portion. This file's location is unambiguous.
PKG_ROOT: Path = Path(__file__).resolve().parents[1]
ASSETS_DIR: Path = PKG_ROOT / "assets"
ROBOTS_DIR: Path = ASSETS_DIR / "robots"
ROBOT_MOTIONS_DIR: Path = ASSETS_DIR / "robot_motions"

# Downloaded databases and generated feature caches must remain writable even
# when the installed package is read-only. Importing does not create directories.
DATABASE_DIR: Path = Path(
    os.environ.get("PHP_MOTION_DATABASE_DIR", "~/.cache/php-parkour/databases")
).expanduser().resolve()

G1_URDF: str = str(ROBOTS_DIR / "unitree_description" / "urdf" / "g1" / "main_mesh_collision_halfspherehand.urdf")
G1_MJCF: str = str(ROBOTS_DIR / "unitree_description" / "mjcf" / "g1.xml")

# LUPUS lives in the repository (robots/lupus, built by build_urdf.py), not in
# the package. LUPUS_DIR overrides the location, e.g. for an installed wheel.
LUPUS_DIR: Path = Path(os.environ.get("LUPUS_DIR", PKG_ROOT.parents[1] / "robots" / "lupus")).expanduser().resolve()
LUPUS_URDF: str = str(LUPUS_DIR / "lupus.urdf")
LUPUS_MJCF: str = str(LUPUS_DIR / "lupus.xml")


def db_path(relative: str) -> str:
    """Resolve a path beneath ``PHP_MOTION_DATABASE_DIR`` (the database cache)."""
    return str(DATABASE_DIR / relative)
