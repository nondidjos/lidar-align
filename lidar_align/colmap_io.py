"""Thin wrapper over pycolmap for loading/saving a reconstruction and pulling out
the 3D point coordinates as a single array (for nearest-plane association)."""
from __future__ import annotations
import os
import numpy as np
import pycolmap


def load(path) -> "pycolmap.Reconstruction":
    p = str(path)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"COLMAP model directory not found: {p!r}")
    has_model = any(os.path.exists(os.path.join(p, f"points3D.{ext}"))
                    for ext in ("bin", "txt"))
    if not has_model:
        raise FileNotFoundError(
            f"no COLMAP model in {p!r} (expected cameras/images/points3D .bin or .txt). "
            f"Point --sparse-in at a 'sparse/0'-style folder.")
    return pycolmap.Reconstruction(p)


def save(rec: "pycolmap.Reconstruction", path) -> None:
    os.makedirs(path, exist_ok=True)
    rec.write(str(path))


def world_points(rec: "pycolmap.Reconstruction"):
    """Return (ids, xyz[N,3]) for all 3D points, order aligned to `ids`."""
    ids = list(rec.points3D.keys())
    xyz = np.array([rec.points3D[i].xyz for i in ids], dtype=np.float64)
    return ids, xyz


def set_world_points(rec: "pycolmap.Reconstruction", ids, xyz) -> None:
    """Write refined coordinates back. Assumes `ids` order matches `xyz` rows."""
    for i, p in zip(ids, xyz):
        rec.points3D[i].xyz = np.asarray(p, dtype=np.float64)
