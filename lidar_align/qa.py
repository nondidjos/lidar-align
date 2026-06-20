"""QA output: colour the SfM points by their point-to-plane distance to the LiDAR and
write a PLY. Open the before/after PLYs in CloudCompare/MeshLab to SEE whether the warp
flattened - on real data there is no ground truth, so this distance field is your evidence.
"""
from __future__ import annotations
import numpy as np


def _colormap(x):
    """Blue (near) -> green -> red (far) for x in [0,1]. Returns (N,3) uint8."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def write_ply(path, points, colors):
    points = np.asarray(points, np.float32)
    colors = np.asarray(colors, np.uint8)
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["r"], arr["g"], arr["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def residual_ply(rec, planes, path, dmax=None):
    """Write a residual-coloured PLY of the reconstruction's points; return (dist, dmax)."""
    from . import colmap_io
    _, X = colmap_io.world_points(rec)
    _, _, dist, _, _ = planes.query(X)
    if dmax is None:
        dmax = float(np.percentile(dist, 95)) or 1.0
    write_ply(path, X, _colormap(dist / max(dmax, 1e-9)))
    return dist, dmax


def residual_stats(dist):
    d = np.asarray(dist, float)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
               p95=float(np.percentile(d, 95)), max=float(d.max()))
