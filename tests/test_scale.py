"""Capping LiDAR residuals preserves accuracy: a spatially-even subset still pins the
solve. (Speed is shown separately by the benchmark; this test guards correctness.)"""
import numpy as np
import pycolmap
from pycolmap import Rotation3d, Sim3d

from lidar_align.lidar_index import LidarPlanes
from lidar_align.refine import refine_reconstruction


def build():
    np.random.seed(2)
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs = 1; o.num_cameras_per_rig = 1
    o.num_frames_per_rig = 10; o.num_points3D = 2000; o.track_length = 10
    rec = pycolmap.synthesize_dataset(o)
    ids = list(rec.points3D.keys())
    P = np.array([rec.points3D[p].xyz for p in ids])
    step = 0.01
    gu, gv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    grid = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)
    patches = []
    for Xi in P:
        n = np.random.randn(3); n /= np.linalg.norm(n)
        a = np.cross(n, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
        patches.append(Xi + (grid[:, 0:1] * a + grid[:, 1:2] * b) * step)
    planes = LidarPlanes(np.vstack(patches), k_plane=16)
    ang = 0.015
    rec.transform(Sim3d(1.02, Rotation3d(np.array([0, 0, np.sin(ang/2), np.cos(ang/2)])),
                        np.array([0.04, -0.03, 0.05])))
    truth = {p: P[k] for k, p in enumerate(ids)}
    return rec, planes, truth, ids


def err(rec, truth, ids):
    return float(np.mean([np.linalg.norm(rec.points3D[p].xyz - truth[p]) for p in ids]))


if __name__ == "__main__":            # guard so the benchmark can import build()/err()
    rec, planes, truth, ids = build()
    e0 = err(rec, truth, ids)
    refine_reconstruction(rec, planes, w_lidar=20.0, huber=0.2, outer_iters=10,
                          inner_iters=25, max_assoc_dist=0.5, planarity_min=0.05,
                          max_lidar_residuals=800, anneal=True, fix_intrinsics=True,
                          verbose=False)
    e1 = err(rec, truth, ids)
    print(f"2000 pts, capped to 800 ties: error {e0:.4f} -> {e1:.4f} (ratio {e1/e0:.3f})")
    assert e1 < 0.25 * e0, f"capped refine lost accuracy (ratio {e1/e0:.3f})"
    print("SCALE/CAP TEST: PASS")
