"""End-to-end proof with genuinely planar LiDAR geometry.

A global similarity (scale+rotation+translation) is invisible to reprojection BA, so only
the LiDAR point-to-plane datum can undo it. We synthesize a ground-truth reconstruction,
build a LiDAR cloud of small planar patches centred on each true point (so the local-PCA
index has real planes to fit), apply the similarity, then refine and check it snaps back.

Run:  GLOG_minloglevel=3 PYTHONPATH=. .venv/Scripts/python.exe tests/test_synth.py
"""
import numpy as np
import pycolmap
from pycolmap import Rotation3d, Sim3d

from lidar_align.lidar_index import LidarPlanes
from lidar_align.refine import refine_reconstruction

np.random.seed(0)

# 1. ground-truth reconstruction
opts = pycolmap.SyntheticDatasetOptions()
opts.num_rigs = 1; opts.num_cameras_per_rig = 1
opts.num_frames_per_rig = 10; opts.num_points3D = 300; opts.track_length = 10
rec = pycolmap.synthesize_dataset(opts)
print(f"synth: {rec.num_reg_images()} images, {rec.num_points3D()} points")

ids = list(rec.points3D.keys())
xyz_true = {p: np.array(rec.points3D[p].xyz, float) for p in ids}
P = np.array([xyz_true[p] for p in ids])

# 2. LiDAR reference: a small symmetric planar patch on each true point (centroid == truth)
step = 0.01
gu, gv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))   # 5x5 symmetric grid
grid = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)
patches = []
for Xi in P:
    n = np.random.randn(3); n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    off = (grid[:, 0:1] * a + grid[:, 1:2] * b) * step
    patches.append(Xi + off)
lidar = np.vstack(patches)
planes = LidarPlanes(lidar, k_plane=16)

# 3. global similarity perturbation (reprojection-invariant; only LiDAR can fix it)
ang = 0.015
sim = Sim3d(1.02, Rotation3d(np.array([0.0, 0.0, np.sin(ang/2), np.cos(ang/2)])),
            np.array([0.04, -0.03, 0.05]))
rec.transform(sim)

def mean_err():
    return float(np.mean([np.linalg.norm(rec.points3D[p].xyz - xyz_true[p]) for p in ids]))

e0 = mean_err()
print(f"error to truth BEFORE refine: {e0:.4f}")

# 4. refine against the LiDAR datum (annealing + planarity gating on)
refine_reconstruction(rec, planes, w_lidar=20.0, huber=0.2,
                      outer_iters=12, inner_iters=30, max_assoc_dist=0.5,
                      planarity_min=0.05, anneal=True, fix_intrinsics=True, verbose=False)

e1 = mean_err()
print(f"error to truth AFTER  refine: {e1:.4f}   (ratio {e1/e0:.3f})")

assert e1 < 0.25 * e0, f"LiDAR datum failed to recover the similarity (ratio {e1/e0:.3f})"
print("SYNTHETIC E2E: PASS  - LiDAR planes recovered the global similarity")
