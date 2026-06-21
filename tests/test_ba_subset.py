"""Regression tests for the scaling fixes that make the align actually complete on real data:

1. _voxel_pick is outlier-robust: a few stray SfM points must not collapse the candidate grid
   to a handful of cells (the bug that left ~1900 candidates out of millions -> a weak gauge).
2. _subsample_model reduces a reconstruction to ~target points.
3. A tie-heavy refine COMPLETES quickly and recovers the camera poses. With the old
   multi-threaded Ceres + Python LiDAR cost this deadlocked/crawled; num_threads=1 fixes it.

Run:  GLOG_minloglevel=3 PYTHONPATH=. .venv/Scripts/python.exe tests/test_ba_subset.py
"""
from __future__ import annotations
import os
import sys
import tempfile
import shutil
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pycolmap
import laspy
from pycolmap import Rotation3d, Sim3d

from lidar_align.refine import _voxel_pick, _subsample_model, _spatial_subsample, refine
from lidar_align.export_xmp import camera_pose


def test_spatial_subsample_outlier_robust():
    # 60k real points in a ~60 m box + a few junk points flung tens of km out (real SfM data).
    # Raw min/max grid sizing would collapse all ties to a handful; the percentile grid must not.
    rng = np.random.default_rng(0)
    real = rng.uniform(-30, 30, size=(60000, 3))
    out = rng.uniform(-53000, 53000, size=(80, 3))
    X = np.vstack([real, out]); plan = rng.uniform(0, 1, len(X)); idx = np.arange(len(X))
    sel = _spatial_subsample(X, plan, idx, 30000)
    assert len(sel) > 15000, f"outliers collapsed tie selection: only {len(sel)} of 30000"
    print(f"  spatial_subsample: {len(sel):,} ties kept despite 80 km-scale outliers")


def test_voxel_pick_outlier_robust():
    rng = np.random.default_rng(0)
    bulk = rng.uniform(0, 1, size=(200_000, 3))                 # dense bulk in a unit cube
    outliers = rng.uniform(-1e6, 1e6, size=(50, 3))             # a few far-flung strays
    X = np.vstack([bulk, outliers])
    keep = _voxel_pick(X, 20_000)
    # raw min/max would size the cell ~1e6 -> the bulk collapses into ~1 voxel (the old bug).
    # The percentile-based grid must keep a healthy fraction instead.
    assert len(keep) > 5_000, f"outliers collapsed the grid: only {len(keep)} candidates"
    print(f"  voxel_pick: {len(keep):,} candidates kept despite 50 extreme outliers")


def test_subsample_model():
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs = 1; o.num_cameras_per_rig = 1
    o.num_frames_per_rig = 10; o.num_points3D = 40_000; o.track_length = 6
    rec = pycolmap.synthesize_dataset(o)
    kept = _subsample_model(rec, 8_000)
    assert kept == rec.num_points3D()
    assert kept <= 8_000 + 1 and kept > 1_000, f"subsample kept {kept}"
    print(f"  subsample_model: 40,000 -> {kept:,} points")


def test_tie_heavy_refine_completes_and_recovers():
    """The hang regression: enough LiDAR ties that the old multi-threaded solve crawled.
    Must finish fast (single-threaded) and pull the camera poses back toward truth."""
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs = 1; o.num_cameras_per_rig = 1
    o.num_frames_per_rig = 20; o.num_points3D = 20_000; o.track_length = 8
    rec = pycolmap.synthesize_dataset(o)
    truth = {im.image_id: camera_pose(im) for im in rec.images.values() if im.has_pose}
    P = np.array([rec.points3D[i].xyz for i in rec.points3D], float)
    rng = np.random.default_rng(3)
    grid = np.column_stack([x.ravel() for x in np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))]).astype(float)
    pat = []
    for Xi in P:
        n = rng.standard_normal(3); n /= np.linalg.norm(n)
        a = np.cross(n, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
        pat.append(Xi + (grid[:, 0:1] * a + grid[:, 1:2] * b) * 0.01)
    work = tempfile.mkdtemp(prefix="ba_subset_")
    try:
        lid = np.vstack(pat)
        h = laspy.LasHeader(point_format=3, version="1.2"); h.scales = [1e-4] * 3; h.offsets = lid.min(0)
        las = laspy.LasData(h); las.x, las.y, las.z = lid[:, 0], lid[:, 1], lid[:, 2]
        las.write(os.path.join(work, "l.las"))
        rec.transform(Sim3d(1.02, Rotation3d(np.array([0, 0, np.sin(.01), np.cos(.01)])),
                            np.array([.05, -.04, .06])))
        si = os.path.join(work, "si"); os.makedirs(si); rec.write(si)

        def err(r):
            return float(np.mean([np.linalg.norm(camera_pose(im)[1] - truth[im.image_id][1])
                                  for im in r.images.values() if im.has_pose and im.image_id in truth]))
        e0 = err(pycolmap.Reconstruction(si))
        t = time.time()
        refine(sparse_in=si, lidar=os.path.join(work, "l.las"),
               sparse_out=os.path.join(work, "so"), prealign=False,
               w_lidar=20.0, huber=0.2, outer_iters=8, inner_iters=25,
               max_assoc_dist=0.5, planarity_min=0.05,
               max_lidar_residuals=15000, ba_max_points=10000, verbose=False)
        dt = time.time() - t
        e1 = err(pycolmap.Reconstruction(os.path.join(work, "so")))
        # must complete fast (old multi-threaded code would not) and improve the poses
        assert dt < 90, f"refine took {dt:.0f}s - did it hit the threading stall again?"
        assert e1 < e0 * 0.6, f"poses barely moved: {e0:.4f} -> {e1:.4f}"
        print(f"  tie-heavy refine: cam err {e0:.4f} -> {e1:.4f} m in {dt:.1f}s (no stall)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_voxel_pick_outlier_robust()
    test_spatial_subsample_outlier_robust()
    test_subsample_model()
    test_tie_heavy_refine_completes_and_recovers()
    print("BA SUBSET/SCALE TEST: PASS")
