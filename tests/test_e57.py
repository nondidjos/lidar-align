"""e57 ingest + conversion.

lidar_index now reads .e57 directly (applying each scan's pose), and scripts/e57_to_laz.py
converts e57 -> LAZ. This builds a synthetic TWO-scan .e57 with pye57 using translation-only
poses (identity quaternion is convention-proof), so global == local + t, then checks:
  1. the loader reads the e57 and recovers the global truth (poses applied, scans merged),
  2. crop-to-volume drops the far scan,
  3. the converter round-trips e57 -> LAZ within 1 mm quantisation.

Run:  py -3.11 tests/test_e57.py
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pye57
from lidar_align.lidar_index import _load_points


def _sorted(a):
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def _write_scan(e, global_pts, translation):
    """Store scanner-local coords so read_scan(transform=True) recovers `global_pts`
    (identity rotation + translation pose)."""
    loc = global_pts - np.asarray(translation, float)
    e.write_scan_raw({"cartesianX": np.ascontiguousarray(loc[:, 0]),
                      "cartesianY": np.ascontiguousarray(loc[:, 1]),
                      "cartesianZ": np.ascontiguousarray(loc[:, 2])},
                     rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                     translation=np.asarray(translation, float))


def test_e57_ingest_and_convert():
    work = tempfile.mkdtemp(prefix="lidar_align_e57_")
    try:
        rng = np.random.default_rng(0)
        g0 = rng.uniform([100, 100, 0], [105, 105, 3], size=(4000, 3))   # scan 0 volume
        g1 = rng.uniform([130, 100, 0], [135, 105, 3], size=(4000, 3))   # scan 1, far away
        truth = np.vstack([g0, g1])

        e57_path = os.path.join(work, "site.e57")
        e = pye57.E57(e57_path, mode="w")
        _write_scan(e, g0, [101.0, 102.0, 0.5])
        _write_scan(e, g1, [131.0, 102.0, 0.5])
        del e                                       # finalise/flush the file

        # 1. loader reads .e57 directly, applies poses, merges scans -> global truth
        pts = _load_points(e57_path)
        assert pts.shape == truth.shape, f"e57 read {pts.shape} != truth {truth.shape}"
        assert np.allclose(_sorted(pts), _sorted(truth), atol=1e-6), "e57 global coords wrong"

        # 2. crop to scan-0 volume keeps scan 0, drops the far scan (1 mm margin absorbs
        #    the ~1e-13 float error from the local<->global round-trip at the box edge)
        crop = _load_points(e57_path, crop_aabb=(g0.min(0), g0.max(0)), crop_margin=1e-3)
        assert len(crop) == len(g0), f"crop kept {len(crop)} != {len(g0)} (far scan leaked?)"

        # 3. converter e57 -> LAZ, reload, round-trip within 1 mm quantisation
        from scipy.spatial import cKDTree
        from scripts.e57_to_laz import convert
        laz = os.path.join(work, "out.laz")
        convert(e57_path, laz)
        back = _load_points(laz)
        assert back.shape == truth.shape, f"laz {back.shape} != truth {truth.shape}"
        # nearest-neighbour, not sorted rows: 1 mm LAZ quantisation can reorder near-ties
        dmax = cKDTree(truth).query(back)[0].max()
        assert dmax < 2e-3, f"laz round-trip max NN dist {dmax * 1000:.2f} mm > 2 mm"

        print(f"E57 INGEST + CONVERT: PASS  ({len(truth):,} pts, 2 scans; "
              f"e57-direct + crop + laz round-trip)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_e57_ingest_and_convert()
