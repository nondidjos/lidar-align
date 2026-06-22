"""End-to-end integration test of the on-disk refinement pipeline.

Unlike tests/test_synth.py (which drives the in-memory solver `refine_reconstruction`
straight from numpy arrays), this exercises the *file I/O* path the real CLI uses:

    sparse_in/  (COLMAP .bin on disk)  +  lidar.las  (laspy out-of-core reader)
        -> lidar_align.refine.refine(...)   # prealign + refine + QA + XMP orchestrator
        -> sparse_refined/ (.bin)  +  QA .ply  +  RealityScan .xmp sidecars

It seeds the prealign with deliberately *noisy* correspondences, so the coarse Umeyama
step lands a few cm off and the LiDAR point-to-plane refinement has real work to do. The
test then asserts the full disk->disk pipeline recovers the global similarity onto the
LiDAR datum and writes every optional artifact, reloading the refined model FROM DISK to
score it (so colmap_io load+save, the LAS reader, prealign, refine, qa and export_xmp are
all on the path).

Run:  py -3.11 tests/test_integration.py
  or: .venv/Scripts/python tests/test_integration.py
"""
from __future__ import annotations
import os
os.environ.setdefault("GLOG_minloglevel", "3")   # silence ceres/glog chatter

import glob
import shutil
import sys
import tempfile

import numpy as np

# allow `python tests/test_integration.py` from anywhere (no PYTHONPATH needed)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pycolmap
from pycolmap import Rotation3d, Sim3d

from lidar_align.refine import refine as run_refine
from lidar_align.colmap_io import load as load_model
from lidar_align.export_xmp import read_xmp_pose


def _planar_lidar(points, step=0.01, half=2, seed=0):
    """A small symmetric planar patch centred on each true point, random orientation.
    centroid(patch) == the true point, so the local-PCA index fits an exact plane there."""
    rng = np.random.default_rng(seed)
    gu, gv = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    grid = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)
    out = []
    for Xi in points:
        n = rng.standard_normal(3); n /= np.linalg.norm(n)
        a = np.cross(n, [1.0, 0.0, 0.0]); a /= np.linalg.norm(a)
        b = np.cross(n, a)
        off = (grid[:, 0:1] * a + grid[:, 1:2] * b) * step
        out.append(Xi + off)
    return np.vstack(out)


def _write_las(path, pts):
    """Write an LAS so the test goes through the real laspy chunked reader, not arrays."""
    import laspy
    pts = np.asarray(pts, np.float64)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([1e-4, 1e-4, 1e-4])     # 0.1 mm quantisation: << patch step
    header.offsets = pts.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.write(str(path))


def test_pipeline_end_to_end():
    np.random.seed(0)
    work = tempfile.mkdtemp(prefix="lidar_align_it_")
    try:
        sparse_in = os.path.join(work, "sparse_in")
        sparse_out = os.path.join(work, "sparse_refined")
        qa_out = os.path.join(work, "qa")
        xmp_out = os.path.join(work, "xmp")
        lidar_las = os.path.join(work, "lidar.las")

        # 1. ground-truth reconstruction
        opts = pycolmap.SyntheticDatasetOptions()
        opts.num_rigs = 1; opts.num_cameras_per_rig = 1
        opts.num_frames_per_rig = 10; opts.num_points3D = 300; opts.track_length = 10
        rec = pycolmap.synthesize_dataset(opts)
        n_images = rec.num_reg_images()
        ids = list(rec.points3D.keys())
        truth = {p: np.array(rec.points3D[p].xyz, float) for p in ids}

        # 2. LiDAR reference -> real .las file (exercises the out-of-core LAS path)
        _write_las(lidar_las, _planar_lidar(np.array([truth[p] for p in ids])))

        # 3. perturb by a global similarity (invisible to reprojection BA; only the LiDAR
        #    datum can undo it) and write the perturbed model to disk as the SfM input.
        ang = np.deg2rad(2.0)
        sim = Sim3d(1.03, Rotation3d(np.array([0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)])),
                    np.array([0.10, -0.08, 0.12]))
        rec.transform(sim)
        os.makedirs(sparse_in, exist_ok=True)
        rec.write(sparse_in)
        perturbed = {p: np.array(rec.points3D[p].xyz, float) for p in ids}
        e_raw = float(np.mean([np.linalg.norm(perturbed[p] - truth[p]) for p in ids]))

        # 4. NOISY correspondences -> coarse Umeyama prealign lands a few cm off, leaving
        #    genuine point-to-plane work for the refiner (so this tests refine, not prealign).
        rng = np.random.default_rng(1)
        sample = ids[:: max(len(ids) // 12, 1)]
        corr = [[perturbed[p].tolist(),
                 (truth[p] + rng.normal(0, 0.02, 3)).tolist()] for p in sample]

        # 5. drive the on-disk orchestrator with EVERY optional output enabled
        run_refine(
            sparse_in=sparse_in, lidar=lidar_las, sparse_out=sparse_out,
            prealign=True, prealign_method="auto", correspondences=corr,
            w_lidar=20.0, huber=0.2, outer_iters=12, inner_iters=30,
            max_assoc_dist=0.5, planarity_min=0.05, anneal=True,
            qa_out=qa_out, xmp_out=xmp_out,
        )

        # 6. reload the refined model FROM DISK and score against truth. The pipeline now cleans
        #    low-quality/outlier points, so score only over the points that survived.
        ref = load_model(sparse_out)
        have = set(ref.points3D.keys())
        surviving = [p for p in ids if p in have]
        assert len(surviving) > 0.5 * len(ids), f"cleaning removed too many points ({len(surviving)}/{len(ids)})"
        e_final = float(np.mean([
            np.linalg.norm(np.array(ref.points3D[p].xyz, float) - truth[p]) for p in surviving]))
        print(f"mean error  raw={e_raw:.4f} m  ->  refined={e_final:.4f} m  "
              f"(ratio {e_final / e_raw:.3f})")

        assert e_final < 0.30 * e_raw, \
            f"pipeline failed to recover the similarity (ratio {e_final / e_raw:.3f})"
        assert e_final < 0.05, \
            f"did not converge near the LiDAR datum (e_final={e_final:.4f} m)"

        # 7. every on-disk artifact present and well-formed
        assert os.path.exists(os.path.join(sparse_out, "points3D.bin")), \
            "refined COLMAP model not written"
        for name in ("residual_before.ply", "residual_after.ply"):
            f = os.path.join(qa_out, name)
            assert os.path.getsize(f) > 0, f"missing/empty QA output {name}"
        xmps = glob.glob(os.path.join(xmp_out, "**", "*.xmp"), recursive=True)
        assert len(xmps) == n_images, f"expected {n_images} xmp sidecars, got {len(xmps)}"
        R_rc, _ = read_xmp_pose(xmps[0])                 # parses back -> proves valid XMP
        assert abs(np.linalg.det(R_rc) - 1) < 1e-6, "xmp rotation is not a valid rotation"

        print(f"INTEGRATION E2E: PASS  ({n_images} imgs; refined model + "
              f"{len(xmps)} xmp + QA plys on disk)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_georeferenced_recenter():
    """Georeferenced LiDAR (UTM) -> the aligned model lands at million-metre coords, which RS can't
    place (cameras off-screen, scan auto-tilted). The pipeline must recenter the export to a local
    origin, write coordinate_offset.txt, keep the .xmp positions small, and round-trip exactly:
    local_camera_position + offset == the true georeferenced camera centre."""
    from lidar_align.export_xmp import camera_pose, read_xmp_pose
    np.random.seed(0)
    work = tempfile.mkdtemp(prefix="lidar_align_geo_")
    try:
        sparse_in = os.path.join(work, "sparse_in")
        sparse_out = os.path.join(work, "sparse_refined")
        xmp_out = os.path.join(work, "xmp")
        lidar_las = os.path.join(work, "lidar.las")
        UTM = np.array([500000.0, 4500000.0, 60.0])      # a realistic georeferenced origin

        opts = pycolmap.SyntheticDatasetOptions()
        opts.num_rigs = 1; opts.num_cameras_per_rig = 1
        opts.num_frames_per_rig = 10; opts.num_points3D = 300; opts.track_length = 10
        rec = pycolmap.synthesize_dataset(opts)
        ids = list(rec.points3D.keys())
        truth = {p: np.array(rec.points3D[p].xyz, float) for p in ids}
        truth_cam = {iid: camera_pose(im)[1] for iid, im in rec.images.items() if im.has_pose}

        # LiDAR sits in UTM; the SfM model is in its own small frame (as COLMAP would produce).
        _write_las(lidar_las, _planar_lidar(np.array([truth[p] + UTM for p in ids])))
        ang = np.deg2rad(2.0)
        rec.transform(Sim3d(1.03, Rotation3d(np.array([0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)])),
                            np.array([0.10, -0.08, 0.12])))
        os.makedirs(sparse_in, exist_ok=True); rec.write(sparse_in)
        perturbed = {p: np.array(rec.points3D[p].xyz, float) for p in ids}
        rng = np.random.default_rng(1)
        sample = ids[:: max(len(ids) // 12, 1)]
        corr = [[perturbed[p].tolist(), (truth[p] + UTM + rng.normal(0, 0.02, 3)).tolist()]
                for p in sample]

        run_refine(sparse_in=sparse_in, lidar=lidar_las, sparse_out=sparse_out,
                   prealign=True, prealign_method="auto", correspondences=corr,
                   w_lidar=20.0, huber=0.2, outer_iters=12, inner_iters=30,
                   max_assoc_dist=0.5, planarity_min=0.05, anneal=True, xmp_out=xmp_out)

        # offset file written next to BOTH the model and the xmp sidecars
        for d in (sparse_out, xmp_out):
            assert os.path.exists(os.path.join(d, "coordinate_offset.txt")), f"no offset file in {d}"
        last = open(os.path.join(xmp_out, "coordinate_offset.txt")).read().strip().splitlines()[-1]
        offset = np.array([float(x) for x in last.split()[1:]])
        assert np.linalg.norm(offset) > 1e5, f"offset {offset} is not georeferenced-scale"

        name2id = {os.path.splitext(im.name)[0]: iid
                   for iid, im in rec.images.items() if im.has_pose}
        xmps = glob.glob(os.path.join(xmp_out, "**", "*.xmp"), recursive=True)
        errs = []
        for xf in xmps:
            _, C = read_xmp_pose(xf)
            assert np.linalg.norm(C) < 1e4, f"xmp position not local: |C|={np.linalg.norm(C):.0f} m"
            base = os.path.splitext(os.path.relpath(xf, xmp_out))[0].replace("\\", "/")
            iid = name2id.get(base, name2id.get(os.path.basename(base)))
            if iid is not None:
                errs.append(np.linalg.norm((C + offset) - (truth_cam[iid] + UTM)))
        e = float(np.mean(errs))
        assert e < 0.05, f"georeference round-trip error {e:.4f} m (local pos + offset != true UTM pose)"
        print(f"GEOREF RECENTER: PASS  offset {offset.tolist()} (UTM-scale), {len(xmps)} xmp local, "
              f"round-trip err {e * 100:.2f} cm")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_end_to_end()
    test_georeferenced_recenter()
