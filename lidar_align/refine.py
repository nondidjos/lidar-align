"""Joint refinement: pycolmap's reprojection bundle adjustment + LiDAR point-to-plane
residuals, so a survey-grade scan becomes the datum the photogrammetry bends onto.

Design (verified against pycolmap 4.0.4 / pyceres 2.6):
  - We do NOT hand-roll the reprojection problem. pycolmap's `CeresBundleAdjuster` already
    builds it with the correct camera models and the quat+translation pose manifold (a 7-D
    block that pyceres has no ready-made product manifold for). We grab its `.problem` and
    *add* our LiDAR residuals to the same Ceres problem - exactly how colmap-pcd modifies
    colmap's BA - then `.solve()`.
  - The gauge is left UNSPECIFIED (unfixed): the LiDAR point-to-plane terms supply the
    datum (position, orientation, scale), which is the whole point. Coarse pre-align the
    model into the LiDAR frame once before refining (see README).
  - point.xyz is a writable reference shared by both the reprojection and LiDAR costs, so
    the solve updates the reconstruction in place.

NOTE on GPU/CUDA warnings: The pip-installed pyceres wheels do NOT include CUDA/cuDSS support.
CerES will print warnings like "Requested GPU for bundle adjustment, but Ceres was compiled
without CUDA support" - this is expected! The solver runs on CPU and works correctly.
For GPU acceleration, compile Ceres from source with CUDA (advanced, rarely needed).
"""
from __future__ import annotations
import os
import numpy as np
import pyceres
import pycolmap

import time

from . import colmap_io
from .lidar_index import LidarPlanes


def _rss_gb():
    """Resident memory in GB (psutil if available, else Windows ctypes, else -1)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = _PMC(); c.cb = ctypes.sizeof(_PMC)
        if ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.WorkingSetSize / 1e9
    except Exception:
        pass
    return -1.0


class _Stages:
    """Print '[stage +M:SS  RAM X.XGB] label' markers so a run shows where time and memory go."""
    def __init__(self, enabled=True):
        self.t0 = time.time()
        self.enabled = enabled

    def __call__(self, label):
        if not self.enabled:
            return
        el = int(time.time() - self.t0)
        rss = _rss_gb()
        ram = f"{rss:.1f}GB" if rss >= 0 else "n/a"
        print(f"[stage +{el // 60:d}:{el % 60:02d}  RAM {ram}] {label}")


class PointToPlaneCost(pyceres.CostFunction):
    """residual = w * n . (X - p)   - one residual, one parameter block (the 3D point)."""

    def __init__(self, plane_pt, plane_n, weight):
        super().__init__()
        self.p = np.asarray(plane_pt, dtype=np.float64)
        self.n = np.asarray(plane_n, dtype=np.float64)
        self.w = float(weight)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3])

    def Evaluate(self, parameters, residuals, jacobians):
        X = np.asarray(parameters[0], dtype=np.float64)
        residuals[0] = self.w * float(self.n @ (X - self.p))
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = self.w * self.n  # d/dX (n.(X-p)) = n
        return True


def _make_adjuster(rec, fix_intrinsics, inner_iters):
    """A pycolmap CeresBundleAdjuster over all registered images + points, gauge unfixed."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = not fix_intrinsics
    opts.refine_principal_point = not fix_intrinsics
    opts.refine_extra_params = not fix_intrinsics
    opts.refine_points3D = True
    opts.refine_rig_from_world = True
    opts.print_summary = False
    try:
        opts.ceres.solver_options.max_num_iterations = int(inner_iters)
        # CRITICAL: our LiDAR PointToPlaneCost is a Python cost function. With more than one Ceres
        # thread the worker threads contend savagely on Python's GIL - measured on a 50k-point
        # model with 2000 ties: 13 s at num_threads=1 vs >180 s at num_threads=2. So the solve
        # MUST be single-threaded. (The reprojection part loses multicore; we offset that by
        # subsampling the model via ba_max_points so the factorization stays small.)
        opts.ceres.solver_options.num_threads = 1
    except Exception:
        pass  # fall back to pycolmap defaults if the options layout differs

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        config.add_image(image_id)
    # Points observed by the added images are optimised by default - we do NOT loop
    # add_variable_point over every point. That loop was a redundant 3.7M-iteration Python
    # call that held the GIL (freezing the UI) and added minutes to the build, with zero
    # effect on the result (verified: the synth case recovers identically without it).
    # LiDAR is the datum -> do not let BA pin the gauge to the arbitrary SfM frame.
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.UNSPECIFIED)

    return pycolmap.create_default_ceres_bundle_adjuster(opts, config, rec)


def _cuda_check_once():
    """Print one helpful note if Ceres GPU solvers aren't available (pyceres wheels lack CUDA)."""
    if hasattr(_cuda_check_once, "_done"):
        return
    _cuda_check_once._done = True
    # pyceres wheels don't include CUDA; this warning comes from Ceres itself.
    # Print a friendly note instead of letting users panic about the warning.
    print("[note] Ceres GPU solvers unavailable (pyceres wheel lacks CUDA/cuDSS). "
          "The solver runs on CPU - this is normal and works fine.")


def _spatial_subsample(X, plan, idx, cap):
    """Pick <= cap of the kept indices, spatially even (one per voxel) and favouring the
    most-planar association per voxel. You don't need a residual on every point - a few
    thousand well-distributed planar ties pin the 7-DOF gauge and the warp just as well,
    and it keeps the (Python) LiDAR cost from dominating at 7k-image / million-point scale."""
    if cap is None or len(idx) <= cap:
        return idx
    pts = X[idx]
    # Size the grid from the 1-99 percentile extent, NOT raw min/max: a few junk SfM points
    # flung kilometres out would otherwise blow the cell size up so the whole real model lands
    # in one voxel, collapsing thousands of ties down to a handful. Clip keys into the grid so
    # those outliers just share edge cells.
    lo = np.percentile(pts, 1.0, axis=0)
    hi = np.percentile(pts, 99.0, axis=0)
    B = max(round(cap ** (1.0 / 3.0)), 1) + 1
    cell = np.maximum(hi - lo, 1e-9) / B
    keys = np.clip(np.floor((pts - lo) / cell), 0, B - 1).astype(np.int64)
    order = np.argsort(-plan[idx])              # most-planar first
    ki = keys[order]
    span = [int(v) + 1 for v in ki.max(0)]     # keys are >= 0 already (shifted by lo)
    if span[0] * span[1] * span[2] < (1 << 62):   # one packed key -> one sort, not a row lexsort
        packed = (ki[:, 0] * span[1] + ki[:, 1]) * span[2] + ki[:, 2]
        _, first = np.unique(packed, return_index=True)
    else:
        _, first = np.unique(ki, axis=0, return_index=True)
    return idx[order[first]]


def _voxel_pick(X, cap):
    """Up to ~`cap` point indices, one per voxel (spatially even). No-op when len(X) <= cap.

    Outlier-robust: the grid is sized from the 1-99 percentile extent, not raw min/max, so a
    handful of stray SfM points can't blow the cell size up and collapse the whole cloud into a
    few voxels (which previously left only ~1900 candidates out of millions). Keys are clipped
    into the grid so outliers just share the edge cells.
    """
    n = len(X)
    if cap is None or n <= cap:
        return np.arange(n)
    lo = np.percentile(X, 1.0, axis=0)
    hi = np.percentile(X, 99.0, axis=0)
    B = max(round(cap ** (1.0 / 3.0)), 1) + 1
    cell = np.maximum(hi - lo, 1e-9) / B
    keys = np.clip(np.floor((X - lo) / cell), 0, B - 1).astype(np.int64)
    packed = (keys[:, 0] * B + keys[:, 1]) * B + keys[:, 2]   # mixed-radix, collision-free
    _, first = np.unique(packed, return_index=True)
    return first


def _subsample_model(rec, target):
    """Delete all but ~`target` spatially-even points so the bundle adjustment is tractable.

    A reprojection BA over millions of points factorizes a matrix that blows up super-linearly
    (and pycolmap gives no way to swap in an iterative solver). But the camera poses we export
    are well-constrained by a few hundred thousand well-distributed points + their observations,
    so we keep a representative subset and drop the rest. Deletion is fast (~750k pts/s).
    Returns the kept count.
    """
    ids = np.array(list(rec.points3D.keys()), dtype=np.int64)
    X = np.array([p.xyz for p in rec.points3D.values()], np.float64)
    keep = set(int(i) for i in ids[_voxel_pick(X, target)])
    for pid in ids.tolist():
        if pid not in keep:
            rec.delete_point3D(pid)
    return rec.num_points3D()


def refine_reconstruction(rec, planes, w_lidar=5.0, huber=0.1,
                          outer_iters=5, inner_iters=50, max_assoc_dist=0.5,
                          planarity_min=0.1, anneal=True, max_lidar_residuals=30000,
                          fix_intrinsics=True, verbose=True, cancel_cb=None,
                          early_stop_tol=0.0, prev_err=None, stage=None):
    """Refine an in-memory reconstruction against a LidarPlanes index. Mutates `rec`.

    Annealing (when `anneal`): early rounds use a loose association radius, soft robust
    loss and gentle LiDAR weight so a rough pre-align can settle; later rounds tighten
    the radius, sharpen the loss and ramp the weight up so the scan fully dominates.
    Associations on non-planar neighbourhoods (`planarity <= planarity_min`) are dropped.
    `max_lidar_residuals` caps the per-round LiDAR ties (spatially even) for speed at scale.
    
    `early_stop_tol`: stop if median pt2plane residual improvement < this (metres).
    `prev_err`: previous median error for early stopping comparison (meters).
    """
    prev_med = prev_err if prev_err is not None else float("inf")
    # Build the bundle adjuster (the reprojection problem over ALL images + points) ONCE and
    # reuse it every round - rebuilding it per round re-creates millions of residual blocks
    # outer_iters times. Each round we only swap the (<= cap) LiDAR point-to-plane blocks.
    _stage = stage if stage is not None else (lambda *a: None)
    ba = _make_adjuster(rec, fix_intrinsics, inner_iters)
    problem = ba.problem
    lidar_blocks = []
    _stage("built bundle adjuster")
    # Cap the per-round nearest-plane query: a few thousand well-spread ties pin the gauge and
    # warp; querying every point each round is wasted k-NN + PCA work at 7k-image scale.
    cand_cap = max(8 * max_lidar_residuals, 200_000)
    _cuda_check_once()

    # Reading every point's coordinates each round (world_points over millions of points) costs
    # ~24 s/round at 7k-image scale and we keep only ~cand_cap of them. Instead read all points
    # ONCE to choose a spatially-even candidate set, keep the live Point3D handles, and each
    # round read just that set's CURRENT coordinates. cand_pts[k].xyz stays a writable reference
    # the solve mutates in place, exactly like rec.points3D[id].xyz. For sets <= cand_cap the
    # candidate set is every point, so behaviour is identical to the per-round full read.
    all_objs = list(rec.points3D.values())
    X_all = np.array([p.xyz for p in all_objs], np.float64)
    cand = _voxel_pick(X_all, cand_cap)
    cand_pts = [all_objs[k] for k in cand]
    n_cand = len(cand_pts)
    del all_objs, X_all
    _stage(f"picked {n_cand:,} candidate points")

    for it in range(outer_iters):
        if cancel_cb is not None and cancel_cb():        # cooperative stop between rounds
            if verbose:
                print(f"[cancelled before outer {it}]")
            break
        for bid in lidar_blocks:                          # drop last round's LiDAR ties
            problem.remove_residual_block(bid)
        lidar_blocks = []

        frac = it / (outer_iters - 1) if (anneal and outer_iters > 1) else 1.0
        assoc = max_assoc_dist * (4.0 ** (1.0 - frac))   # loose -> tight
        w = w_lidar * (0.3 + 0.7 * frac)                  # gentle -> full
        hub = huber * (3.0 ** (1.0 - frac))               # soft -> sharp

        X = np.array([p.xyz for p in cand_pts], np.float64)   # current coords of the candidate set
        C, N, dist, plan, nn = planes.query(X)
        # gate: close to the plane AND actually near a scan point (no edge-bleed) AND planar
        keep = (dist < assoc) & (nn < assoc) & (plan > planarity_min)
        keep_idx = np.flatnonzero(keep)

        # With gauge UNSPECIFIED the LiDAR residuals are the only datum; a round with no
        # associations would leave the solve rank-deficient. Skip it rather than diverge.
        if len(keep_idx) == 0:
            if verbose:
                print(f"[outer {it}] no associations kept "
                      f"(loosen planarity_min / max_assoc_dist) - skipped")
            continue

        sel = _spatial_subsample(X, plan, keep_idx, max_lidar_residuals)

        # Too few residuals -> poorly aligned or over-filtered. Relaxing the planarity gate
        # in the early rounds is a real behaviour change, so it must not depend on `verbose`.
        if len(sel) < 100:
            if verbose:
                print(f"[outer {it}] only {len(sel)} LiDAR residuals - consider looser "
                      f"planarity_min ({planarity_min}) or larger max_assoc_dist ({max_assoc_dist})")
            if len(sel) < 10 and it < 2 and planarity_min > 0.05:
                if verbose:
                    print(f"[outer {it}] auto-relaxing planarity gate to 0.05")
                keep_idx = np.flatnonzero((plan > 0.05) & (dist < assoc) & (nn < assoc))
                sel = _spatial_subsample(X, plan, keep_idx, max_lidar_residuals)

        # residual is w * n.(X-c); Huber transitions at |r|=arg, so the arg must be w*hub for
        # the robust threshold to be `hub` METRES of point-to-plane distance (else it's hub/w).
        loss = pyceres.HuberLoss(hub * w)
        for k in sel:
            bid = problem.add_residual_block(
                PointToPlaneCost(C[k], N[k], w), loss, [cand_pts[k].xyz])
            lidar_blocks.append(bid)
        if verbose:
            print(f"[outer {it}] solving… ({len(sel):,} lidar ties)")
        ba.solve()
        _stage(f"round {it} solved")

        # residual stats (used by both the log line and early stopping, so always computed)
        dk = dist[sel]
        med = float(np.median(dk)) if len(dk) else float("nan")
        if verbose:
            p95 = float(np.percentile(dk, 95)) if len(dk) else float("nan")
            print(f"[outer {it}] residuals {len(sel):,} (of {len(keep_idx):,} gated / "
                  f"{n_cand:,} cand)  w={w:.1f} huber={hub:.3f} assoc<{assoc:.2f}m  "
                  f"pt2plane med={med:.3f} p95={p95:.3f} m")

        # Early stopping: negligible improvement between rounds -> converged.
        if early_stop_tol > 0 and len(dk) and np.isfinite(med):
            if prev_med - med < early_stop_tol:
                if verbose:
                    print(f"[early stop] round {it}: improvement < {early_stop_tol} m")
                break
            prev_med = med
    return rec


def _sfm_aabb(rec):
    P = np.array([p.xyz for p in rec.points3D.values()], np.float64)   # .values() avoids re-lookup
    return P.min(0), P.max(0)


def _camera_centers(rec):
    out = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        M = np.asarray(im.cam_from_world().matrix(), float)
        out.append(-M[:3, :3].T @ M[:3, 3])
    return np.asarray(out) if out else np.zeros((0, 3))


def _camera_spread(rec):
    """Robust extent of the camera centres (2-98 percentile box diagonal, in metres). A near-zero
    value means the poses collapsed to a point - a divergent free-gauge solve."""
    C = _camera_centers(rec)
    if len(C) < 2:
        return 0.0
    lo, hi = np.percentile(C, [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def _median_p2plane(rec, planes, sample=20000):
    """Median |point-to-plane| (metres) over a sample of model points - the scalar used to judge
    whether the refine actually improved the alignment or made it worse."""
    ids = list(rec.points3D.keys())
    if not ids:
        return float("nan")
    idx = (np.random.default_rng(0).choice(len(ids), sample, replace=False)
           if len(ids) > sample else np.arange(len(ids)))
    X = np.array([rec.points3D[ids[i]].xyz for i in idx], np.float64)
    _, _, d, _, _ = planes.query(X)
    return float(np.median(np.abs(d)))


def refine(sparse_in, lidar, sparse_out,
           w_lidar=5.0, huber=0.1, outer_iters=5, inner_iters=50,
           voxel=None, k_plane=16, max_assoc_dist=0.5, planarity_min=0.1,
           anneal=True, max_lidar_residuals=30000, fix_intrinsics=True,
           prealign=False, prealign_voxel=0.5, prealign_method="auto",
           correspondences=None, crop_margin=2.0, planes=None,
           qa_out=None, xmp_out=None, xmp_pose_prior="locked", xmp_axis_flip=None,
           cancel_cb=None, early_stop_tol=0.0, verbose=True, ba_max_points=300_000):
    from .lidar_index import _load_points

    stage = _Stages(enabled=verbose)
    rec = colmap_io.load(sparse_in)
    print(f"loaded {rec.num_reg_images()} images, {rec.num_points3D()} points")
    print(f"camera spread (as loaded, SfM frame): {_camera_spread(rec):.2f}")
    stage(f"loaded reconstruction ({rec.num_reg_images()} imgs, {rec.num_points3D():,} pts)")

    # A full reprojection BA over millions of points doesn't scale (the matrix factorization
    # blows up and pycolmap won't let us pick an iterative solver), so reduce to a representative
    # subset up front. The exported camera poses stay well-constrained; the saved sparse model is
    # this subset. Set ba_max_points=None/0 to keep every point (only viable for small models).
    if ba_max_points and rec.num_points3D() > ba_max_points:
        before = rec.num_points3D()
        print(f"reducing model {before:,} -> ~{ba_max_points:,} points so the bundle adjustment "
              f"is tractable (poses stay constrained; saved model is this subset)…")
        kept = _subsample_model(rec, ba_max_points)
        print(f"  reduced to {kept:,} points")
        stage(f"reduced model to {kept:,} pts")

    if prealign:
        from .prealign import prealign_reconstruction
        print(f"pre-align: loading coarse cloud ({prealign_voxel} m voxel)…")
        coarse = _load_points(lidar, voxel=prealign_voxel,
                              log=(print if verbose else None), cancel_cb=cancel_cb)
        info = prealign_reconstruction(rec, coarse, correspondences=correspondences,
                                       method=prealign_method, voxel=prealign_voxel)
        print(f"prealign[{prealign_method}]: scale={info['scale']:.4f} "
              f"fitness={info['fitness']:.3f} rmse={info['inlier_rmse']:.3f}")
        stage("pre-align done")

    # units sanity: SfM extent (post-prealign) should match the LiDAR's metric scale
    lo, hi = _sfm_aabb(rec)
    print(f"SfM extent (post-prealign): {np.round(hi - lo, 2)} units "
          f"-> metric params (assoc/crop in same units)")

    if planes is None:
        # This load + voxel + KD-tree build is the longest silent stage on a big raw cloud;
        # stream progress and honour Stop so it isn't a multi-hour black hole.
        print("building plane index: loading cloud, voxel-merging, KD-tree…")
        planes = LidarPlanes.from_file(lidar, voxel=voxel, crop_aabb=(lo, hi),
                                       crop_margin=crop_margin, k_plane=k_plane,
                                       log=(print if verbose else None), cancel_cb=cancel_cb)
    print(f"lidar index: {planes.pts.shape[0]:,} points (cropped to SfM volume)")
    stage(f"built lidar index ({planes.pts.shape[0]:,} cloud pts)")

    # foot-gun: the nearest-neighbour gate needs assoc > LiDAR point spacing, else every
    # valid point is rejected (nn ~ voxel). Warn if the user downsampled too aggressively.
    if voxel and voxel > 0.5 * max_assoc_dist:
        import warnings
        warnings.warn(f"voxel ({voxel}) is large vs max_assoc_dist ({max_assoc_dist}); the "
                      f"nearest-neighbour gate may reject most points. Raise max_assoc_dist "
                      f"or lower voxel.")

    # Scale sanity check: compare LiDAR density to SfM extent to catch unit mismatches
    # (e.g., LiDAR in feet vs SfM in meters)
    if verbose and voxel is None and planes.pts.shape[0] > 1000:
        extent_m = float(np.max(hi - lo))
        if extent_m > 0:
            density = planes.pts.shape[0] / (extent_m ** 3)
            if density < 1.0 or density > 1e6:
                print(f"[scale check] LiDAR density ({density:.1f} pts/m³) looks unusual; "
                      f"verify units match SfM (m) and LiDAR (m).")
            elif density > 100:
                print(f"[scale check] Dense LiDAR ({density:.1f} pts/m³); consider "
                      f"--voxel to speed up plane queries.")

    if qa_out:
        from . import qa
        os.makedirs(qa_out, exist_ok=True)
        d0, dmax = qa.residual_ply(rec, planes, os.path.join(qa_out, "residual_before.ply"))
        print(f"QA before: {qa.residual_stats(d0)}")
        stage("QA before written")

    # Snapshot the pre-aligned model so we can fall back if the refine diverges. A free-gauge
    # bundle adjustment over thousands of images can occasionally collapse the scale to a point
    # (cameras pile onto one spot) when the pre-align already satisfied the LiDAR ties; a rigid
    # pre-align cannot do that, so its result is the safe floor.
    import tempfile
    import shutil
    guarding = bool(outer_iters)
    pre_spread = _camera_spread(rec)
    print(f"camera spread (pre-aligned, before refine): {pre_spread:.2f} m")
    pre_med = _median_p2plane(rec, planes) if guarding else float("nan")
    snap = tempfile.mkdtemp(prefix="lidar_prealign_")
    try:
        if guarding and pre_spread > 1e-6:
            colmap_io.save(rec, snap)

        refine_reconstruction(rec, planes, w_lidar=w_lidar, huber=huber,
                              outer_iters=outer_iters, inner_iters=inner_iters,
                              max_assoc_dist=max_assoc_dist, planarity_min=planarity_min,
                              anneal=anneal, max_lidar_residuals=max_lidar_residuals,
                              fix_intrinsics=fix_intrinsics, cancel_cb=cancel_cb,
                              early_stop_tol=early_stop_tol, verbose=verbose, stage=stage)

        if guarding and pre_spread > 1e-6:
            post_spread = _camera_spread(rec)
            post_med = _median_p2plane(rec, planes)
            # Keep the refine only if the cameras stayed put (spread within 30% of the pre-align)
            # AND the fit didn't get worse. A free-gauge BA tends to shrink/drift the cameras even
            # when the points stay on the planes, so a meaningful drop in spread = reject. (The old
            # 0.25 threshold let a 4x collapse through right at the boundary.)
            collapsed = post_spread < 0.70 * pre_spread or post_spread > 1.50 * pre_spread
            worse = (np.isfinite(pre_med) and np.isfinite(post_med)
                     and post_med > pre_med * 1.10 + 1e-4)
            if collapsed or worse:
                why = (f"moved the cameras off the pre-align (spread {pre_spread:.2f} m -> "
                       f"{post_spread:.2f} m)" if collapsed else
                       f"made the fit worse (median {pre_med*100:.1f} cm -> {post_med*100:.1f} cm)")
                print(f"[WARNING] the refine {why} - reverting to the pre-aligned result (a clean "
                      f"rigid placement). Set 'Re-match rounds' = 0 to skip the refine next time.")
                rec = colmap_io.load(snap)
                stage("reverted to pre-align (refine did not help)")
            else:
                print(f"refine kept: median fit {pre_med*100:.1f} cm -> {post_med*100:.1f} cm")
    finally:
        shutil.rmtree(snap, ignore_errors=True)
    print(f"camera spread: {_camera_spread(rec):.1f} m "
          f"({'OK' if _camera_spread(rec) > 1.0 else 'COLLAPSED - check the cloud/ties'})")

    # Always save what we have (a stopped run still leaves a partially-aligned model), but skip
    # the slow QA plane-query and XMP export on cancel so Stop returns promptly.
    cancelled = cancel_cb is not None and cancel_cb()
    colmap_io.save(rec, sparse_out)
    print(f"wrote refined model -> {sparse_out}")
    stage("saved refined model")

    if qa_out and not cancelled:
        from . import qa
        d1, _ = qa.residual_ply(rec, planes, os.path.join(qa_out, "residual_after.ply"),
                                dmax=dmax)
        print(f"QA after:  {qa.residual_stats(d1)}")

    if xmp_out and not cancelled:
        from .export_xmp import export_xmp
        n = export_xmp(rec, xmp_out, pose_prior=xmp_pose_prior, axis_flip=xmp_axis_flip)
        print(f"exported {n} RealityScan .xmp pose sidecars -> {xmp_out}")
    if cancelled:
        print("[stopped] skipped QA/XMP export; saved the partial model only")
