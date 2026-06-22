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


def _solver_threads():
    """All cores (capped) for the Ceres solve. Safe now the LiDAR cost is native (no GIL)."""
    return max(1, min(os.cpu_count() or 1, 32))


def _native_plane_cost(plane_pt, plane_n, weight, soft=100.0):
    """Point-to-plane as a NATIVE (C++) NormalPrior cost so the bundle adjust can MULTI-THREAD.

    Why not the Python PointToPlaneCost above: pyceres calls a Python cost's Evaluate() from every
    Ceres worker thread, so 2+ threads contend savagely on the GIL - measured 13 s at 1 thread vs
    >180 s at 2. That forced num_threads=1 and, single-threaded, the solve scales with point count,
    which is what forced the heavy model subsample (3.76M -> ~64k). A native cost has no GIL: it
    multi-threads cleanly AND is ~4x faster even on one thread - so we can keep far more points.

    A true point-to-plane residual w*n.(X-p) has a rank-1 (singular) information matrix, which
    NormalPrior can't take (it factorises the information, needing it invertible). So use an
    anisotropic Gaussian prior: full weight w along the normal n, a negligibly weak pull (w/soft)
    in-plane. The point is pinned in-plane by its reprojection observations, so that tiny in-plane
    term is dominated and the fit matches the exact point-to-plane cost - verified: a full
    reprojection+LiDAR synthetic recovers identically (camera/point error ratio 0.002 vs 0.003).

    Residual is the 3-vector sqrt(info).(X-p); its norm is ~|w*n.(X-p)| (in-plane part is ~soft x
    smaller), so HuberLoss(hub*w) still robustifies at `hub` metres of point-to-plane distance.
    """
    n = np.asarray(plane_n, dtype=np.float64)
    n = n / (np.linalg.norm(n) or 1.0)   # the projector identity below needs a UNIT normal; a
                                         # non-unit n makes nn a scaled projector and the closed-form
                                         # cov indefinite (silently wrong). Callers pass eigh normals
                                         # (already unit); this just keeps the helper safe to reuse.
    nn = np.outer(n, n)
    # weight<=0 means "no LiDAR pull". Clamp so the cov (which divides by w^2) can't ZeroDivision /
    # blow up - a near-zero weight then just makes the tie negligible, matching the old Python cost
    # which no-op'd at weight 0 (a user setting w_lidar=0 to disable LiDAR must not crash the refine).
    w2 = max(float(weight), 1e-12) ** 2
    s2 = soft * soft
    # NormalPrior wants the COVARIANCE. info = w2*nn + (w2/s2)*(I-nn); since nn and (I-nn) are
    # complementary projectors, its inverse is closed-form - no per-tie np.linalg.inv (which, at
    # hundreds of thousands of ties x many rounds, is real time): cov = (1/w2)*nn + (s2/w2)*(I-nn).
    cov = (s2 / w2) * np.eye(3) + ((1.0 - s2) / w2) * nn
    return pyceres.factors.NormalPrior(
        np.ascontiguousarray(np.asarray(plane_pt, np.float64).reshape(3, 1)),
        np.ascontiguousarray(cov))


def _anchor_frames(rec, config):
    """Pin two well-separated camera frames so the global 7-DOF gauge (scale/rotation/position)
    can't run away during the solve, while every other camera and point is still free to warp
    onto the LiDAR (this is what fixes drift). Anchoring is only right once a pre-align has put
    the model roughly in the LiDAR frame - the anchors then hold that frame. Returns #pinned.

    Use camera pinning, NOT config.fix_gauge(THREE_POINTS): the latter crashes Ceres
    (LossFunctionWrapper abort) with this pycolmap/pyceres. Pinning frame poses is the standard
    constant-parameter-block mechanism and is stable.
    """
    ims = [im for im in rec.images.values() if im.has_pose]
    if len(ims) < 2:
        return 0
    cen = np.array([(-np.asarray(im.cam_from_world().matrix(), float)[:3, :3].T
                     @ np.asarray(im.cam_from_world().matrix(), float)[:3, 3]) for im in ims])
    axis = int(np.argmax(np.ptp(cen, axis=0)))        # most-spread axis -> pick the extremes
    pinned = 0
    for k in (int(np.argmin(cen[:, axis])), int(np.argmax(cen[:, axis]))):
        try:
            config.set_constant_rig_from_world_pose(ims[k].frame_id)
            pinned += 1
        except Exception:
            pass
    return pinned


def _make_adjuster(rec, fix_intrinsics, inner_iters, anchor=False):
    """A pycolmap CeresBundleAdjuster over all registered images + points.

    `anchor`: pin two cameras to fix the gauge (use after a pre-align) so a free-gauge solve
    can't collapse the scale; otherwise leave the gauge UNSPECIFIED (the LiDAR ties are the only
    datum - needed when there's no pre-align, e.g. the synthetic tests)."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = not fix_intrinsics
    opts.refine_principal_point = not fix_intrinsics
    opts.refine_extra_params = not fix_intrinsics
    opts.refine_points3D = True
    opts.refine_rig_from_world = True
    opts.print_summary = False
    try:
        opts.ceres.solver_options.max_num_iterations = int(inner_iters)
        # The LiDAR cost is now a NATIVE NormalPrior (see _native_plane_cost), so the solve can use
        # every core - no Python/GIL in the hot path. (The old Python PointToPlaneCost forced
        # num_threads=1: 2+ threads contended on the GIL and ran ~10x SLOWER. Native both
        # multi-threads cleanly and is faster per thread, which is what lets us keep far more
        # points instead of collapsing the model to a tiny subset.)
        opts.ceres.solver_options.num_threads = _solver_threads()
    except Exception:
        pass  # fall back to pycolmap defaults if the options layout differs

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        config.add_image(image_id)
    # Points observed by the added images are optimised by default - we do NOT loop
    # add_variable_point over every point. That loop was a redundant 3.7M-iteration Python
    # call that held the GIL (freezing the UI) and added minutes to the build, with zero
    # effect on the result (verified: the synth case recovers identically without it).
    if anchor and _anchor_frames(rec, config) >= 2:
        pass            # gauge fixed by the two pinned cameras; the rest warps onto the LiDAR
    else:
        # No pre-align -> the LiDAR ties are the only datum; leave the gauge free so they can
        # set position/orientation/scale (and so a global similarity error is correctable).
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


def _trim_outliers(rec, margin=3.0):
    """Delete far-flung outlier points - junk SfM triangulations sitting kilometres from the
    scene. They blow up the model's apparent extent and wreck the pre-align's scale estimate
    (verified: 20 stray points pulled the recovered scale from a true 40 down to 0.28, i.e. the
    whole model came out ~140x too small - that's the 'cameras collapsed' symptom). Keep points
    within `margin` x the robust 0.5-99.5 percentile half-extent of the bulk. Returns #deleted.
    """
    ids = np.array(list(rec.points3D.keys()), dtype=np.int64)
    if len(ids) < 100:
        return 0
    X = np.array([rec.points3D[int(i)].xyz for i in ids], np.float64)
    lo = np.percentile(X, 0.5, axis=0)
    hi = np.percentile(X, 99.5, axis=0)
    c = (lo + hi) / 2.0
    half = np.maximum(hi - lo, 1e-9) / 2.0 * margin
    keep = np.all(np.abs(X - c) <= half, axis=1)
    for pid in ids[~keep].tolist():
        rec.delete_point3D(pid)
    return int((~keep).sum())


def _clean_model(rec, min_track=3, error_pct=99.5):
    """Drop unreliable SfM points before alignment, using COLMAP's own quality signals:
      - short tracks: a point seen by < min_track cameras is poorly triangulated (weak/no
        parallax) - this is where the wild far-outliers come from;
      - the worst reprojection-error tail (above the error_pct percentile).
    Then geometric far-outliers as a backstop. The pre-align scale and the QA are only as good as
    the points fed in, so this is the sanity gate on the COLMAP cloud. Returns #deleted.
    """
    ids = np.array(list(rec.points3D.keys()), dtype=np.int64)
    if len(ids) < 200:
        return _trim_outliers(rec)
    errs = np.array([rec.points3D[int(i)].error for i in ids], np.float64)
    tls = np.array([rec.points3D[int(i)].track.length() for i in ids])
    pos = errs > 0
    err_thr = np.percentile(errs[pos], error_pct) if pos.any() else np.inf
    bad = (tls < min_track) | (errs > err_thr)
    for pid in ids[bad].tolist():
        rec.delete_point3D(pid)
    return int(bad.sum()) + _trim_outliers(rec)


def refine_reconstruction(rec, planes, w_lidar=5.0, huber=0.1,
                          outer_iters=5, inner_iters=50, max_assoc_dist=0.5,
                          planarity_min=0.1, anneal=True, max_lidar_residuals=150_000,
                          fix_intrinsics=True, verbose=True, cancel_cb=None,
                          early_stop_tol=0.0, prev_err=None, stage=None, anchor=False):
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
    ba = _make_adjuster(rec, fix_intrinsics, inner_iters, anchor=anchor)
    problem = ba.problem
    lidar_blocks = []
    _stage("built bundle adjuster" + (" (gauge anchored on 2 cameras)" if anchor else ""))
    # Candidate pool for the per-round nearest-plane query. We query this set's CURRENT positions
    # every round (multithreaded KD-tree), gate it, then subsample to max_lidar_residuals ties. Keep
    # ~3x the tie target as headroom for the gating, but cap it ABSOLUTELY: with a big tie target the
    # old `8 * max_lidar_residuals` blew this up to millions of k-NN+PCA queries per round (minutes
    # that look like a freeze). 1.5M is plenty to draw a well-distributed tie set from.
    cand_cap = min(max(3 * max_lidar_residuals, 200_000), 1_500_000)
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
                _native_plane_cost(C[k], N[k], w), loss, [cand_pts[k].xyz])
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
           anneal=True, max_lidar_residuals=150_000, fix_intrinsics=True,
           prealign=False, prealign_voxel=0.5, prealign_method="auto",
           correspondences=None, crop_margin=2.0, planes=None,
           qa_out=None, xmp_out=None, xmp_pose_prior="locked", xmp_axis_flip=None,
           cancel_cb=None, early_stop_tol=0.0, verbose=True, ba_max_points=800_000):
    from .lidar_index import _load_points

    stage = _Stages(enabled=verbose)
    rec = colmap_io.load(sparse_in)
    print(f"loaded {rec.num_reg_images()} images, {rec.num_points3D()} points")
    print(f"camera spread (as loaded, SfM frame): {_camera_spread(rec):.2f}")
    # Clean the cloud BEFORE anything else: short-track / high-error / far-outlier points are junk
    # triangulations that wreck the pre-align scale (a handful can make the model ~100x too small).
    n_out = _clean_model(rec)
    if n_out:
        print(f"cleaned {n_out:,} low-quality/outlier points (short tracks, high error, far flyers) "
              f"-> {rec.num_points3D():,} kept for alignment")
    stage(f"loaded reconstruction ({rec.num_reg_images()} imgs, {rec.num_points3D():,} pts)")

    # Pre-align on the FULL cleaned cloud (more points = better FPFH coverage for scale recovery);
    # we only thin the model AFTER, for the bundle adjustment.
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

    # The LiDAR cost is now native (multi-threaded), so the model no longer has to be thinned to a
    # tiny subset. Solve time still scales with point count (residuals x iterations), but with the
    # whole solve now on every core (the i9 deploy box has 24 threads / 64 GB) we can keep a much
    # larger representative subset (default 800k, ~12x the old single-threaded budget) and still run
    # in <= the old wall-clock - constraining partial-overlap mismatch and drift far better. Lower
    # ba_max_points on a weak machine; None/0 keeps every point (slow/heavy at millions x thousands).
    if ba_max_points and rec.num_points3D() > ba_max_points:
        before = rec.num_points3D()
        kept = _subsample_model(rec, ba_max_points)
        print(f"reduced model {before:,} -> {kept:,} points for the bundle adjust (memory ceiling)")
        stage(f"reduced model to {kept:,} pts")

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

    def _run_pass(anchor_on):
        refine_reconstruction(rec, planes, w_lidar=w_lidar, huber=huber,
                              outer_iters=outer_iters, inner_iters=inner_iters,
                              max_assoc_dist=max_assoc_dist, planarity_min=planarity_min,
                              anneal=anneal, max_lidar_residuals=max_lidar_residuals,
                              fix_intrinsics=fix_intrinsics, cancel_cb=cancel_cb,
                              early_stop_tol=early_stop_tol, verbose=verbose, stage=stage,
                              anchor=anchor_on)

    def _bad():
        ps = _camera_spread(rec)
        pm = _median_p2plane(rec, planes)
        collapsed = ps < 0.70 * pre_spread or ps > 1.50 * pre_spread
        worse = np.isfinite(pre_med) and np.isfinite(pm) and pm > pre_med * 1.10 + 1e-4
        return collapsed, worse, ps, pm

    try:
        if guarding and pre_spread > 1e-6:
            colmap_io.save(rec, snap)

        # Pass 1: free gauge. Corrects a residual global similarity AND drift when it converges
        # (this is the best result, and what the synthetic tests exercise).
        _run_pass(False)

        if guarding and pre_spread > 1e-6:
            collapsed, worse, ps, pm = _bad()
            not_cancelled = cancel_cb is None or not cancel_cb()
            if (collapsed or worse) and prealign and not_cancelled:
                # The free-gauge pass diverged (typically a scale collapse at thousands of images).
                # The pre-align already fixed the global frame, so retry with the gauge ANCHORED on
                # two cameras: the LiDAR can then warp out the DRIFT without the scale running away.
                print(f"[refine] free-gauge pass diverged (spread {pre_spread:.2f}->{ps:.2f} m, "
                      f"median {pre_med*100:.1f}->{pm*100:.1f} cm) - retrying with the gauge "
                      f"anchored on 2 cameras…")
                rec = colmap_io.load(snap)
                stage("retry refine: gauge anchored on 2 cameras")
                _run_pass(True)
                collapsed, worse, ps, pm = _bad()
            if collapsed or worse:
                print(f"[WARNING] refine still off (spread {ps:.2f} m, median {pm*100:.1f} cm) - "
                      f"keeping the pre-aligned result (a clean rigid placement). Set 'Re-match "
                      f"rounds' = 0 to skip the refine next time.")
                rec = colmap_io.load(snap)
                stage("reverted to pre-align")
            else:
                print(f"refine kept: median fit {pre_med*100:.1f} cm -> {pm*100:.1f} cm, "
                      f"camera spread {ps:.1f} m")
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
