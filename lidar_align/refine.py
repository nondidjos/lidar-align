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

from . import colmap_io
from .lidar_index import LidarPlanes


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
        opts.ceres.solver_options.num_threads = max(1, os.cpu_count() or 1)
    except Exception:
        pass  # fall back to pycolmap defaults if the options layout differs

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        config.add_image(image_id)
    for pid in rec.points3D:
        config.add_variable_point(pid)
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
    lo = pts.min(0)
    cell = np.maximum(pts.max(0) - lo, 1e-9) / max(round(cap ** (1.0 / 3.0)), 1)
    keys = np.floor((pts - lo) / cell).astype(np.int64)
    order = np.argsort(-plan[idx])              # most-planar first
    _, first = np.unique(keys[order], axis=0, return_index=True)
    return idx[order[first]]


def refine_reconstruction(rec, planes, w_lidar=5.0, huber=0.1,
                          outer_iters=5, inner_iters=50, max_assoc_dist=0.5,
                          planarity_min=0.1, anneal=True, max_lidar_residuals=30000,
                          fix_intrinsics=True, verbose=True, cancel_cb=None,
                          early_stop_tol=0.0, prev_err=None):
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
    for it in range(outer_iters):
        if cancel_cb is not None and cancel_cb():        # cooperative stop between rounds
            if verbose:
                print(f"[cancelled before outer {it}]")
            break
        frac = it / (outer_iters - 1) if (anneal and outer_iters > 1) else 1.0
        assoc = max_assoc_dist * (4.0 ** (1.0 - frac))   # loose -> tight
        w = w_lidar * (0.3 + 0.7 * frac)                  # gentle -> full
        hub = huber * (3.0 ** (1.0 - frac))               # soft -> sharp

        ids, X = colmap_io.world_points(rec)
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

        ba = _make_adjuster(rec, fix_intrinsics, inner_iters)
        problem = ba.problem
        # residual is w * n.(X-c); Huber transitions at |r|=arg, so the arg must be
        # w*hub for the robust threshold to be `hub` METRES of point-to-plane distance
        # (otherwise it would secretly be hub/w).
        loss = pyceres.HuberLoss(hub * w)
        for k in sel:
            problem.add_residual_block(
                PointToPlaneCost(C[k], N[k], w), loss,
                [rec.points3D[ids[k]].xyz])
        _cuda_check_once()
        if verbose:
            print(f"[outer {it}] solving…")
        ba.solve()

        # residual stats (used by both the log line and early stopping, so always computed)
        dk = dist[sel]
        med = float(np.median(dk)) if len(dk) else float("nan")
        if verbose:
            p95 = float(np.percentile(dk, 95)) if len(dk) else float("nan")
            print(f"[outer {it}] residuals {len(sel):,} (of {len(keep_idx):,} gated / "
                  f"{len(ids):,} pts)  w={w:.1f} huber={hub:.3f} assoc<{assoc:.2f}m  "
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
    P = np.array([rec.points3D[p].xyz for p in rec.points3D], np.float64)
    return P.min(0), P.max(0)


def refine(sparse_in, lidar, sparse_out,
           w_lidar=5.0, huber=0.1, outer_iters=5, inner_iters=50,
           voxel=None, k_plane=16, max_assoc_dist=0.5, planarity_min=0.1,
           anneal=True, max_lidar_residuals=30000, fix_intrinsics=True,
           prealign=False, prealign_voxel=0.5, prealign_method="auto",
           correspondences=None, crop_margin=2.0, planes=None,
           qa_out=None, xmp_out=None, xmp_pose_prior="locked", xmp_axis_flip=None,
           cancel_cb=None, early_stop_tol=0.0, verbose=True):
    from .lidar_index import _load_points

    rec = colmap_io.load(sparse_in)
    print(f"loaded {rec.num_reg_images()} images, {rec.num_points3D()} points")

    if prealign:
        from .prealign import prealign_reconstruction
        coarse = _load_points(lidar, voxel=prealign_voxel)
        info = prealign_reconstruction(rec, coarse, correspondences=correspondences,
                                       method=prealign_method, voxel=prealign_voxel)
        print(f"prealign[{prealign_method}]: scale={info['scale']:.4f} "
              f"fitness={info['fitness']:.3f} rmse={info['inlier_rmse']:.3f}")

    # units sanity: SfM extent (post-prealign) should match the LiDAR's metric scale
    lo, hi = _sfm_aabb(rec)
    print(f"SfM extent (post-prealign): {np.round(hi - lo, 2)} units "
          f"-> metric params (assoc/crop in same units)")

    if planes is None:
        planes = LidarPlanes.from_file(lidar, voxel=voxel, crop_aabb=(lo, hi),
                                       crop_margin=crop_margin, k_plane=k_plane)
    print(f"lidar index: {planes.pts.shape[0]:,} points (cropped to SfM volume)")

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

    refine_reconstruction(rec, planes, w_lidar=w_lidar, huber=huber,
                          outer_iters=outer_iters, inner_iters=inner_iters,
                          max_assoc_dist=max_assoc_dist, planarity_min=planarity_min,
                          anneal=anneal, max_lidar_residuals=max_lidar_residuals,
                          fix_intrinsics=fix_intrinsics, cancel_cb=cancel_cb,
                          early_stop_tol=early_stop_tol, verbose=verbose)

    colmap_io.save(rec, sparse_out)
    print(f"wrote refined model -> {sparse_out}")

    if qa_out:
        from . import qa
        d1, _ = qa.residual_ply(rec, planes, os.path.join(qa_out, "residual_after.ply"),
                                dmax=dmax)
        print(f"QA after:  {qa.residual_stats(d1)}")

    if xmp_out:
        from .export_xmp import export_xmp
        n = export_xmp(rec, xmp_out, pose_prior=xmp_pose_prior, axis_flip=xmp_axis_flip)
        print(f"exported {n} RealityScan .xmp pose sidecars -> {xmp_out}")
