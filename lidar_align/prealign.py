"""Coarse Sim3 pre-alignment of an SfM reconstruction into the LiDAR frame.

SfM is solved only up to an arbitrary similarity (7 DOF: scale, rotation, translation).
The point-to-plane refiner needs the model already roughly in the LiDAR frame, or
nearest-plane association is meaningless. This estimates that Sim3 and applies it.

Modes:
  - correspondences: >=3 pairs (SfM xyz <-> LiDAR xyz) -> exact Umeyama similarity.
  - scaled ICP: refine an initial guess against the LiDAR cloud (Open3D, with_scaling),
    seeded from correspondences or a centroid/extent coarse init.
"""
from __future__ import annotations
import numpy as np


def umeyama_sim3(src, dst):
    """Least-squares similarity mapping src -> dst. Returns (scale, R(3,3), t(3))."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if len(src) < 3:
        raise ValueError("need >= 3 correspondences")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / len(src)
    U, Dvec, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    s = float((Dvec * np.diag(S)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def _decompose_sim(T):
    M = T[:3, :3]; t = T[:3, 3].copy()
    s = float(abs(np.linalg.det(M)) ** (1.0 / 3.0))
    R = M / s
    U, _, Vt = np.linalg.svd(R)         # re-orthonormalise
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return s, R, t


def scaled_icp(src_pts, dst_pts, init=None, voxel=0.2, max_corr=None, max_iter=80):
    """Scaled point-to-point ICP src -> dst. Returns (s, R, t, fitness, inlier_rmse)."""
    import open3d as o3d

    def mk(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.asarray(p, np.float64))
        return c.voxel_down_sample(voxel) if (voxel and voxel > 0) else c

    src, dst = mk(src_pts), mk(dst_pts)
    if init is None:
        sp, dp = np.asarray(src.points), np.asarray(dst.points)
        s0 = dp.std(0).mean() / max(sp.std(0).mean(), 1e-9)
        T0 = np.eye(4); T0[:3, :3] = np.eye(3) * s0
        T0[:3, 3] = dp.mean(0) - s0 * sp.mean(0)
    else:
        s, R, t = init
        T0 = np.eye(4); T0[:3, :3] = s * R; T0[:3, 3] = t
    if max_corr is None:
        max_corr = 5 * (voxel or 0.2)
    res = o3d.pipelines.registration.registration_icp(
        src, dst, max_corr, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    s, R, t = _decompose_sim(np.asarray(res.transformation))
    return s, R, t, float(res.fitness), float(res.inlier_rmse)


def global_register(src_pts, dst_pts, voxel=1.0, ransac_n=4, max_iter=400_000):
    """Rotation-agnostic coarse Sim3 via FPFH + RANSAC. SfM comes out in an arbitrary
    orientation; centroid/extent ICP only has a ~30 deg basin, so for a badly rotated
    frame this is the robust init. FPFH is scale-sensitive, so we pre-match scale from
    extents first. Returns (s, R, t, fitness). Needs scene structure to work - on a
    featureless cloud, fall back to manual `correspondences`."""
    import open3d as o3d
    src = np.asarray(src_pts, np.float64); dst = np.asarray(dst_pts, np.float64)
    s0 = dst.std(0).mean() / max(src.std(0).mean(), 1e-9)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src0 = (src - mu_s) * s0 + mu_d        # scale+translate to overlap dst (rotation unknown)

    def prep(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.ascontiguousarray(p))
        c = c.voxel_down_sample(voxel)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            c, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return c, f

    cs, fs = prep(src0); cd, fd = prep(dst)
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
    ]
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        cs, cd, fs, fd, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n, checkers,
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999))
    M0 = np.eye(4); M0[:3, :3] = s0 * np.eye(3); M0[:3, 3] = mu_d - s0 * mu_s
    s, R, t = _decompose_sim(np.asarray(res.transformation) @ M0)
    return s, R, t, float(res.fitness)


def apply_sim3(rec, s, R, t):
    """Apply a Sim3 (x' = s R x + t) to a pycolmap reconstruction, in place."""
    import pycolmap
    from scipy.spatial.transform import Rotation as Rot
    quat = Rot.from_matrix(np.asarray(R, np.float64)).as_quat()   # [x, y, z, w]
    sim = pycolmap.Sim3d(float(s), pycolmap.Rotation3d(quat), np.asarray(t, np.float64))
    rec.transform(sim)
    return sim


def prealign_reconstruction(rec, lidar_pts, correspondences=None, method="auto", voxel=0.3):
    """Bring `rec` into the LiDAR frame, then polish with scaled ICP. Init source:
      - correspondences given -> exact Umeyama (most reliable; survives any rotation)
      - method="global"       -> FPFH+RANSAC (rotation-agnostic; needs scene structure)
      - method="auto"         -> centroid/extent coarse init (only ~30 deg rotation basin)
    """
    import pycolmap  # noqa: F401  (ensures Sim3d available downstream)
    sfm_pts = np.array([rec.points3D[p].xyz for p in rec.points3D], np.float64)
    init = None
    if correspondences is not None and len(correspondences) > 0:
        corr = np.asarray(correspondences, np.float64)
        if corr.ndim == 3 and corr.shape[1:] == (2, 3):     # N pairs [[src],[dst]]
            src, dst = corr[:, 0, :], corr[:, 1, :]
        else:                                                # (src_array, dst_array)
            src = np.asarray(correspondences[0], np.float64)
            dst = np.asarray(correspondences[1], np.float64)
        init = umeyama_sim3(src, dst)
    elif method == "global":
        gs, gR, gt, gfit = global_register(sfm_pts, lidar_pts, voxel=voxel * 3)
        if gfit < 0.3:
            import warnings
            warnings.warn(f"global_register fitness low ({gfit:.2f}) - FPFH likely failed "
                          f"(featureless/sparse cloud). Provide `correspondences` instead.")
        init = (gs, gR, gt)
    s, R, t, fit, rmse = scaled_icp(sfm_pts, lidar_pts, init=init, voxel=voxel)
    apply_sim3(rec, s, R, t)
    return dict(scale=s, R=R, t=t, fitness=fit, inlier_rmse=rmse)
