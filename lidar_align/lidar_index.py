"""LiDAR reference index with local plane fitting.

Two things the naive version got wrong and this one does properly:

1. Local PCA planes, not single-nearest-point normals. For each query we fit a plane to
   the k nearest cloud points (centroid + smallest-eigenvector normal) and report a
   planarity score so non-planar neighbourhoods (edges, vegetation, clutter) can be
   rejected before they corrupt the bundle adjustment.

2. True out-of-core ingest for the fat cloud. `.las/.laz` are streamed in chunks and
   cropped to the working volume (+margin) *while reading*, so a 290 GB scan never has to
   fit in RAM - only the LiDAR near your photos survives. `.ply/.pcd` load in full (fine
   after a decimated export); for a literal raw PLY, convert to LAS or pre-decimate.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import cKDTree


def _voxel(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point per voxel (first seen)."""
    if not voxel or voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(0)                                    # non-negative
    span = [int(v) + 1 for v in keys.max(0)]              # per-axis extent (Python ints)
    # Dedup on a single packed key (one sort) instead of np.unique(axis=0) (row lexsort) when
    # the mixed-radix code fits in int64; identical first-seen result, much faster at scale.
    if span[0] * span[1] * span[2] < (1 << 62):
        packed = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, idx = np.unique(packed, return_index=True)
    else:
        _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def _load_points(path, voxel=None, crop_aabb=None, crop_margin=0.0, max_points=None):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    lo = hi = None
    if crop_aabb is not None:
        lo = np.asarray(crop_aabb[0], np.float64) - crop_margin
        hi = np.asarray(crop_aabb[1], np.float64) + crop_margin

    if ext in (".las", ".laz"):
        import laspy
        kept = []
        with laspy.open(path) as f:
            for chunk in f.chunk_iterator(5_000_000):
                xyz = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
                if lo is not None:
                    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
                    xyz = xyz[m]
                if voxel and len(xyz):
                    xyz = _voxel(xyz, voxel)
                if len(xyz):
                    kept.append(xyz)
        pts = np.concatenate(kept) if kept else np.empty((0, 3))
        pts = _voxel(pts, voxel)  # merge per-chunk voxel overlaps
    elif ext == ".e57":
        # e57 holds one or more scans in scanner-local frames plus a per-scan pose; pye57's
        # transform=True applies that pose so we get global coordinates. Read one scan at a
        # time and crop to the working volume before accumulating - same out-of-core spirit
        # as the LAS path, so only the LiDAR near your photos is held in memory.
        import pye57
        e = pye57.E57(path)
        kept = []
        for i in range(e.scan_count):
            d = e.read_scan(i, transform=True, ignore_missing_fields=True)
            try:
                xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
            except KeyError:
                raise ValueError(
                    f"e57 scan {i} has no cartesian points (spherical-only e57 is "
                    f"unsupported; re-export cartesian or convert with scripts/e57_to_laz.py)")
            xyz = xyz.astype(np.float64)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]      # drop invalid/empty returns
            if lo is not None:
                m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
                xyz = xyz[m]
            if voxel and len(xyz):
                xyz = _voxel(xyz, voxel)
            if len(xyz):
                kept.append(xyz)
        pts = np.concatenate(kept) if kept else np.empty((0, 3))
        pts = _voxel(pts, voxel)                          # merge per-scan voxel overlaps
    else:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        pts = np.asarray(pc.points, np.float64)
        if lo is not None:
            m = np.all((pts >= lo) & (pts <= hi), axis=1)
            pts = pts[m]
        pts = _voxel(pts, voxel)

    if max_points and len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    return np.ascontiguousarray(pts, np.float64)


def _fit_planes(neigh: np.ndarray):
    """neigh: (M, k, 3) -> centroid (M,3), unit normal (M,3), planarity (M,) in [0,1]."""
    c = neigh.mean(axis=1)
    q = neigh - c[:, None, :]
    cov = np.einsum("mki,mkj->mij", q, q) / neigh.shape[1]
    w, v = np.linalg.eigh(cov)                 # ascending eigenvalues l0<=l1<=l2
    normals = v[:, :, 0]                        # smallest eigenvector
    l0, l1, l2 = w[:, 0], w[:, 1], w[:, 2]
    # Flatness (off-plane vs smaller in-plane eigenvalue): 1 = flat, independent of how
    # isotropic the neighbourhood is. But flatness alone calls a *line* flat (l0~l1~0),
    # so also require the neighbourhood to be genuinely 2D (l1 not tiny vs l2); otherwise
    # it's a line/degenerate edge and planarity is forced to 0.
    flat = 1.0 - l0 / np.maximum(l1, 1e-12)
    is_2d = l1 > 0.04 * np.maximum(l2, 1e-12)
    planarity = np.where(is_2d, flat, 0.0)
    return c, normals, planarity


class LidarPlanes:
    def __init__(self, points: np.ndarray, k_plane: int = 16):
        self.pts = np.ascontiguousarray(points, np.float64)
        if len(self.pts) < k_plane:
            raise ValueError(f"cloud has {len(self.pts)} points, need >= k_plane={k_plane}")
        self.tree = cKDTree(self.pts)
        self.k_plane = int(k_plane)

    @classmethod
    def from_file(cls, path, voxel=None, crop_aabb=None, crop_margin=2.0,
                  k_plane=16, max_points=None):
        pts = _load_points(path, voxel=voxel, crop_aabb=crop_aabb,
                            crop_margin=crop_margin, max_points=max_points)
        return cls(pts, k_plane=k_plane)

    def query(self, X: np.ndarray, k_plane=None, batch=200_000):
        """X[M,3] -> (centroid[M,3], normal[M,3], pt2plane_dist[M], planarity[M], nn[M]).

        The plane is the local PCA fit at each query's k nearest cloud points; the
        residual the refiner builds is `n . (X - centroid)`. `nn` is the distance to the
        single nearest cloud point - used to reject edge-bleed (a point coplanar with, but
        laterally beyond, a real surface has small pt2plane but large nn).
        """
        X = np.ascontiguousarray(X, np.float64)
        k = int(k_plane or self.k_plane)
        M = len(X)
        C = np.empty((M, 3)); N = np.empty((M, 3))
        D = np.empty(M); P = np.empty(M); NN = np.empty(M)
        for s in range(0, M, batch):
            e = min(s + batch, M)
            d, idx = self.tree.query(X[s:e], k=k, workers=-1)
            NN[s:e] = d[:, 0]                           # nearest-neighbour distance
            neigh = self.pts[idx]                       # (b, k, 3)
            c, n, pl = _fit_planes(neigh)
            C[s:e], N[s:e], P[s:e] = c, n, pl
            D[s:e] = np.abs(np.einsum("mi,mi->m", n, X[s:e] - c))
        return C, N, D, P, NN
