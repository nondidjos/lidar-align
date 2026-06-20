"""Out-of-core path: stream a LAS, crop to the working volume while reading, fit planes.

Proves the mechanism that lets a 290 GB scan be used without loading it whole: only points
inside the SfM bounding box (+margin) survive the chunked read.
"""
import os
import tempfile
import numpy as np
import laspy

from lidar_align.lidar_index import LidarPlanes, _load_points

np.random.seed(0)

# big cloud spanning a 200 m box; the "working volume" is a 2 m cube near the origin
N = 2_000_000
pts = (np.random.rand(N, 3) - 0.5) * 200.0
# plant a dense planar patch inside the working volume so plane fitting has something real
plane = np.zeros((4000, 3))
plane[:, 0] = np.random.uniform(-1, 1, 4000)
plane[:, 1] = np.random.uniform(-1, 1, 4000)
plane[:, 2] = 0.5 + 0.002 * np.random.randn(4000)     # flat at z=0.5
pts = np.vstack([pts, plane])

# scratch on the same drive as the project (C: is typically tiny/full on this box)
scratch = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tmp")
os.makedirs(scratch, exist_ok=True)
tmp = tempfile.mkdtemp(dir=scratch)
las_path = os.path.join(tmp, "big.las")
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.offsets = pts.min(0)
hdr.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(hdr)
las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
las.write(las_path)
print(f"wrote {len(pts):,}-pt LAS spanning {pts.min(0).round(1)}..{pts.max(0).round(1)}")

# stream + crop to a 2 m working cube (no full load)
aabb = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
kept = _load_points(las_path, crop_aabb=aabb, crop_margin=0.0)
print(f"after streaming crop: {len(kept):,} points")

assert len(kept) < N * 0.01, "crop did not reduce the cloud"
assert np.all((kept >= -1.0) & (kept <= 1.0)), "points outside the crop survived"

# plane fitting works on the cropped cloud
planes = LidarPlanes.from_file(las_path, crop_aabb=aabb, crop_margin=0.0, k_plane=16)
C, Nz, D, P, NN = planes.query(np.array([[0.0, 0.0, 0.8]]))  # 0.3 m above the z=0.5 plane
print(f"plane probe: normal_z={abs(Nz[0,2]):.3f} planarity={P[0]:.3f} pt2plane={D[0]:.3f}")
assert abs(Nz[0, 2]) > 0.95 and P[0] > 0.8 and abs(D[0] - 0.3) < 0.02

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("OUT-OF-CORE TEST: PASS")
