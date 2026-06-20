"""Unit test: local PCA plane fitting in LidarPlanes recovers a known plane."""
import numpy as np
from lidar_align.lidar_index import LidarPlanes, _load_points

np.random.seed(0)

# a tilted plane through the origin: normal n0, sample a dense grid on it
n0 = np.array([0.2, -0.3, 1.0]); n0 /= np.linalg.norm(n0)
# two in-plane basis vectors
a = np.cross(n0, [1, 0, 0]); a /= np.linalg.norm(a)
b = np.cross(n0, a)
u, v = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
pts = (u.ravel()[:, None] * a + v.ravel()[:, None] * b)
pts += 0.001 * np.random.randn(*pts.shape)        # tiny noise

planes = LidarPlanes(pts, k_plane=16)

# base point projected ONTO the plane (plane passes through origin), then lifted 0.25 m
base = np.array([0.1, -0.05, 0.0])
base -= (base @ n0) * n0                            # project onto plane
X = (base + 0.25 * n0)[None, :]
C, N, D, P, NN = planes.query(X)

normal_err = 1.0 - abs(float(N[0] @ n0))          # 0 == parallel
print(f"normal err={normal_err:.4f}  planarity={P[0]:.4f}  pt2plane={D[0]:.4f} (true 0.25)")

assert normal_err < 1e-3, "fitted normal off"
assert P[0] > 0.9, "planarity should be ~1 on a plane"
assert abs(D[0] - 0.25) < 5e-3, "point-to-plane distance wrong"
print("INDEX TEST: PASS")
