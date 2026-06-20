"""FPFH+RANSAC global pre-align recovers a large rotation that centroid-init ICP can't."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lidar_align.prealign import global_register, scaled_icp

np.random.seed(0)

# structured, asymmetric scene: three perpendicular walls of different sizes (a corner),
# so the alignment is unique and FPFH has features to latch onto.
def wall(n, u, v, fixed_axis):
    p = np.zeros((n, 3))
    a, b = [i for i in range(3) if i != fixed_axis]
    p[:, a] = np.random.uniform(*u, n)
    p[:, b] = np.random.uniform(*v, n)
    return p

P = np.vstack([
    wall(4000, (0, 6), (0, 3), 2),    # big
    wall(3000, (0, 6), (0, 2), 1),    # medium
    wall(1500, (0, 3), (0, 2), 0),    # small -> asymmetric
])
P += 0.005 * np.random.randn(*P.shape)

# known similarity with a LARGE rotation (~69 deg) + scale + translation
axis = np.array([0.2, 1.0, 0.3]); axis /= np.linalg.norm(axis)
R_true = Rot.from_rotvec(1.2 * axis).as_matrix()
s_true, t_true = 1.7, np.array([10.0, -5.0, 3.0])
dst = s_true * (P @ R_true.T) + t_true

# global init, then ICP polish (the real prealign path)
gs, gR, gt, fit = global_register(P, dst, voxel=0.3)
s, R, t, f2, rmse = scaled_icp(P, dst, init=(gs, gR, gt), voxel=0.1)

aligned = s * (P @ R.T) + t
err = float(np.sqrt(((aligned - dst) ** 2).sum(1)).mean())
print(f"global fitness={fit:.3f}  after-ICP fitness={f2:.3f}  mean_residual={err:.4f}  "
      f"(scene scale ~{dst.std():.1f})")
assert err < 0.05, f"global+ICP failed to recover the 69deg similarity (residual {err})"
print("GLOBAL REGISTRATION TEST: PASS")
