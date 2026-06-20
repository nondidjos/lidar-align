"""Write an on-disk fixture for a real CLI run of refine_align.py:
  data/sparse_in/      perturbed COLMAP model (global similarity applied)
  data/lidar.las       planar-patch LiDAR reference (streamed by the CLI)
  data/truth.npz       ground-truth point positions + 'before' error
  <root>/config.yaml   config the CLI reads
"""
import os, sys
import numpy as np
import yaml
import laspy
import pycolmap
from pycolmap import Rotation3d, Sim3d

root = sys.argv[1] if len(sys.argv) > 1 else "D:/lidar-align/.verify"
data = os.path.join(root, "data")
os.makedirs(data, exist_ok=True)
np.random.seed(3)

opts = pycolmap.SyntheticDatasetOptions()
opts.num_rigs = 1; opts.num_cameras_per_rig = 1
opts.num_frames_per_rig = 10; opts.num_points3D = 300; opts.track_length = 10
rec = pycolmap.synthesize_dataset(opts)

ids = list(rec.points3D.keys())
truth = np.array([rec.points3D[p].xyz for p in ids], float)

# planar-patch LiDAR (centroid of each patch == a true point)
step = 0.01
gu, gv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
grid = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)
patches = []
for Xi in truth:
    n = np.random.randn(3); n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    patches.append(Xi + (grid[:, 0:1] * a + grid[:, 1:2] * b) * step)
lidar = np.vstack(patches)

las_path = os.path.join(data, "lidar.las")
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.offsets = lidar.min(0); hdr.scales = [0.0005, 0.0005, 0.0005]
las = laspy.LasData(hdr)
las.x, las.y, las.z = lidar[:, 0], lidar[:, 1], lidar[:, 2]
las.write(las_path)

# perturb out of the LiDAR frame with a global similarity, write as sparse_in
ang = 0.02
rec.transform(Sim3d(1.03, Rotation3d(np.array([0.0, 0.0, np.sin(ang/2), np.cos(ang/2)])),
                    np.array([0.05, -0.04, 0.06])))
sp_in = os.path.join(data, "sparse_in"); os.makedirs(sp_in, exist_ok=True)
rec.write(sp_in)

before = float(np.mean(np.linalg.norm(
    np.array([rec.points3D[p].xyz for p in ids]) - truth, axis=1)))
np.savez(os.path.join(data, "truth.npz"), ids=np.array(ids), truth=truth, before=before)

cfg = dict(
    sparse_in=sp_in.replace("\\", "/"),
    sparse_out=os.path.join(data, "sparse_refined").replace("\\", "/"),
    lidar=las_path.replace("\\", "/"),
    prealign=True, prealign_voxel=0.1,
    voxel=None, crop_margin=2.0, k_plane=16,
    w_lidar=20.0, huber=0.2, outer_iters=10, inner_iters=30,
    max_assoc_dist=0.5, planarity_min=0.05, anneal=True, fix_intrinsics=True,
)
with open(os.path.join(root, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)

print(f"FIXTURE READY  before_error={before:.4f}  lidar_pts={len(lidar)}")
