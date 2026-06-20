"""Pre-alignment test: recover a known Sim3 that moved the model out of the LiDAR frame."""
import numpy as np
import pycolmap
from pycolmap import Rotation3d, Sim3d

from lidar_align.prealign import umeyama_sim3, apply_sim3, prealign_reconstruction

np.random.seed(1)

opts = pycolmap.SyntheticDatasetOptions()
opts.num_rigs = 1; opts.num_cameras_per_rig = 1
opts.num_frames_per_rig = 8; opts.num_points3D = 200; opts.track_length = 8
rec = pycolmap.synthesize_dataset(opts)

ids = list(rec.points3D.keys())
truth = {p: np.array(rec.points3D[p].xyz, float) for p in ids}
lidar = np.array([truth[p] for p in ids])           # LiDAR frame == truth

def err():
    return float(np.mean([np.linalg.norm(rec.points3D[p].xyz - truth[p]) for p in ids]))

# move the model out of the LiDAR frame with a strong known similarity
ang = 0.3
G = Sim3d(0.7, Rotation3d(np.array([0.0, np.sin(ang/2), 0.0, np.cos(ang/2)])),
          np.array([2.0, -1.0, 0.5]))
rec.transform(G)
e0 = err()
print(f"before pre-align: {e0:.4f}")

# (A) exact Umeyama from correspondences (current sfm xyz -> truth)
sfm_now = np.array([rec.points3D[p].xyz for p in ids])
s, R, t = umeyama_sim3(sfm_now, lidar)
apply_sim3(rec, s, R, t)
e1 = err()
print(f"after umeyama:    {e1:.6f}")
assert e1 < 1e-4, f"umeyama did not recover the similarity ({e1})"

# (B) full prealign_reconstruction (umeyama seed + scaled ICP polish) on a re-perturbed model
rec.transform(G)                                    # knock it back out of frame
sub = np.random.choice(len(ids), 25, replace=False)
src = np.array([rec.points3D[ids[i]].xyz for i in sub])
dst = lidar[sub]
info = prealign_reconstruction(rec, lidar, correspondences=(src, dst), voxel=0.0)
e2 = err()
print(f"after prealign:   {e2:.6f}  (fitness {info['fitness']:.3f})")
assert e2 < 1e-2, f"prealign_reconstruction failed ({e2})"
print("PREALIGN TEST: PASS")
