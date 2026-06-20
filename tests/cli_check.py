"""Inspect the artifact the CLI wrote: load data/sparse_refined and compare to truth."""
import os, sys
import numpy as np
import pycolmap

root = sys.argv[1] if len(sys.argv) > 1 else "D:/lidar-align/.verify"
data = os.path.join(root, "data")
d = np.load(os.path.join(data, "truth.npz"))
ids = d["ids"]; truth = d["truth"]; before = float(d["before"])

rec = pycolmap.Reconstruction(os.path.join(data, "sparse_refined"))
after = float(np.mean([np.linalg.norm(rec.points3D[int(i)].xyz - truth[k])
                       for k, i in enumerate(ids)]))

print(f"BEFORE={before:.4f}  AFTER={after:.4f}  ratio={after/before:.3f}  "
      f"images={rec.num_reg_images()}  points={rec.num_points3D()}")
assert after < 0.25 * before, f"CLI refine did not reduce error (ratio {after/before:.3f})"
print("CLI E2E: PASS")
