"""XMP pose export round-trips exactly; QA PLY is well-formed."""
import os, shutil, tempfile
import numpy as np
import pycolmap

from lidar_align.export_xmp import export_xmp, read_xmp_pose, camera_pose, _resolve_flip
from lidar_align.qa import write_ply, _colormap

np.random.seed(0)
opts = pycolmap.SyntheticDatasetOptions()
opts.num_rigs = 1; opts.num_cameras_per_rig = 1
opts.num_frames_per_rig = 6; opts.num_points3D = 100; opts.track_length = 6
rec = pycolmap.synthesize_dataset(opts)

scratch = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tmp")
os.makedirs(scratch, exist_ok=True)
out = tempfile.mkdtemp(dir=scratch)

n = export_xmp(rec, out, pose_prior="locked")
print(f"wrote {n} xmp sidecars")

# round-trip: read each xmp, undo the axis flip, compare R_cw and C to source
F = _resolve_flip("rc_default")          # diagonal +-1 -> its own inverse
maxerrR = maxerrC = 0.0
for image in rec.images.values():
    R, C = camera_pose(image)
    base = os.path.splitext(image.name)[0]
    R_rc, C2 = read_xmp_pose(os.path.join(out, base + ".xmp"))
    R_back = F @ R_rc
    maxerrR = max(maxerrR, float(np.abs(R_back - R).max()))
    maxerrC = max(maxerrC, float(np.abs(C2 - C).max()))
print(f"roundtrip maxerr  R={maxerrR:.2e}  C={maxerrC:.2e}")
assert maxerrR < 1e-6 and maxerrC < 1e-6, "xmp pose round-trip mismatch"

# the written rotation must be a valid rotation (orthonormal, det +1)
first = list(rec.images.values())[0]
R_rc, _ = read_xmp_pose(os.path.join(out, os.path.splitext(first.name)[0] + ".xmp"))
assert abs(np.linalg.det(R_rc) - 1) < 1e-6
assert np.allclose(R_rc @ R_rc.T, np.eye(3), atol=1e-6)
print("xmp rotation orthonormal, det=+1")

# QA PLY: header + binary body well-formed
pts = np.random.randn(500, 3); dist = np.random.rand(500)
ply = os.path.join(out, "qa.ply")
write_ply(ply, pts, _colormap(dist))
raw = open(ply, "rb").read()
assert b"element vertex 500" in raw and b"property uchar red" in raw
assert len(raw) == raw.index(b"end_header\n") + len(b"end_header\n") + 500 * 15
print("QA PLY well-formed")

shutil.rmtree(out, ignore_errors=True)
print("EXPORT/QA TEST: PASS")
