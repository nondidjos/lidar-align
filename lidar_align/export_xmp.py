"""Export refined COLMAP/GLOMAP camera poses as RealityScan (.xmp) sidecars.

This closes the loop: take the LiDAR-corrected poses and re-import them into RealityScan
(locked pose priors) so RealityScan does the dense mesh + texture with the correct
alignment.

RealityCapture / RealityScan `xcr` convention (matches RS's own native XMP):
  xcr:Position  = camera centre C in world coordinates (child element)
  xcr:Rotation  = 3x3 world->camera rotation, row-major (9 values), child element, RC axes
  xcr:PosePrior = "locked" (fixed) | "initial" (adjustable)   - rdf:Description attribute
  xcr:Coordinates = "absolute"                                - rdf:Description attribute

RC camera axes differ from COLMAP's computer-vision axes (COLMAP: +x right, +y down,
+z forward). The common mapping flips Y and Z:  R_rc = F @ R_cw,  F = diag(1,-1,-1).

! The axis flip is the widely-used convention but cannot be verified here without
RealityScan. Import ONE image first; if it looks mirrored/upside-down, pass a different
`axis_flip` preset (see AXIS_PRESETS). The pose *encoding* round-trips exactly
(tests/test_export.py); only the RC axis convention is the unverified bit.
"""
from __future__ import annotations
import os
import numpy as np

AXIS_PRESETS = {
    "rc_default": np.diag([1.0, -1.0, -1.0]),   # flip Y,Z  (COLMAP CV -> RC)
    "identity":   np.diag([1.0, 1.0, 1.0]),
    "flip_xz":    np.diag([-1.0, 1.0, -1.0]),
    "flip_xy":    np.diag([-1.0, -1.0, 1.0]),
}

_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
   xcr:Version="3"
   xcr:PosePrior="{prior}"
   xcr:Coordinates="absolute">
   <xcr:Rotation>{rot}</xcr:Rotation>
   <xcr:Position>{pos}</xcr:Position>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def camera_pose(image):
    """(R_cw 3x3 world->camera, C camera centre in world) for a pycolmap Image."""
    M = np.asarray(image.cam_from_world().matrix(), float)  # 3x4 [R|t], world->cam
    R, t = M[:3, :3], M[:3, 3]
    C = -R.T @ t
    return R, C


def _resolve_flip(axis_flip):
    if axis_flip is None or axis_flip == "":
        return AXIS_PRESETS["rc_default"]
    if isinstance(axis_flip, str):
        if axis_flip not in AXIS_PRESETS:
            raise ValueError(f"unknown xmp_axis_flip {axis_flip!r}; "
                             f"choose from {list(AXIS_PRESETS)}")
        return AXIS_PRESETS[axis_flip]
    return np.asarray(axis_flip, float)


def export_xmp(rec, out_dir, pose_prior="locked", axis_flip=None):
    """Write one <image_basename>.xmp per registered image. Returns the count."""
    F = _resolve_flip(axis_flip)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for image in rec.images.values():
        if not image.has_pose:
            continue
        R, C = camera_pose(image)
        R_rc = F @ R
        rot = " ".join(f"{v:.15g}" for v in R_rc.reshape(-1))
        pos = " ".join(f"{v:.15g}" for v in C)
        base = os.path.splitext(image.name)[0]      # may contain a subdir
        target = os.path.join(out_dir, base + ".xmp")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(_XMP.format(prior=pose_prior, rot=rot, pos=pos))
        n += 1
    return n


def read_xmp_pose(path):
    """Parse an .xmp back to (R_rc 3x3, C). For round-trip verification."""
    import re
    s = open(path).read()
    rot = re.search(r"<xcr:Rotation>([^<]+)</xcr:Rotation>", s).group(1)
    pos = re.search(r"<xcr:Position>([^<]+)</xcr:Position>", s).group(1)
    R_rc = np.array([float(v) for v in rot.split()], float).reshape(3, 3)
    C = np.array([float(v) for v in pos.split()], float)
    return R_rc, C
