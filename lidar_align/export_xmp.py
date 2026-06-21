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
   xcr:Coordinates="absolute"{calib}>
   <xcr:Rotation>{rot}</xcr:Rotation>
   <xcr:Position>{pos}</xcr:Position>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

# Calibration-prior attributes (RC convention: focal as 35mm-equiv, principal point offset from
# centre normalised by image WIDTH, aspect = fy/fx). Written as "initial" so RealityScan seeds
# from the SfM-solved intrinsics but can still refine them - important for video-extracted frames
# that have no EXIF focal for RS to fall back on. Distortion is left for RS to estimate (mapping
# COLMAP's distortion models to RC's is convention-fragile and can't be verified here).
_CALIB = ('\n   xcr:CalibrationPrior="{cprior}"'
          '\n   xcr:FocalLength35mm="{f35:.9g}"'
          '\n   xcr:PrincipalPointU="{ppu:.9g}"'
          '\n   xcr:PrincipalPointV="{ppv:.9g}"'
          '\n   xcr:AspectRatio="{ar:.9g}"'
          '\n   xcr:Skew="0"')


def _calib_fields(cam, cprior):
    """RC calibration-prior attribute string for a pycolmap Camera, or '' if intrinsics
    look degenerate (then the sidecar is pose-only and RS self-calibrates from EXIF)."""
    W, H = int(cam.width), int(cam.height)
    fx, fy = float(cam.focal_length_x), float(cam.focal_length_y)
    cx, cy = float(cam.principal_point_x), float(cam.principal_point_y)
    if W <= 0 or H <= 0 or fx <= 0 or fy <= 0:
        return ""
    return _CALIB.format(cprior=cprior, f35=fx * 36.0 / W, ppu=(cx - W / 2.0) / W,
                         ppv=(cy - H / 2.0) / W, ar=fy / fx)


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


def _outlier_camera_ids(rec, factor=30.0, floor=1000.0):
    """Image ids whose camera centre is a far-flung outlier - a pose that drifted in an
    underconstrained solve. Robust: distance from the median centre beyond `factor` x the median
    such distance (or `floor` metres, whichever is larger). Empty when the centres are coherent,
    so it's a no-op on a clean run; only the genuine flyers (km-scale here) get dropped. If the
    majority drifted the median itself is bad and this can't help - that needs a re-align.
    """
    items = [(iid, camera_pose(im)[1]) for iid, im in rec.images.items() if im.has_pose]
    if len(items) < 8:
        return set()
    C = np.array([c for _, c in items])
    d = np.linalg.norm(C - np.median(C, axis=0), axis=1)
    thresh = max(factor * float(np.median(d)), floor)
    return {iid for (iid, _), di in zip(items, d) if di > thresh}


def export_xmp(rec, out_dir, pose_prior="locked", axis_flip=None,
               include_intrinsics=True, calib_prior="initial", only=None,
               drop_outlier_cameras=True):
    """Write one <image_basename>.xmp per registered image. Returns the count.

    `include_intrinsics`: also write the SfM-solved focal/principal-point as an RS calibration
    prior (`calib_prior`, default "initial" = refinable). Strongly recommended for video frames
    with no EXIF focal, which RS would otherwise have nothing to seed from.
    `only`: if set, export just the image whose name (or basename) matches - handy for quickly
    A/B-ing axis flips on a single image without writing the whole set.
    `drop_outlier_cameras`: skip cameras whose pose drifted far from the rest (flung by an
    underconstrained solve) so they don't wreck RealityScan's scene extent. No-op on a clean run.
    """
    F = _resolve_flip(axis_flip)
    os.makedirs(out_dir, exist_ok=True)
    drop = _outlier_camera_ids(rec) if (drop_outlier_cameras and not only) else set()
    if drop:
        print(f"[xmp] skipping {len(drop)} far-outlier camera(s) with drifted poses; "
              f"exporting the well-placed ones")
    n = 0
    for iid, image in rec.images.items():
        if not image.has_pose or iid in drop:
            continue
        if only and image.name != only and os.path.basename(image.name) != only:
            continue
        R, C = camera_pose(image)
        R_rc = F @ R
        rot = " ".join(f"{v:.15g}" for v in R_rc.reshape(-1))
        pos = " ".join(f"{v:.15g}" for v in C)
        calib = ""
        if include_intrinsics and image.camera_id in rec.cameras:
            calib = _calib_fields(rec.cameras[image.camera_id], calib_prior)
        base = os.path.splitext(image.name)[0]      # may contain a subdir
        target = os.path.join(out_dir, base + ".xmp")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(_XMP.format(prior=pose_prior, rot=rot, pos=pos, calib=calib))
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
