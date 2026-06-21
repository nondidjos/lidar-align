#!/usr/bin/env python
"""Re-export RealityScan .xmp pose sidecars from an ALREADY-refined model.

The align is the slow part; the XMP axis convention is the fiddly part. RealityScan's camera
axes differ from COLMAP's, and the right flip can only be confirmed by importing into RS. This
lets you try flips in seconds against the saved `sparse_refined` model instead of re-running the
whole alignment for each one.

    # write XMPs with a specific flip
    python scripts/reexport_xmp.py data/sfm/sparse_refined data/images --axis-flip identity

    # write one image's XMP in ALL four flips, into out/<preset>/, to find the right one fast
    python scripts/reexport_xmp.py data/sfm/sparse_refined out --all --one IMG_0001.jpg

Presets: rc_default (flip Y,Z - the usual one) | identity | flip_xz | flip_xy
"""
import argparse
import os

from lidar_align import colmap_io
from lidar_align.export_xmp import export_xmp, AXIS_PRESETS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="refined model dir (the sparse_out from the align, holds points3D.bin)")
    ap.add_argument("out", help="output dir for the .xmp sidecars")
    ap.add_argument("--axis-flip", default="rc_default", choices=list(AXIS_PRESETS),
                    help="which COLMAP->RC axis flip to apply (default rc_default)")
    ap.add_argument("--pose-prior", default="locked", choices=["locked", "initial"],
                    help="how RealityScan treats the poses (default locked)")
    ap.add_argument("--no-intrinsics", action="store_true",
                    help="write pose only, omit the focal/principal-point calibration prior")
    ap.add_argument("--all", action="store_true",
                    help="write every axis-flip preset into out/<preset>/ for a quick A/B in RS")
    ap.add_argument("--one", help="with --all, only export this single image name (fast to test)")
    args = ap.parse_args()

    rec = colmap_io.load(args.model)
    presets = list(AXIS_PRESETS) if args.all else [args.axis_flip]
    for preset in presets:
        out = os.path.join(args.out, preset) if args.all else args.out
        n = export_xmp(rec, out, pose_prior=args.pose_prior, axis_flip=preset,
                       include_intrinsics=not args.no_intrinsics, only=args.one)
        print(f"wrote {n} .xmp [{preset}] -> {out}")
    if args.one and not any(im.name == args.one or os.path.basename(im.name) == args.one
                            for im in rec.images.values()):
        print(f"WARNING: image {args.one!r} not found in the model - nothing written")


if __name__ == "__main__":
    main()
