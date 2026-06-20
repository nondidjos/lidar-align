#!/usr/bin/env python
"""Convert a Leica/RTC360-style .e57 (one or more scan setups) into a single LAS/LAZ in
GLOBAL coordinates, ready for the refinement pipeline's out-of-core LiDAR reader.

pye57 applies each scan's pose (transform=True), so the output is one merged cloud in the
project frame. Optional --voxel thins it: survey clouds are enormous, and 0.03 (3 cm) matches
the resample the design notes recommend (realityscan-lidar-alignment.md).

    python scripts/e57_to_laz.py site.e57 data/lidar.laz
    python scripts/e57_to_laz.py site.e57 data/lidar.laz --voxel 0.03

The pipeline also reads .e57 directly now (lidar_index), so this is only needed when you
want a reusable, pre-thinned LAZ instead of re-reading the raw e57 every run.

Needs: pip install pye57 laspy[lazrs]   (both in requirements.txt)
"""
import argparse
import numpy as np


def _voxel(pts, voxel):
    """Keep one point per voxel (first seen)."""
    if not voxel or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def convert(inp, out, voxel=None):
    import pye57
    import laspy
    e = pye57.E57(inp)
    n = e.scan_count
    print(f"{inp}: {n} scan(s)")

    kept = []
    for i in range(n):
        d = e.read_scan(i, transform=True, ignore_missing_fields=True)   # -> global frame
        try:
            xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
        except KeyError:
            raise SystemExit(
                f"scan {i}: no cartesian points (spherical-only e57 unsupported)")
        xyz = xyz.astype(np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if voxel:
            xyz = _voxel(xyz, voxel)
        kept.append(xyz)
        print(f"  scan {i}: {len(xyz):,} pts")

    pts = np.concatenate(kept) if kept else np.empty((0, 3))
    if voxel:
        pts = _voxel(pts, voxel)       # merge overlap across scans
    if len(pts) == 0:
        raise SystemExit("no points after read")

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]              # 1 mm, survey grade
    header.offsets = np.floor(pts.min(axis=0))         # keep scaled ints in range
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.write(out)                                     # .laz -> lazrs backend
    print(f"wrote {len(pts):,} pts -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .e57")
    ap.add_argument("output", help="output .las or .laz")
    ap.add_argument("--voxel", type=float, default=None,
                    help="downsample to one point per VOXEL metres (e.g. 0.03)")
    args = ap.parse_args()
    convert(args.input, args.output, voxel=args.voxel)


if __name__ == "__main__":
    main()
