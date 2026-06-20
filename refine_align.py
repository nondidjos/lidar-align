#!/usr/bin/env python
"""CLI entry: LiDAR-constrained refinement of a COLMAP/GLOMAP reconstruction.

    python refine_align.py --config config.yaml
    python refine_align.py --sparse-in data/sfm/sparse/0 --lidar data/lidar.laz \
                           --sparse-out data/sfm/sparse_refined --prealign --w-lidar 8
"""
import argparse
import yaml

from lidar_align import refine


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="YAML config (CLI flags override its values)")
    ap.add_argument("--sparse-in")
    ap.add_argument("--lidar")
    ap.add_argument("--sparse-out")
    ap.add_argument("--w-lidar", type=float)
    ap.add_argument("--huber", type=float)
    ap.add_argument("--outer-iters", type=int)
    ap.add_argument("--inner-iters", type=int)
    ap.add_argument("--voxel", type=float)
    ap.add_argument("--k-plane", type=int)
    ap.add_argument("--crop-margin", type=float)
    ap.add_argument("--max-assoc-dist", type=float)
    ap.add_argument("--planarity-min", type=float)
    ap.add_argument("--max-lidar-residuals", type=int,
                    help="cap LiDAR ties per round (spatially even) for speed at scale")
    ap.add_argument("--early-stop-tol", type=float, default=0.0,
                    help="stop if median pt2plane improvement < this (metres) between rounds")
    ap.add_argument("--prealign", dest="prealign", action="store_true", default=None,
                    help="run pre-alignment into the LiDAR frame first")
    ap.add_argument("--prealign-method", choices=["auto", "global"])
    ap.add_argument("--prealign-voxel", type=float)
    ap.add_argument("--qa-out", help="dir for residual_before/after.ply")
    ap.add_argument("--xmp-out", help="dir for RealityScan .xmp pose sidecars")
    ap.add_argument("--no-anneal", dest="anneal", action="store_false", default=None,
                    help="disable the loose->tight annealing schedule")
    ap.add_argument("--free-intrinsics", dest="fix_intrinsics", action="store_false",
                    default=None, help="let the solver adjust camera calibration too")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    def pick(cli, key, default=None):
        return cli if cli is not None else cfg.get(key, default)

    refine(
        sparse_in=pick(args.sparse_in, "sparse_in"),
        lidar=pick(args.lidar, "lidar"),
        sparse_out=pick(args.sparse_out, "sparse_out"),
        w_lidar=pick(args.w_lidar, "w_lidar", 5.0),
        huber=pick(args.huber, "huber", 0.1),
        outer_iters=pick(args.outer_iters, "outer_iters", 8),
        inner_iters=pick(args.inner_iters, "inner_iters", 50),
        voxel=pick(args.voxel, "voxel", None),
        k_plane=pick(args.k_plane, "k_plane", 16),
        crop_margin=pick(args.crop_margin, "crop_margin", 2.0),
        max_assoc_dist=pick(args.max_assoc_dist, "max_assoc_dist", 0.5),
        planarity_min=pick(args.planarity_min, "planarity_min", 0.1),
        max_lidar_residuals=pick(args.max_lidar_residuals, "max_lidar_residuals", 30000),
        anneal=pick(args.anneal, "anneal", True),
        prealign=pick(args.prealign, "prealign", False),
        prealign_method=pick(args.prealign_method, "prealign_method", "auto"),
        prealign_voxel=pick(args.prealign_voxel, "prealign_voxel", 0.5),
        correspondences=cfg.get("correspondences"),
        qa_out=pick(args.qa_out, "qa_out", None),
        xmp_out=pick(args.xmp_out, "xmp_out", None),
        xmp_pose_prior=cfg.get("xmp_pose_prior", "locked"),
        xmp_axis_flip=cfg.get("xmp_axis_flip"),
        fix_intrinsics=pick(args.fix_intrinsics, "fix_intrinsics", True),
        early_stop_tol=pick(args.early_stop_tol, "early_stop_tol", 0.0),
    )


if __name__ == "__main__":
    main()
