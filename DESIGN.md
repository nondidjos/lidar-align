# Design & goals

## Goal

Take a photogrammetry sparse model (cameras + 3D points, up-to-scale and floating in an
arbitrary frame) and a survey-grade LiDAR scan, and bend the model onto the scan so the LiDAR
becomes the metric datum. Output corrected camera poses as RealityScan `.xmp` sidecars so the
mesh/texture step (in RealityScan) runs on a correctly scaled, placed, drift-free model.

Target accuracy: centimetre-level point-to-plane fit to the scan over the whole scene.

## Scope

**In:** pre-alignment, LiDAR point-to-plane refinement, QA, RealityScan XMP export, and SfM
orchestration (COLMAP/GLOMAP/hloc/GLUEMAP) as a convenience.

**Out:** meshing, texturing, dense MVS — those stay in RealityScan/COLMAP. lidar-align only
fixes pose/scale/drift.

## Target environment

- **Deploy box:** Windows, Intel i9-12900K (24 threads), RTX 3080, 64 GB RAM. Defaults are
  sized for this (e.g. `ba_max_points=800k`, all-core solve). Lower them on a weak machine.
- **Subjects:** architectural / survey scenes — often repetitive (stairs, railings, façades),
  sometimes shot on wide/fisheye action cams, against dense terrestrial scans (Leica etc.).
- **SfM engines:** COLMAP (SIFT, incremental or GLOMAP global), hloc (SuperPoint+LightGlue,
  native Windows GPU), GLUEMAP (learned, Linux/CUDA via WSL).

## Pipeline

1. **SfM** — photos → COLMAP sparse model. The hard part for repetitive/fisheye scenes;
   choice of engine/mapper matters most here.
2. **Pre-align** — coarse Sim3 onto the scan: manual correspondences or the visual slider tool
   (reliable), else auto FPFH (fails on repetitive structure + partial overlap).
3. **Refine** — point-to-plane bundle adjustment: reuse pycolmap's Ceres problem, add
   `w·n·(X−p)` residuals against local cloud planes under a Huber loss, free gauge so the scan
   sets the frame. Annealed association radius / weight over `outer_iters` rounds.
4. **Export** — corrected sparse model + per-photo `.xmp`.

## Key design decisions

- **Native multithreaded LiDAR cost.** The point-to-plane term is a native `NormalPrior`
  (anisotropic: stiff along the normal, weak in-plane), not a Python cost — so the whole solve
  runs on every core. This is what lets the model keep ~800k points instead of a tiny subset.
- **Georeferenced scans → local frame internally.** UTM/Lambert coordinates make the bundle
  adjust ill-conditioned and break RealityScan placement. The pipeline solves in a local frame
  and writes the result back in the scan's coordinates.
- **Dense scans get auto-voxelled** (~2 cm) for the plane index — survey clouds are far denser
  than plane fitting needs; this caps RAM/time with no effect on the fit.
- **Manual visual alignment is the reliable pre-align.** Auto-scale can't resolve repetitive
  structure + partial overlap (it locks a different wrong scale each run). The slider tool /
  correspondences file pin scale+pose by hand; the refine only polishes from there.
- **Mapper defaults to Incremental.** GLOMAP global silently folds repetitive/fisheye scenes
  into noise; for this tool's subjects, robust beats fast.
- **Fisheye is modelled, not dewarped.** Use `OPENCV_FISHEYE` + a matcher that survives the
  distortion (DSP-SIFT/affine-shape for COLMAP, learned features for hloc/GLUEMAP).

## Status & known gaps

- **The gate is SfM quality.** A broken sparse cloud (repetitive/fisheye matching failure)
  can't be aligned — hence the **Preview model** button to catch it early. hloc is the current
  best bet for wide/fisheye.
- **COLMAP→RealityScan axis convention** (Y/Z flip) is the usual one but unverified; A/B the
  `xmp_axis_flip` presets if cameras import mirrored.
- **Rolling shutter** isn't modelled; cull blurry frames.
