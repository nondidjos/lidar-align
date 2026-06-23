# Design & goals

## Goal

Take a photogrammetry sparse model (cameras + 3D points, up-to-scale and floating in an
arbitrary frame) and a survey-grade LiDAR scan, and bend the model onto the scan so the LiDAR
becomes the metric datum. Output corrected camera poses as RealityScan `.xmp` sidecars so the
mesh/texture step (in RealityScan) runs on a correctly scaled, placed, drift-free model.

Target accuracy: centimetre-level point-to-plane fit to the scan over the whole scene.

## Scope

**In:** the RealityScan round-trip (drive its CLI to pull the alignment in as COLMAP and push
corrected poses back), aligning all RealityScan components in one run, pre-alignment, LiDAR
point-to-plane refinement, QA, and RealityScan XMP export.

**Out:** matching/SfM (RealityScan does it, or bring a COLMAP model from elsewhere), meshing,
texturing, dense MVS. lidar-align only fixes pose/scale/drift.

## Target environment

- **Deploy box:** Windows, Intel i9-12900K (24 threads), RTX 3080, 64 GB RAM. Defaults are
  sized for this (e.g. `ba_max_points=800k`, all-core solve). Lower them on a weak machine.
- **Subjects:** architectural / survey scenes — often repetitive (stairs, railings, façades),
  sometimes shot on wide/fisheye action cams, against dense terrestrial scans (Leica etc.).
- **Matcher:** RealityScan (handles fisheye + repetition natively, exports COLMAP). Any other
  COLMAP sparse model works too, but lidar-align no longer builds one itself.

## Pipeline

1. **Match** — photos → a COLMAP sparse model. RealityScan (it handles fisheye and repeated
   structure natively, then exports to COLMAP), or any COLMAP model brought from elsewhere. This
   is the hard part for repetitive/fisheye scenes, and lidar-align leaves it to RealityScan.
2. **Pre-align** — coarse Sim3 onto the scan. RealityScan components georeferenced to the imported
   scan are already there; otherwise the visual slider tool / manual correspondences (reliable),
   or auto FPFH (fails on repetitive structure + partial overlap).
3. **Refine** — point-to-plane bundle adjustment: reuse pycolmap's Ceres problem, add
   `w·n·(X−p)` residuals against local cloud planes under a Huber loss, free gauge so the scan
   sets the frame. Annealed association radius / weight over `outer_iters` rounds.
4. **Export** — corrected sparse model + per-photo `.xmp`.

## Key design decisions

- **RealityScan matches; lidar-align does the metric bend.** RealityScan survives fisheye and
  repetition where COLMAP/hloc/GLUEMAP/VGGT (all pinhole-trained) fail, so it owns the matching.
  The GUI drives its CLI: Pull exports the alignment to COLMAP, Send re-imports the corrected
  poses — no hand-driven export/import menus.
- **All components in one run.** RealityScan splits hard scenes into components; those already in
  the scan's frame are refined automatically, unbound ones get a manual placement window. No
  per-component runs, no fragile auto-scale/FPFH on this path.
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

## Status & known gaps

- **The RealityScan round-trip is built against the 2.1.1 CLI but unverified on a live install.**
  Three things to confirm on the first real run: the COLMAP export preset (`exportRegistration`
  takes a settings `.xml` saved from the GUI — there is no CLI format flag), that it accepts the
  export directory, and that `-addFolder` re-imports the sidecar `.xmp` as locked poses (else fall
  back to `addImageWithCalibration` per image).
- **The gate is matching quality.** A broken sparse cloud (repetitive/fisheye matching failure)
  can't be aligned — hence the **Preview model** button to catch it early before refining. Getting
  a clean model is RealityScan's job.
- **COLMAP→RealityScan axis convention** (Y/Z flip) is the usual one but unverified; A/B the
  `xmp_axis_flip` presets if cameras import mirrored.
- **Rolling shutter** isn't modelled; cull blurry frames.
