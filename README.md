# lidar-align

Pin a photogrammetry model to a LiDAR scan.

Photogrammetry (COLMAP, GLOMAP, GLUEMAP) gives you a sharp sparse model, but it floats —
wrong scale, wrong orientation, and a slow drift across a large scene — because the solver
only ever minimises pixel error. lidar-align drops point-to-plane terms from a survey cloud
into that same bundle adjustment and lets the scan hold the coordinate frame, so the model
settles onto it.

You get back a corrected sparse model and RealityScan `.xmp` pose files. Meshing and
texturing stay in RealityScan or COLMAP MVS; this only fixes alignment.

## Install

Python 3.10+ and COLMAP on PATH. The rest is pip.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Windows
# .venv/bin/pip install -r requirements.txt       # macOS/Linux
```

COLMAP is a separate download — the GUI can grab it, or pull the Windows CUDA build from
[colmap.github.io](https://colmap.github.io). GLOMAP lives inside COLMAP 4.x as
`colmap global_mapper`.

## Use

GUI:

```bash
.venv\Scripts\python ui\refine_gui.py        # or double-click run_gui.bat
```

Three paths — photos, the reference cloud, an output folder — plus a camera type. Hit
**Build model** to run SfM, then **Align**. Everything else sits under Advanced; the
defaults are a fine starting point. The solver log runs live and Stop bails out after the
current round.

CLI:

```bash
.venv\Scripts\python refine_align.py --config config.yaml
```

Copy `config.example.yaml` and set `sparse_in`, `lidar`, `sparse_out`.

## Inputs

**Camera model** — a COLMAP-format sparse reconstruction (`sparse/0`). Build it in the GUI,
with `scripts/run_sfm.*`, or bring your own from COLMAP / GLOMAP / GLUEMAP.

> [!TIP]
> If your photos have low overlap or wide baselines and SIFT matching falls apart, try
> [GLUEMAP](https://github.com/colmap/gluemap). It writes the same `cameras.bin` /
> `images.bin` / `points3D.bin`, so lidar-align reads it as-is. GLUEMAP needs Linux + an
> NVIDIA CUDA GPU; on Windows the GUI's **Install GLUEMAP** button builds it inside WSL2 and
> then runs it through `wsl`. Pick GLUEMAP as the SfM engine once it's installed.

**Reference cloud** — `.las`/`.laz` stream from disk and get cropped to the photo volume as
they read, so a huge survey never lands in RAM. `.e57` works directly (per-scan poses
applied). Stations and metadata are ignored, only the points matter. For a big survey, thin
it once: `python scripts/e57_to_laz.py site.e57 out.laz --voxel 0.03`.

## How it works

It reuses pycolmap's Ceres bundle adjuster rather than building a second solver: grab the
existing `Problem`, and for each sparse point add a `w * n·(X - p)` point-to-plane residual
against the nearest local cloud plane (PCA over k neighbours), under a Huber loss. The gauge
is left free, so those terms — not COLMAP's arbitrary frame — set position, orientation and
scale. A coarse pre-align gets the model roughly into place first: manual correspondences if
you have them, otherwise FPFH or a centroid guess.

Cost tracks the sparse point count, not the cloud size. The points query the cloud; the
cloud is never walked whole, so a 200 GB scan is fine.

## Output

- `sparse_refined/` — the aligned model.
- `qa/residual_before.ply` and `residual_after.ply` — coloured by point-to-plane distance.
  Open both in CloudCompare; the after should be tighter.
- `.xmp` sidecars, one per photo, with the corrected pose. Drop them next to the images and
  re-import into RealityScan.

> The COLMAP→RealityScan axis flip (Y and Z) is the usual one but unverified here. Import a
> single image first; if it comes in mirrored or upside down, switch `xmp_axis_flip` to
> `identity`, `flip_xz`, or `flip_xy`.

## Config

| Key | Default | Meaning |
|---|---|---|
| `prealign` | `true` | Coarse alignment before refining |
| `voxel` | `null` | Thin the cloud before indexing (m); `null` = full res |
| `w_lidar` | `5.0` | How hard the cloud pulls vs the photo geometry |
| `outer_iters` | `8` | Re-association rounds |
| `max_assoc_dist` | `0.5` | Furthest a point can match the cloud (m) |
| `planarity_min` | `0.1` | Skip non-flat spots (edges, clutter) |
| `early_stop_tol` | `0.0` | Stop once a round improves by less than this (m); 0 = off |
| `xmp_pose_prior` | `locked` | How RealityScan treats the poses |

The rest is in `config.example.yaml`. Manual `correspondences` are config-only.

## Layout

```
lidar_align/        refine.py, prealign.py, lidar_index.py, export_xmp.py, colmap_io.py, qa.py
ui/refine_gui.py    desktop GUI (Tkinter)
scripts/            run_sfm.ps1/.sh, e57_to_laz.py
tests/              synthetic test suites
refine_align.py     CLI
```

## Tests

```bash
bash scripts/run_tests.sh
```

Synthetic coverage: plane fitting, pre-align, FPFH, end-to-end recovery, LAS streaming, e57
read/convert, XMP round-trip, and the on-disk pipeline.

## Standalone build

For a machine with no Python:

```bash
.venv\Scripts\pip install pyinstaller
.venv\Scripts\python -m PyInstaller --noconfirm --windowed --name lidar-align \
  --collect-all open3d --collect-all pycolmap --collect-all pyceres \
  --collect-all laspy --collect-all lazrs --collect-all pye57 \
  --collect-submodules scipy --collect-submodules lidar_align ui/refine_gui.py
```

`dist/lidar-align/` ends up around 300-500 MB. Sanity-check it with `lidar-align.exe --selftest`.

## Notes

- Rolling shutter isn't modelled. Frames from a moving camera carry skew that hurts the
  solve; culling blurry frames helps but won't fix it.
- Built against pycolmap 4.0.4 / pyceres 2.6 — re-run the tests if you bump either.
