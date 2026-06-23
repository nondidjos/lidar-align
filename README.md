# lidar-align

Pin a photogrammetry model to a LiDAR scan.

COLMAP / GLOMAP / hloc / GLUEMAP give a sharp sparse model, but it floats — wrong scale,
wrong orientation, slow drift — because the solver only minimises pixel error. lidar-align
adds point-to-plane terms from a survey cloud to that same bundle adjustment and lets the
scan hold the coordinate frame, so the model settles onto it.

Out: a corrected sparse model and RealityScan `.xmp` pose files. Meshing/texturing stay in
RealityScan; this only fixes alignment. See [DESIGN.md](DESIGN.md) for goals and internals.

## Install

Python 3.10+ and COLMAP on PATH; the rest is pip.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

COLMAP is a separate download (the GUI can fetch it). GLOMAP ships inside COLMAP 4.x.

## Use

```bash
.venv\Scripts\python ui\refine_gui.py      # or double-click run_gui.bat
```

1. **Build model** — runs SfM on a photo folder. Pick a camera type; for wide/fisheye use the
   `hloc` engine (best matching) or the Incremental mapper.
2. **Preview model** — open the cloud in 3D. If it's noise, the SfM failed — fix that first.
3. **Align to cloud** — bends the model onto the LiDAR. For repetitive scenes auto-scale can't
   handle, click **Align visually** first and place it with the sliders.

CLI: `.venv\Scripts\python refine_align.py --config config.yaml` (copy `config.example.yaml`).

## Inputs

- **Model** — a COLMAP sparse reconstruction (`sparse/0`), built here or brought from
  COLMAP / GLOMAP / hloc / GLUEMAP.
- **Reference cloud** — `.las` / `.laz` / `.e57`, streamed from disk and cropped to the photo
  volume while reading, so a huge survey never lands in RAM. **Merge scans** collapses a
  multi-station e57 to one cloud at native spacing for fast reloads.

## Output

- `sparse_refined/` — the aligned model.
- `qa/residual_before.ply`, `residual_after.ply` — coloured by point-to-plane distance.
- `.xmp` sidecars (one per photo) — drop next to the images, re-import into RealityScan. If
  cameras come in mirrored/upside-down, switch `xmp_axis_flip` (`identity` / `flip_xz` / `flip_xy`).

## Key options

| Key | Default | Meaning |
|---|---|---|
| `w_lidar` | `5.0` | How hard the cloud pulls vs the photo geometry |
| `max_assoc_dist` | `0.5` | Furthest a point matches the cloud (m) |
| `outer_iters` | `8` | Re-association rounds |
| `voxel` | auto | Cloud spacing (m); blank auto-caps dense scans to ~2 cm |
| `max_lidar_residuals` | `150000` | Point-to-cloud matches per round |
| `ba_max_points` | `800000` | Model points kept for the solve |

Full set in `config.example.yaml`. Manual `correspondences` are config-only.

## Build (standalone exe)

```bash
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller lidar-align.spec --noconfirm
```

`dist/lidar-align/` is ~190 MB zipped. Verify with `lidar-align.exe --selftest`.

## Tests

```bash
bash scripts/run_tests.sh
```

## Notes

- Built against pycolmap 4.0.4 / pyceres 2.6 — re-run the tests if you bump either.
- The `pyceres` PyPI wheels lack CUDA; "Ceres compiled without CUDA" warnings are expected,
  the CPU solver works fine.
