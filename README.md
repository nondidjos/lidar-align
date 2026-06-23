# lidar-align

Make RealityScan's cameras sit exactly on your LiDAR scan.

RealityScan aligns photos well — even wide/fisheye, even repetitive façades — but the result
floats: no true scale, and slow drift over a long scene, because nothing ties it to the ground.
lidar-align bends that alignment onto a survey scan with point-to-plane bundle adjustment and lets
the scan hold the coordinate frame. The model comes out metric, placed, and drift-free.

Meshing and texturing stay in RealityScan. This only fixes pose, scale, and drift.
See [DESIGN.md](DESIGN.md) for goals and internals.

## The workflow

Everything runs from one desktop window:

```bash
.venv\Scripts\python ui\refine_gui.py      # or double-click run_gui.bat
```

1. **In RealityScan** — import your photos and the LiDAR scan, and align. RealityScan does the hard
   matching (fisheye, repeated structure). It may split the scene into several components.
2. **Pull from RealityScan** — exports that alignment to COLMAP and points lidar-align at it.
3. **Align components** — bends every component onto the scan in one run. Components RealityScan
   already placed on the scan are refined automatically; unbound ones pop a window to place by hand
   (scale / rotate / move), then refine.
4. **Send to RealityScan** — re-imports the corrected poses into a fresh project, ready to mesh.

The corrected poses land as `.xmp` sidecars next to your photos, in the scan's own coordinates.

## Without RealityScan

You don't have to use the round-trip. Point **"Existing model"** at any COLMAP sparse model (from
RealityScan, COLMAP, or anywhere else), give it a reference cloud, and align directly:

- **Preview model** opens the model in 3D — if it's noise, the matching failed upstream; fix that
  before aligning.
- **Align to cloud** does the bend. For repetitive scenes where auto-scale can't lock, click
  **Align visually** and place it with the sliders first.

## Inputs

- **Reference cloud** — `.las` / `.laz` / `.e57`. Streamed from disk and cropped to the photo
  volume as it reads, so a huge survey scan never lands in RAM. **Merge scans** collapses a
  multi-station e57 into one cloud for fast reloads.
- **Georeferenced scans** (UTM / national grid) are solved in a local frame and written back in the
  scan's own coordinates — no offset files, no extra clouds.

## Output

- The aligned model, plus `.xmp` pose sidecars (one per photo) for RealityScan.
- `qa/residual_before.ply` / `residual_after.ply` — coloured by distance to the cloud, so you can
  watch the fit improve (open in CloudCompare).
- If cameras import mirrored or upside-down, change the Axis convention and re-export.

Tuning lives in `config.example.yaml` (CLI: `python refine_align.py --config config.yaml`); the
defaults are sized for a strong workstation.

## Install

Python 3.10+; the rest is pip.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The RealityScan round-trip needs RealityScan 2.1+ (it drives RealityScan's CLI).

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
- The pyceres PyPI wheels lack CUDA; "Ceres compiled without CUDA" warnings are expected; the CPU
  solver works fine.
