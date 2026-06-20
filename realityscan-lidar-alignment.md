# Getting Photos to Align to an RTC360 Scan in RealityScan

Notes on aligning Osmo video frames to a Leica RTC360 reference scan — what actually works, what doesn't, and when to give up on RealityScan's built-in tools and go the custom route.

---

## Why "lock the LiDAR" doesn't work

The first thing you try is locking the scan and expecting the photos to snap to it. It doesn't work, and it's not a weight setting.

RealityScan aligns photos to a scan by **feature matching** — it projects the scan's color or intensity into image space and matches your photos against that. It does not snap photos to geometry directly. Which means:

- A **locked point cloud with no color** (XYZ + intensity only) is visually invisible to the matcher. Geometrically locking it does nothing for alignment.
- **Manual control points can't carry it alone.** A handful of hand-placed ties can't out-vote a warped photo block with thousands of images. You need automatic correspondences in the thousands to actually pull the solve.

**The fix is giving the scan color so the matcher can generate those correspondences itself.** That's it — not a weight slider, not more control points.

---

## Confirm your scan has color

`Features source = Color` is an **import setting**, not something you can change mid-project. Check it now before going further.

Pick a station in the **1Ds panel** → view it in 3D:
- **Renders in RGB color** → good, keep going.
- **Renders grey / intensity** → you need to re-import with color. Nothing below works without it.

---

## Step 1 — Alignment settings

Go to **ALIGNMENT** tab → **Settings** (right panel):

| Setting | Value | Why |
|---|---|---|
| Prefer images as feature source | Enabled | Match scan imagery against your photos |
| Image overlap | High | More matches, slower |
| Max features per image | ~40,000 | More tie candidates in weak-coverage areas |
| Detector sensitivity | High | More features detected per image |
| Image downscale factor | 1 | Full-resolution feature detection |

---

## Step 2 — Align

**ALIGNMENT → Align Images.** Then check **1Ds panel → Components**.

- **One component** containing both photos and scan stations → they're now solved in the LiDAR frame via automatic color ties. Proceed to reconstruction.
- **Two separate components** → the matcher didn't bridge them. See below.

---

## Step 3 — If it splits into separate components

**Option A: Try intensity matching.** Re-import the scan with `Features source = Intensity` instead of Color and run alignment again. Worth trying if the color matching is weak, but it requires a re-import.

**Option B: Bridge with control points.** Place 4+ control points that are visible in both the photo component and the scan component, then re-align. The points don't carry the solve — they just tell the matcher which component is which so the automatic ties can bridge across.

### How to place bridging control points

Pick 4+ sharp, unambiguous features visible in **both** photos and the scan (corners, bolt heads, painted markings). Spread them across the overlap zone.

For each feature:
1. **Open a photo** in 2Ds view (double-click in 1Ds panel).
2. **Right-click the feature** → **Create Control Point** (may say "Add control point" depending on version).
3. **Open 2–3 more photos** with the same feature visible and click the matching spot in each.
4. **Now tie it to the scan**: with the same control point selected, click the matching spot either in the **3Ds view** on the scan surface, or in the station's panorama image. This is the step people miss — without a scan-side observation, the control point doesn't bridge anything.
5. Set **Position accuracy** to a small value (metres). Leave **Image measurements accuracy** at default.

Repeat for all 4+ points, then **Align Images** again.

---

## Step 4 — Reconstruct

**SCENE/VIEW tab** → draw a reconstruction region box → **RECONSTRUCTION → High Detail**.

---

## When to give up on RealityScan

If you're working with a large video capture (thousands of frames from a moving camera) and a big scan, RealityScan's built-in LiDAR alignment will likely disappoint you regardless of the settings. The core problem is that it wasn't designed for this:

- **Video frames** carry rolling shutter and motion blur that RealityScan doesn't model.
- **Thousands of frames** need sequential + vocabulary-tree matching to be tractable — exhaustive matching at 7k frames is ~24 million pairs.
- **The LiDAR constraint** in RealityScan is soft. It can't force the photogrammetry to bend onto the scan the way a direct geometric constraint can.

For this scale, the custom route below is the one that actually works.

---

# Custom route — 7k Osmo frames + 290 GB RTC360

This is the pipeline this repo implements: GLOMAP (or GLUEMAP for low visual overlap) for the image solve, `pyceres` point-to-plane refinement against the LiDAR cloud. See the [README](README.md) for the full walkthrough.

## Why GLOMAP/GLUEMAP + custom refinement, not Colmap-PCD

Colmap-PCD does exactly the right thing mathematically — it adds LiDAR point-to-plane residuals to the bundle adjustment. But it's built on older COLMAP with incremental SfM, tested to 450 images, Ubuntu-only, and hasn't been touched since ~2023. At 7k frames it's not a real option.

The approach here decouples the two concerns:

- **GLOMAP / GLUEMAP** handles the image solve. Global SfM is 1–2 orders faster than incremental and built for video frames at scale. GLUEMAP is especially suited for wide baselines and low overlap.
- **The LiDAR constraint** is a post-BA refinement step — same math as Colmap-PCD, but bolted onto pycolmap's bundle adjuster rather than forked into COLMAP.

Neither half has to do the other's job.

## The pipeline

```
[1] Frame cull       sharp-frame-extractor    7k → ~3–4k sharp frames
[2] Features+match   COLMAP (GPU SIFT)         sequential + vocab-tree loop closure
[3] Global solve     GLOMAP / GLUEMAP          poses + sparse cloud
[4] LiDAR prep       Cyclone decimate → LAZ    voxel 3–5 cm; streams out-of-core in lidar-align
[5] Refine           lidar-align               point-to-plane BA, LiDAR is the datum
[6] Export           .xmp sidecars             back into RealityScan, or COLMAP MVS
```

## LiDAR prep

Export from Cyclone REGISTER 360 as a **unified, decimated cloud** (Reduce cloud → single point cloud, resample at 3–5 cm average spacing) as PLY or E57. 290 GB collapses to a manageable size. You can also feed the raw E57 directly to lidar-align — it reads per-scan poses and crops to the SfM bounding box, so only the relevant portion loads.

## Frame culling

Run sharpness-based selection before SfM. [sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor) (Tenengrad / Sobel energy) works well. At 60–80% target overlap, 7k video frames typically thin to 3–4k usable shots. Fewer, sharper frames = a faster and more stable solve.

## Rolling shutter — the wildcard

COLMAP (and RealityScan) don't model rolling shutter. A moving Osmo produces systematic RS skew that directly degrades pose registration. Frame culling and tight feature settings help, but they don't fix it. If RS is the dominant error, even a good LiDAR constraint can only partially overcome it. This is worth flagging because it may also be why RealityScan was struggling in the first place.

---

## References

- [RealityScan — LiDAR scan import](https://rshelp.capturingreality.com/en-US/tutorials/importlaser.htm)
- [Laser scan import (Epic KB)](https://dev.epicgames.com/community/learning/knowledge-base/Xdo8/realityscan-laser-scan-import)
- [RealityScan alignment settings](https://rshelp.capturingreality.com/en-US/appbasics/alignsettings.htm)
- [RealityScan camera priors](https://rshelp.capturingreality.com/en-US/appbasics/camerasettings_priors.htm)
- [Colmap-PCD](https://github.com/XiaoBaiiiiii/colmap-pcd) — the reference implementation for the point-to-plane BA approach
- [GLOMAP](https://github.com/colmap/glomap) — global SfM mapper
- [GLOMAP Windows build](https://github.com/jonstephens85/glomap_windows)
- [GLOMAP paper](https://arxiv.org/pdf/2407.20219)
- [GLUEMAP](https://github.com/colmap/gluemap) — hybrid SfM mapper (great for low visual overlap / wide baselines)
- [Cyclone REGISTER 360 — decimate on export](https://rcdocs.leica-geosystems.com/cyclone-register-360/latest/option-to-decimate-point-clouds-upon-export)
- [sharp-frame-extractor](https://github.com/cansik/sharp-frame-extractor)
