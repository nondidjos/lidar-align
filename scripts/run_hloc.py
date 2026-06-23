#!/usr/bin/env python3
"""Native-Windows learned-feature SfM via hloc (SuperPoint + LightGlue) -> a COLMAP model that
lidar-align reads. Runs on the GPU with NO WSL / restart / password - the path when GLUEMAP's
Linux+CUDA install is blocked. Learned features match across heavy fisheye distortion AND repeated
structure (stairs, balusters) where plain SIFT collapses into noise.

ONE-TIME SETUP (no admin, no WSL):
    python -m venv hloc-env
    hloc-env\\Scripts\\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install git+https://github.com/cvg/Hierarchical-Localization.git
    # LightGlue installs as an hloc dependency; if it's missing:  pip install lightglue

RUN (in that env):
    python run_hloc.py --images "C:/path/to/frames" --output "C:/path/to/sfm_hloc" \
        --camera-model OPENCV_FISHEYE
Then in lidar-align set 'Existing model' to  <output>/sparse  (hloc writes <output>/sparse/0).

NOTES
- Thin the frames first (every 2nd-3rd) for speed - learned matching is heavier than SIFT.
- num-matched controls retrieval breadth: more pairs catch the revisited areas (down the stairs,
  around the walled space) at the cost of time.
- This is a standalone script run in its OWN python env (torch isn't bundled in the lidar-align
  exe). hloc's API shifts between versions; if a call name changed, the error names the symbol and
  it's a one-line fix.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in _IMG_EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description="learned-feature SfM (hloc) -> COLMAP model")
    ap.add_argument("--images", required=True, help="folder of frames")
    ap.add_argument("--output", required=True, help="output folder (gets features/, pairs.txt, sparse/)")
    ap.add_argument("--camera-model", default="OPENCV_FISHEYE",
                    help="COLMAP camera model: OPENCV_FISHEYE for wide/fisheye, OPENCV otherwise")
    ap.add_argument("--num-matched", type=int, default=20,
                    help="retrieval pairs per image (more = catches loops/revisits, slower)")
    ap.add_argument("--feature", default="superpoint_max",
                    help="hloc feature conf (superpoint_max = dense keypoints, best for hard scenes)")
    ap.add_argument("--matcher", default="superpoint+lightglue", help="hloc matcher conf")
    args = ap.parse_args()

    images = Path(args.images)
    if not images.is_dir():
        print(f"[hloc] images folder not found: {images}")
        return 2
    n_imgs = _count_images(images)
    if n_imgs < 3:
        print(f"[hloc] only {n_imgs} images found under {images}")
        return 2

    try:
        import pycolmap
        from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
    except Exception as e:
        print(f"[hloc] import failed ({e}).\n"
              f"       install into a python env:  pip install torch torchvision "
              f"--index-url https://download.pytorch.org/whl/cu124  &&  "
              f"pip install git+https://github.com/cvg/Hierarchical-Localization.git")
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sfm_pairs = out / "pairs.txt"
    sfm_dir = out / "sparse"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs[args.feature]
    matcher_conf = match_features.confs[args.matcher]

    print(f"[hloc] {n_imgs} images | feature={args.feature} matcher={args.matcher} "
          f"camera={args.camera_model}")

    # 1. global descriptors (NetVLAD) -> retrieval pairs. Retrieval (not just sequential) is what
    #    links the revisited places - down the stairs, around the walled space.
    print("[hloc] (1/4) NetVLAD retrieval…")
    retrieval_path = extract_features.main(retrieval_conf, images, out)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=min(args.num_matched, n_imgs - 1))

    # 2. local features (SuperPoint) - learned, distortion-robust.
    print("[hloc] (2/4) SuperPoint features…")
    feature_path = extract_features.main(feature_conf, images, out)

    # 3. match (LightGlue) - learned matcher, survives the fisheye warp + repeats.
    print("[hloc] (3/4) LightGlue matching…")
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], out)

    # 4. reconstruct into a COLMAP model with the fisheye camera model.
    print(f"[hloc] (4/4) reconstruction (camera_model={args.camera_model})…")
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, feature_path, match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=dict(camera_model=args.camera_model))

    if model is None:
        print("[hloc] reconstruction produced no model - check the logs above")
        return 1
    print(f"[hloc] DONE -> {sfm_dir / '0'}  ({model.num_reg_images()} imgs, "
          f"{model.num_points3D()} points)")
    print(f"[hloc] in lidar-align, set 'Existing model' to:  {sfm_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
