#!/usr/bin/env bash
# COLMAP feature extraction + sequential matching, then GLOMAP global mapping, or GLUEMAP mapping.
# Tuned for ~7k frames from a single Osmo (one camera, ordered video).
# GLOMAP is merged into COLMAP 4.x as `colmap global_mapper` -- no separate glomap binary.
# Requires colmap or gluemap-demo on PATH.
# Usage: ./run_sfm.sh [images] [work] [vocab_tree] [camera_model] [method] [gluemap_config]
set -euo pipefail

IMAGES="${1:-data/images}"
WORK="${2:-data/sfm}"
VOCAB="${3:-data/vocab_tree.bin}"
CAMERA_MODEL="${4:-OPENCV}"         # OPENCV (rectilinear) or OPENCV_FISHEYE (very wide)
METHOD="${5:-GLOMAP}"               # GLOMAP or GLUEMAP
GLUEMAP_CONFIG="${6:-}"
MAX_FEATURES="${7:-4096}"           # matches the GUI's Balanced preset; raise for more detail

mkdir -p "$WORK"

if [ "$METHOD" = "GLUEMAP" ]; then
  echo "== GLUEMAP (Global SfM Meets Feedforward Reconstruction) =="
  if [ -n "$GLUEMAP_CONFIG" ]; then
    gluemap-demo --images_path "$IMAGES" --write_path "$WORK" --intrinsics_mode SHARED --config "$GLUEMAP_CONFIG"
  else
    gluemap-demo --images_path "$IMAGES" --write_path "$WORK" --intrinsics_mode SHARED
  fi
  echo "GLUEMAP complete -> COLMAP model written under $WORK"
  exit 0
fi

DB="$WORK/database.db"

# GPU SIFT at 4096 features. NOTE: --SiftExtraction.estimate_affine_shape / --domain_size_pooling
# improve matches on blurry frames but force COLMAP onto CPU-only SIFT, which on thousands of
# sharp frames means a multi-hour run and a huge database - so they are deliberately NOT used
# here. Only add them for genuinely soft footage.
echo "== feature_extractor =="
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model "$CAMERA_MODEL" \
  --SiftExtraction.max_num_features "$MAX_FEATURES" \
  --FeatureExtraction.use_gpu 1

echo "== sequential_matcher (video order + loop closure) =="
colmap sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 10 \
  --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 1 \
  --SequentialMatching.vocab_tree_path "$VOCAB" \
  --FeatureMatching.use_gpu 1

# Estimate focal priors from the view graph (video frames usually lack EXIF focal, so
# global_mapper otherwise rejects many pairs). Non-fatal if the command is unavailable.
echo "== view_graph_calibrator (focal priors) =="
colmap view_graph_calibrator --database_path "$DB" || echo "view_graph_calibrator unavailable/failed - continuing"

echo "== global_mapper (GLOMAP global SfM, built into COLMAP 4.x) =="
mkdir -p "$WORK/sparse"
colmap global_mapper \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --output_path "$WORK/sparse"

echo "SfM complete -> $WORK/sparse/0"
