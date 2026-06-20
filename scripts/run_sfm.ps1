# Run feature extraction and mapping using either GLOMAP (default) or GLUEMAP.
# Tuned for ~7k frames from a single Osmo (one camera, ordered video).
#
# GLOMAP is merged into COLMAP 4.x as `colmap global_mapper` -- no separate glomap binary.
# Requires colmap on PATH (extract colmap-x64-windows-cuda.zip and add its bin\ to PATH).
# Vocab tree for loop closure: https://demuc.de/colmap/

param(
  [string]$Images      = "data/images",
  [string]$Work        = "data/sfm",
  [string]$VocabTree   = "data/vocab_tree.bin",
  [string]$CameraModel = "OPENCV",        # OPENCV for rectilinear Osmo; OPENCV_FISHEYE if very wide
  [string]$Method      = "GLOMAP",        # GLOMAP or GLUEMAP
  [string]$GluemapConfig = ""             # Optional path to GLUEMAP config YAML
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Work | Out-Null

if ($Method -eq "GLUEMAP") {
  Write-Host "== GLUEMAP (Global SfM Meets Feedforward Reconstruction) =="
  if ($GluemapConfig -ne "") {
    gluemap-demo --images_path $Images --write_path $Work --intrinsics_mode SHARED --config $GluemapConfig
  } else {
    gluemap-demo --images_path $Images --write_path $Work --intrinsics_mode SHARED
  }
  if (-not $?) { throw "GLUEMAP failed" }
  Write-Host "GLUEMAP complete -> COLMAP model written under $Work"
  Exit
}

$db = Join-Path $Work "database.db"

Write-Host "== feature_extractor =="
colmap feature_extractor `
  --database_path $db `
  --image_path $Images `
  --ImageReader.single_camera 1 `
  --ImageReader.camera_model $CameraModel `
  --SiftExtraction.max_num_features 8192 `
  --SiftExtraction.estimate_affine_shape 1 `
  --SiftExtraction.domain_size_pooling 1 `
  --FeatureExtraction.use_gpu 1
if (-not $?) { throw "feature_extractor failed" }

Write-Host "== sequential_matcher (video order + loop closure) =="
colmap sequential_matcher `
  --database_path $db `
  --SequentialMatching.overlap 10 `
  --SequentialMatching.quadratic_overlap 1 `
  --SequentialMatching.loop_detection 1 `
  --SequentialMatching.vocab_tree_path $VocabTree `
  --FeatureMatching.use_gpu 1
if (-not $?) { throw "sequential_matcher failed" }

Write-Host "== global_mapper (GLOMAP global SfM, built into COLMAP 4.x) =="
$sparse = Join-Path $Work "sparse"
New-Item -ItemType Directory -Force $sparse | Out-Null
colmap global_mapper `
  --database_path $db `
  --image_path $Images `
  --output_path $sparse
if (-not $?) { throw "global_mapper failed" }

Write-Host "SfM complete -> $sparse/0"
