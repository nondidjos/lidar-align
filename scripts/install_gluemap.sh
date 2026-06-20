#!/usr/bin/env bash
# One-shot GLUEMAP installer for Linux / WSL2. No sudo: everything goes through micromamba.
# Requires an NVIDIA CUDA GPU (PyTorch GPU build). Idempotent - safe to re-run.
set -euo pipefail

PREFIX="${GLUEMAP_PREFIX:-$HOME/gluemap}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"

echo "== GLUEMAP installer =="
echo "repo:   $PREFIX"
echo "env:    micromamba 'gluemap'"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. GLUEMAP needs an NVIDIA CUDA GPU and will not run without one."
fi

# 1. micromamba
if ! command -v micromamba >/dev/null 2>&1; then
  echo "== installing micromamba =="
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
fi
MM="$(command -v micromamba || echo "$BIN/micromamba")"

# 2. conda env (Eigen/Ceres/METIS/Boost/compilers/CUDA/PyTorch from conda-forge - no apt)
if ! "$MM" env list | grep -qE '(^|\s)gluemap(\s|$)'; then
  echo "== creating env (downloads CUDA PyTorch, several GB) =="
  "$MM" create -y -n gluemap python=3.10
  "$MM" install -y -n gluemap -c conda-forge \
    eigen=3.4.0 ceres-solver=2.2.0 metis=5.1.0 boost=1.85.0 libstdcxx-ng=15.2.0 \
    pytorch-gpu=2.4.1 torchvision=0.19.1 cuda-version=12.4 compilers cmake make \
    pybind11 git wget
fi
run() { "$MM" run -n gluemap "$@"; }

# 3. clone
if [ ! -d "$PREFIX/.git" ]; then
  echo "== cloning gluemap =="
  run git clone https://github.com/colmap/gluemap.git "$PREFIX"
fi
cd "$PREFIX"
run git submodule update --init --recursive

# 4. build
echo "== building gluemap (pip install -e .) =="
run bash -lc 'CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .'

# 5. model checkpoints
echo "== downloading model checkpoints (several GB) =="
mkdir -p checkpoints
[ -f checkpoints/dino_salad.ckpt ] || run wget -O checkpoints/dino_salad.ckpt \
  https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
[ -f checkpoints/vggsfm_v2_0_0_track_predictor.bin ] || run wget -O checkpoints/vggsfm_v2_0_0_track_predictor.bin \
  https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt
run pip install -q -U "huggingface_hub[cli]"
if [ ! -f checkpoints/pi3.safetensors ]; then
  run hf download yyfz233/Pi3 model.safetensors --local-dir checkpoints
  mv -f checkpoints/model.safetensors checkpoints/pi3.safetensors
fi
[ -f "checkpoints/checkpoint-dg+visym.pth" ] || run hf download \
  doppelgangers25/doppelgangers_plusplus checkpoint-dg+visym.pth --local-dir checkpoints

# 6. PATH wrapper so `which gluemap-demo` works in a plain (non-activated) shell
cat > "$BIN/gluemap-demo" <<EOF
#!/usr/bin/env bash
exec "$MM" run -n gluemap gluemap-demo "\$@"
EOF
chmod +x "$BIN/gluemap-demo"

echo "== verify =="
run python -c "import pygluemap; print('pygluemap', pygluemap.__file__)" || true
echo
echo "DONE. gluemap-demo -> $BIN/gluemap-demo"
echo "If the app can't find it, add ~/.local/bin to PATH:  echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc"
