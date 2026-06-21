#!/usr/bin/env python
"""Desktop GUI for lidar-align.

One window, two actions:
  Build model   runs COLMAP + GLOMAP on a photo folder to make a sparse camera model.
  Align         bends that model onto the LiDAR cloud and writes RealityScan XMP poses.

You give three paths (photos, LiDAR, project folder); everything else is derived under the
project folder. Advanced knobs are hidden until you open them.

Run:    .venv\\Scripts\\python ui\\refine_gui.py      (or run_gui.bat)
Frozen: lidar-align.exe --selftest
"""
from __future__ import annotations
import os
os.environ.setdefault("GLOG_minloglevel", "3")

import contextlib
import io
import json
import queue
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Project root: the exe's folder when frozen, else the repo root (this file is ui/).
if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(sys.executable)
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

# Friendly camera names -> COLMAP camera_model. OPENCV covers most lenses; fisheye for
# action cams / very wide; simple radial for a quick single-parameter fit.
CAMERAS = [
    ("Standard lens", "OPENCV"),
    ("Wide / action cam (fisheye)", "OPENCV_FISHEYE"),
    ("Simple (single distortion term)", "SIMPLE_RADIAL"),
]
_CAMERA_MODEL = dict(CAMERAS)

# SfM speed/quality presets -> (feature quality, max features, frames matched ahead).
# Grounded in COLMAP's own quality tiers: plain GPU SIFT (no affine/DSP) is the norm;
# DSP-SIFT/affine is the slow CPU "max" path COLMAP reserves for hard cases.
_SFM_PRESETS = {
    "Fast (low RAM)":     ("fast", 2048, 5),
    "Balanced":           ("fast", 4096, 10),
    "High detail":        ("fast", 8192, 10),
    "Max quality (slow)": ("high", 8192, 15),
}
_SFM_PRESET_NAMES = list(_SFM_PRESETS) + ["Custom"]

_COLMAP_LOCAL_DIR = os.path.join(os.environ.get("APPDATA", ROOT), "lidar-align", "colmap")
_SETTINGS_FILE = os.path.join(os.environ.get("APPDATA", ROOT), "lidar-align", "gui_settings.json")
# Cap the on-screen log: an unbounded Text widget gets progressively slower to insert into and
# scroll, which is what makes the window go "Not Responding" on long, chatty runs (6h COLMAP).
_LOG_MAX_LINES = 5000


def _ssl_context():
    """A TLS context that VERIFIES certificates (certifi bundle if present, else the system
    store). We download and then run colmap.exe, so never disable verification."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _colmap_help(colmap, cmd):
    """Return the `colmap <cmd> -h` text, so we can pick option names that match the
    installed COLMAP. (4.0 renamed e.g. SiftExtraction.use_gpu -> FeatureExtraction.use_gpu;
    probing the actual binary avoids guessing per version.)"""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        r = subprocess.run([colmap, cmd, "-h"], capture_output=True, text=True,
                           timeout=30, creationflags=flags)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def _resolve(help_text, *candidates):
    """First candidate option present in help_text; falls back to the first (best effort
    when help is unavailable). Use for options that must be passed."""
    for c in candidates:
        if c in help_text:
            return c
    return candidates[0]


def _present(help_text, *candidates):
    """First candidate present in help_text, or None when help is non-empty and none match
    (so optional flags can be skipped rather than rejected)."""
    if not help_text:
        return candidates[0]
    for c in candidates:
        if c in help_text:
            return c
    return None


def _find_colmap():
    """colmap.exe from PATH first, then our managed local install."""
    found = shutil.which("colmap")
    if found:
        return found
    for c in (os.path.join(_COLMAP_LOCAL_DIR, "bin", "colmap.exe"),
              os.path.join(_COLMAP_LOCAL_DIR, "colmap.exe")):
        if os.path.isfile(c):
            return c
    if os.path.isdir(_COLMAP_LOCAL_DIR):
        for d, _, files in os.walk(_COLMAP_LOCAL_DIR):
            if "colmap.exe" in files:
                return os.path.join(d, "colmap.exe")
    return None


def _latest_colmap_windows_url():
    """(url, filename) for the latest Windows COLMAP zip (CUDA preferred)."""
    api = "https://api.github.com/repos/colmap/colmap/releases/latest"
    req = urllib.request.Request(api, headers={"User-Agent": "lidar-align-gui"})
    with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as r:
        data = json.loads(r.read())
    assets = data.get("assets", [])
    cuda = [a for a in assets if "windows" in a["name"].lower()
            and "cuda" in a["name"].lower() and a["name"].endswith(".zip")]
    win = [a for a in assets if "windows" in a["name"].lower() and a["name"].endswith(".zip")]
    chosen = cuda or win
    if not chosen:
        raise RuntimeError("No Windows COLMAP zip in the latest GitHub release.")
    return chosen[0]["browser_download_url"], chosen[0]["name"]


class _QueueWriter(io.TextIOBase):
    """File-like that forwards writes onto the GUI log queue (thread-safe)."""
    def __init__(self, q):
        self._q = q

    def write(self, s):
        if s:
            self._q.put(("log", s))
        return len(s)

    def flush(self):
        pass


def _validate_images_dir(images_dir, sparse_out):
    """After refine(): check the model's image names resolve under images_dir."""
    try:
        import pycolmap
    except ImportError:
        return
    try:
        rec = pycolmap.Reconstruction(sparse_out)
    except Exception as exc:
        print(f"[images check] could not read refined model: {exc}")
        return
    total = found = 0
    missing = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        total += 1
        if (os.path.isfile(os.path.join(images_dir, img.name))
                or os.path.isfile(os.path.join(images_dir, os.path.basename(img.name)))):
            found += 1
        else:
            missing.append(img.name)
    print(f"\n[images check] {found}/{total} image names matched under {images_dir}")
    if missing:
        for m in missing[:10]:
            print(f"    missing: {m}")
        if len(missing) > 10:
            print(f"    … and {len(missing) - 10} more")


def _resource(rel):
    """Resolve a repo resource path, working both from source and from a PyInstaller bundle."""
    parts = rel.split("/")
    base = getattr(sys, "_MEIPASS", ROOT)
    p = os.path.join(base, *parts)
    return p if os.path.exists(p) else os.path.join(ROOT, *parts)


def _winpath_to_wsl(p):
    """C:\\Users\\x -> /mnt/c/Users/x, so a Windows path can be handed to a WSL command."""
    p = os.path.abspath(p)
    drive, rest = os.path.splitdrive(p)
    if drive:
        return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
    return p.replace("\\", "/")


def _find_gluemap():
    """Locate gluemap-demo (Linux+CUDA tool). Returns (mode, exe):
       ('win', path)  native Windows exe on PATH (rare),
       ('wsl', name)  found inside WSL -> run as `wsl <name> ...`,
       (None, None)   not found."""
    for name in ("gluemap-demo", "gluemap"):
        p = shutil.which(name)
        if p:
            return "win", p
    if shutil.which("wsl"):
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            # login shell so ~/.local/bin (where the installer drops the wrapper) is on PATH
            r = subprocess.run(["wsl", "bash", "-lc", "command -v gluemap-demo || command -v gluemap"],
                               capture_output=True, text=True, timeout=25, creationflags=flags)
            if r.returncode == 0 and r.stdout.strip():
                return "wsl", "gluemap-demo"
        except Exception:
            pass
    return None, None


def _wsl_ready():
    """True only if WSL has a usable Linux distro. `wsl.exe` ships with Windows even when no
    distro is installed, so checking for the launcher isn't enough - actually run a no-op."""
    if not shutil.which("wsl"):
        return False
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        r = subprocess.run(["wsl", "-e", "true"], capture_output=True, timeout=20,
                           creationflags=flags)
        return r.returncode == 0
    except Exception:
        return False


def _kill_tree(proc):
    """Kill a subprocess AND its children. colmap can spawn workers, and on Windows
    Popen.terminate() only kills the parent - so Stop / window-close use taskkill /T to avoid
    orphaned colmap.exe. Also unblocks a reader stuck on a quiet child's stdout."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            proc.terminate()
    except Exception:
        pass


class _Tip:
    """Hover tooltip: a small popup shown near the cursor while over `widget`."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + 18
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#fffbe6",
                 relief="solid", borderwidth=1, wraplength=340,
                 font=("Segoe UI", 9), padx=6, pady=4).pack()

    def _hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("lidar-align")
        self.q = queue.Queue()
        self._worker = None
        self._cancel = threading.Event()
        self._colmap_exe = _find_colmap()
        self.var = {}            # key -> tk variable
        self._adv_open = False
        self._throb_state = 0
        self._job_start = None      # wall-clock start of the running job
        self._last_beat = 0.0       # last heartbeat-to-log time
        self._last_output = 0.0     # last time a log line arrived (quiet detection)
        self._proc = None           # current subprocess, so Stop / close can kill it
        self._chain = None          # (refine_kwargs, images_dir, project): auto-align after Build
        self._build()
        self._load_settings()
        self.root.after(100, self._drain)
        self.root.after(100, self._throb_tick)  # start throbber animation

    # ── persist the form across sessions ──────────────────────────────────────
    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE) as f:
                data = json.load(f)
        except Exception:
            return
        for k, val in data.items():
            if k in self.var:
                try:
                    self.var[k].set(val)
                except Exception:
                    pass
        if "sfm_preset" not in data:   # settings saved before presets -> reset SfM to the sane default
            self._apply_preset()

    def _save_settings(self):
        try:
            os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
            data = {k: var.get() for k, var in self.var.items()}
            with open(_SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ── small widget builders ────────────────────────────────────────────────
    def _label_with_tip(self, parent, row, text, tip):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Label(f, text=text).pack(side="left")
        if tip:
            q = ttk.Label(f, text="(?)", foreground="#0066cc", cursor="question_arrow")
            q.pack(side="left", padx=(3, 0))
            _Tip(q, tip)

    def _path_row(self, parent, row, label, key, default, kind, types=None, tip=None):
        self._label_with_tip(parent, row, label, tip)
        var = tk.StringVar(value=default)
        self.var[key] = var
        ttk.Entry(parent, textvariable=var, width=44).grid(row=row, column=1, sticky="we")
        ttk.Button(parent, text="…", width=3,
                   command=lambda: self._browse(var, kind, types)
                   ).grid(row=row, column=2, padx=(4, 0))
        parent.columnconfigure(1, weight=1)

    def _browse(self, var, kind, types):
        cur = var.get().strip()
        init = self._abs(cur) if cur and os.path.exists(self._abs(cur)) else ROOT
        if kind == "dir":
            p = filedialog.askdirectory(initialdir=init)
        else:
            p = filedialog.askopenfilename(initialdir=init, filetypes=types or [("All", "*.*")])
        if p:
            var.set(p)

    def _slider(self, parent, row, label, key, lo, hi, default, fmt="{:.2f}", tip=None):
        self._label_with_tip(parent, row, label, tip)
        var = tk.DoubleVar(value=default)
        self.var[key] = var
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                  length=150).grid(row=row, column=1, sticky="we")
        vlbl = ttk.Label(parent, width=7, anchor="e")
        vlbl.grid(row=row, column=2, padx=(6, 0))
        var.trace_add("write", lambda *a: vlbl.config(text=fmt.format(var.get())))
        vlbl.config(text=fmt.format(default))
        parent.columnconfigure(1, weight=1)

    def _spin(self, parent, row, label, key, lo, hi, default, inc=1, tip=None):
        self._label_with_tip(parent, row, label, tip)
        var = tk.StringVar(value=str(default))
        self.var[key] = var
        ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=var,
                    width=10).grid(row=row, column=1, sticky="w")

    def _entry(self, parent, row, label, key, default, width=14, tip=None):
        self._label_with_tip(parent, row, label, tip)
        var = tk.StringVar(value=default)
        self.var[key] = var
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="w")

    def _check(self, parent, row, label, key, default, tip=None):
        var = tk.BooleanVar(value=default)
        self.var[key] = var
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Checkbutton(f, text=label, variable=var).pack(side="left")
        if tip:
            q = ttk.Label(f, text="(?)", foreground="#0066cc", cursor="question_arrow")
            q.pack(side="left", padx=(3, 0))
            _Tip(q, tip)

    def _combo(self, parent, row, label, key, values, default, tip=None):
        self._label_with_tip(parent, row, label, tip)
        var = tk.StringVar(value=default)
        self.var[key] = var
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly",
                     width=22).grid(row=row, column=1, sticky="w")

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)
        top.columnconfigure(0, weight=1)

        # Inputs
        inp = ttk.Labelframe(top, text="Project", padding=10)
        inp.grid(row=0, column=0, sticky="ew")
        self._path_row(inp, 0, "Photos folder", "photos", "data/images", "dir",
                       tip="Folder of input photos for Step 1 (building the camera model). "
                           "Leave blank if you already have a model and only want to align.")
        self._path_row(inp, 1, "Reference cloud (.las / .laz / .e57)", "lidar",
                       "data/lidar.laz", "file",
                       [("Point cloud", "*.las *.laz *.e57 *.ply *.pcd"), ("All", "*.*")],
                       tip="The point cloud to align to - usually a LiDAR/survey scan, but any "
                           "registered cloud works. Its coordinate frame becomes the target.")
        self._path_row(inp, 2, "Project folder (output goes here)", "project", "data/sfm", "dir",
                       tip="Working folder. The camera model, aligned result, QA clouds and the "
                           "COLMAP database are all written under here.")
        self._combo(inp, 3, "Camera type", "camera", [c[0] for c in CAMERAS], CAMERAS[0][0],
                    tip="Lens type of your photos. Standard suits most cameras and phones; "
                        "fisheye for very wide or action cams; simple is a quick single-"
                        "distortion fit.")

        # Advanced toggle + frame
        self._adv_btn = ttk.Button(top, text="▸ Advanced settings", command=self._toggle_adv)
        self._adv_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._adv = ttk.Frame(top)
        self._adv.grid(row=2, column=0, sticky="ew")
        self._adv.grid_remove()
        self._build_advanced(self._adv)

        # Action bar
        bar = ttk.Frame(top)
        bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.all_btn = ttk.Button(bar, text="▶ Build + Align (all)", command=self.run_all)
        self.all_btn.pack(side="left")
        self.sfm_btn = ttk.Button(bar, text="1. Build model", command=self.sfm_run)
        self.sfm_btn.pack(side="left", padx=(6, 0))
        self.refine_btn = ttk.Button(bar, text="2. Align to cloud", command=self.refine_run)
        self.refine_btn.pack(side="left", padx=(6, 0))
        self.dedup_btn = ttk.Button(bar, text="Merge scans", command=self.dedup_run)
        self.dedup_btn.pack(side="left", padx=(6, 0))
        _Tip(self.dedup_btn,
             "One-time prep for big multi-station scans (e.g. an RTC360 .e57 that's huge only "
             "because every station is stored separately). Merges them into one cloud at the "
             "Downsample/native spacing (Advanced; defaults to 2 mm), removing redundant overlap "
             "without losing surface detail. Writes a .dedup.laz next to the source and repoints "
             "the Reference cloud at it, so later runs load in seconds instead of re-reading the raw file.")
        self.stop_btn = ttk.Button(bar, text="■ Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        # Status: text label + animated throbber
        self.status_text = ttk.Label(bar, text="idle")
        self.status_text.pack(side="right", padx=(0, 2))
        self.status_throb = ttk.Label(bar, text="")
        self.status_throb.pack(side="right")

        # COLMAP status line
        cb = ttk.Frame(top)
        cb.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._colmap_lbl = ttk.Label(cb, text="")
        self._colmap_lbl.pack(side="left")
        self._colmap_dl_btn = ttk.Button(cb, text="⬇ Download COLMAP",
                                         command=self._colmap_download)
        self._gluemap_btn = ttk.Button(cb, text="⬇ Install GLUEMAP (WSL)",
                                       command=self._gluemap_install)
        self._gluemap_btn.pack(side="right")
        self._update_colmap_banner()

        # Log
        ttk.Label(top, text="Log").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.log = scrolledtext.ScrolledText(top, height=14, width=104,
                                             font=("Consolas", 9), wrap="none")
        self.log.grid(row=6, column=0, sticky="nsew")
        self.log.configure(state="disabled")
        top.rowconfigure(6, weight=1)

    def _build_advanced(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Fine alignment solver
        solv = ttk.Labelframe(parent, text="Fine alignment (solver)", padding=8)
        solv.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        self._slider(solv, 0, "Snap strength", "w_lidar", 1.0, 20.0, 5.0, "{:.1f}",
                     tip="How strongly points are pulled onto the reference cloud's surfaces "
                         "versus keeping the original photo geometry. Higher = conform to the "
                         "cloud more (closer to the scan); lower = trust the photos more.")
        self._slider(solv, 1, "Outlier distance (m)", "huber", 0.01, 0.5, 0.1, "{:.2f}",
                     tip="Matches farther than this from the cloud (metres) are treated as "
                         "outliers and down-weighted, so bad matches don't drag the model. "
                         "Set near your expected noise (~0.1 m).")
        self._slider(solv, 2, "Match distance (m)", "max_assoc_dist", 0.1, 2.0, 0.5, "{:.2f}",
                     tip="Maximum distance (metres) to pair a model point with the cloud. Points "
                         "with no cloud surface within range are skipped. Make it a bit larger "
                         "than the misalignment left after rough align.")
        self._slider(solv, 3, "Flat-surface cutoff", "planarity_min", 0.0, 0.5, 0.1, "{:.2f}",
                     tip="Only matches on locally flat patches of the cloud are trusted. Minimum "
                         "flatness (0 = anything, 1 = perfectly flat). Higher ignores edges, "
                         "foliage and clutter; lower uses more of the cloud.")
        self._spin(solv, 4, "Re-match rounds", "outer_iters", 1, 30, 8,
                   tip="How many times to re-pair points to the cloud and re-solve (ICP-style). "
                       "More rounds settle a rough start; 8 is usually plenty.")
        self._spin(solv, 5, "Solver steps per round", "inner_iters", 10, 200, 50, inc=10,
                   tip="Optimiser iterations within each round. Higher converges harder per "
                       "round; 50 is plenty.")
        self._check(solv, 6, "Gradual tightening", "anneal", True,
                    tip="Start with a loose match distance and gentle pull, then tighten each "
                        "round. Lets a rough alignment settle without diverging. Leave on.")
        self._check(solv, 7, "Lock camera calibration", "fix_intrinsics", True,
                    tip="Keep focal length and lens distortion fixed; only move camera positions "
                        "and 3D points. Leave on unless your calibration is wrong.")

        # Reference cloud + rough alignment
        li = ttk.Labelframe(parent, text="Reference cloud & rough align", padding=8)
        li.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)
        self._check(li, 0, "Rough align first", "prealign", True,
                    tip="Coarsely place the model into the cloud's coordinate frame before fine "
                        "alignment. Needed unless the model is already roughly in the cloud's "
                        "coordinates.")
        self._combo(li, 1, "Rough align method", "prealign_method", ["auto", "global"], "auto",
                    tip="auto: fast; assumes the model is rotated less than ~30 deg from the "
                        "cloud. global: feature-based; handles any rotation but needs structure "
                        "in the cloud.")
        self._entry(li, 2, "Rough align spacing (m)", "prealign_voxel", "0.5",
                    tip="Cloud point spacing (metres) used only during rough alignment. Coarser "
                        "= faster, less precise rough align.")
        self._entry(li, 3, "Downsample cloud (m)", "voxel", "",
                    tip="Keep one cloud point per cube of this size (metres) to run faster. "
                        "Blank = full-resolution cloud. E.g. 0.03 thins to 3 cm spacing.")
        self._entry(li, 4, "Keep-around margin (m)", "crop_margin", "2.0",
                    tip="How far beyond the photographed area (metres) to keep cloud points. "
                        "Larger keeps more surrounding surface (slower); too small can clip "
                        "useful surfaces.")
        self._spin(li, 5, "Points per surface fit", "k_plane", 4, 64, 16,
                   tip="At each spot a small local surface is fit through this many nearest cloud "
                       "points to find its orientation. More = smoother, noise-robust surfaces; "
                       "fewer = follows fine detail.")
        self._entry(li, 6, "Max matches per round", "max_lidar_residuals", "30000",
                    tip="Cap on how many point-to-cloud matches to use each round (spread evenly "
                        "across the scene). Limits time and memory on huge models; a few thousand "
                        "is enough to pin the alignment.")

        # Outputs
        out = ttk.Labelframe(parent, text="Outputs", padding=8)
        out.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=4)
        self._check(out, 0, "Write XMP next to photos", "xmp_next", True,
                    tip="Write the pose sidecar files into the photos folder, where RealityScan "
                        "expects them.")
        self._path_row(out, 1, "…or XMP folder", "xmp_dir", "", "dir",
                       tip="Write the XMP sidecars here instead. If set, this overrides 'next to "
                           "photos'.")
        self._check(out, 2, "Write QA clouds", "write_qa", True,
                    tip="Save before/after point clouds coloured by distance to the reference "
                        "cloud, so you can confirm the alignment improved (open in CloudCompare).")
        self._combo(out, 3, "RealityScan pose mode", "xmp_pose_prior",
                    ["locked", "initial", "exact"], "locked",
                    tip="How RealityScan treats the exported poses: locked = use as-is, don't "
                        "change; initial = starting guess it may refine; exact = treat as exact "
                        "measurements.")
        self._combo(out, 4, "Axis convention", "xmp_axis_flip",
                    ["", "rc_default", "identity", "flip_xz", "flip_xy"], "",
                    tip="Coordinate-axis convention for RealityScan. Leave blank unless imported "
                        "cameras look mirrored or upside-down, then try the presets.")
        self._path_row(out, 5, "Existing model (skip Step 1)", "model_override", "", "dir",
                       tip="Point straight at an existing COLMAP/GLOMAP/GLUEMAP model folder (sparse/0) "
                           "to skip building one. Perfect for using models pre-built with GLUEMAP. "
                           "Blank = use the model Step 1 makes.")

        # Photo alignment (SfM)
        sf = ttk.Labelframe(parent, text="Photo alignment (SfM, Step 1)", padding=8)
        sf.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=4)
        self._combo(sf, 0, "SfM Engine", "sfm_engine", ["COLMAP/GLOMAP", "GLUEMAP"], "COLMAP/GLOMAP",
                    tip="COLMAP/GLOMAP: standard global mapper using SIFT features; runs on "
                        "Windows. GLUEMAP: deep-learning hybrid mapper (Linux + CUDA) - install "
                        "gluemap-demo natively or inside WSL2 and the app runs it via 'wsl'. "
                        "GLUEMAP ignores the SIFT quality/keypoint options below.")
        self._preset_row(sf, 1)
        self._combo(sf, 2, "Feature quality", "sfm_quality", ["high", "fast"], "fast",
                    tip="fast: GPU SIFT, much faster - the sane default. high: affine-shape + "
                        "DSP-SIFT, more matches but CPU-only and very slow on many photos; only "
                        "for genuinely hard / low-texture sets. (COLMAP/GLOMAP only)")
        self._spin(sf, 3, "Keypoints per photo", "sfm_max_feats", 1024, 32768, 4096, inc=1024,
                   tip="Features per photo. More = denser matches + bigger database + more RAM "
                       "and time in the mapper. 2048-4096 is plenty for high-overlap video; 8192 "
                       "only for sparse / low-texture sets. (COLMAP/GLOMAP only)")
        self._spin(sf, 4, "Frames matched ahead", "sfm_overlap", 1, 50, 10,
                   tip="For ordered photos/video: how many neighbouring frames each photo is "
                       "matched against. Higher is more robust to fast motion; slower. (COLMAP/GLOMAP only)")
        self._path_row(sf, 5, "Loop-closure file (optional)", "sfm_vocab", "", "file",
                       [("Vocab tree", "*.bin"), ("All", "*.*")],
                       tip="Optional COLMAP vocabulary-tree .bin file that lets it recognise when "
                           "the camera revisits the same place (loop closure). (COLMAP/GLOMAP only)")
        self._path_row(sf, 6, "GLUEMAP config (optional)", "sfm_gluemap_config", "", "file",
                       [("YAML config", "*.yaml *.yml"), ("All", "*.*")],
                       tip="Optional YAML configuration file for GLUEMAP (e.g. configs/example.yaml). "
                           "If blank, GLUEMAP uses its default settings.")

    def _preset_row(self, parent, row):
        self._label_with_tip(
            parent, row, "Speed / quality preset",
            "A starting point for the SfM knobs below. Fast = small database + low RAM; "
            "Balanced = the sane default; High detail = 8192 features; Max quality = affine/DSP "
            "on CPU (very slow). Tweak the knobs after picking, or choose Custom to leave them.")
        var = tk.StringVar(value="Balanced")
        self.var["sfm_preset"] = var
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", width=22,
                          values=_SFM_PRESET_NAMES)
        cb.grid(row=row, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

    def _apply_preset(self):
        p = _SFM_PRESETS.get(self.var["sfm_preset"].get())
        if not p:            # "Custom" -> leave the knobs untouched
            return
        q, feats, ov = p
        self.var["sfm_quality"].set(q)
        self.var["sfm_max_feats"].set(str(feats))
        self.var["sfm_overlap"].set(str(ov))

    def _toggle_adv(self):
        self._adv_open = not self._adv_open
        if self._adv_open:
            self._adv.grid()
            self._adv_btn.config(text="▾ Advanced settings")
        else:
            self._adv.grid_remove()
            self._adv_btn.config(text="▸ Advanced settings")

    # ── derived paths ──────────────────────────────────────────────────────────
    def _abs(self, p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    @staticmethod
    def _has_model(d):
        return any(os.path.isfile(os.path.join(d, f"points3D.{e}")) for e in ("bin", "txt"))

    def _find_model_dir(self, proj, override=""):
        """Locate the COLMAP model folder. COLMAP/GLOMAP write <proj>/sparse/0; GLUEMAP's
        layout under --write_path isn't fixed, so fall back to a shallow search for the
        folder that actually holds points3D.bin/.txt."""
        if override:
            return self._abs(override)
        for c in (os.path.join(proj, "sparse", "0"), os.path.join(proj, "sparse"), proj):
            if self._has_model(c):
                return c
        if os.path.isdir(proj):
            for d, _, files in os.walk(proj):
                if "points3D.bin" in files or "points3D.txt" in files:
                    return d
        return os.path.join(proj, "sparse", "0")   # default, for the not-found message

    def _paths(self):
        proj = self._abs(self.var["project"].get().strip() or "data/sfm")
        override = self.var["model_override"].get().strip()
        return {
            "project": proj,
            "db": os.path.join(proj, "database.db"),
            "sparse": os.path.join(proj, "sparse"),
            "sparse_in": self._find_model_dir(proj, override),
            "sparse_out": os.path.join(proj, "sparse_refined"),
            "qa": os.path.join(proj, "qa"),
        }

    # ── COLMAP banner / download ────────────────────────────────────────────────
    def _update_colmap_banner(self):
        self._colmap_exe = _find_colmap()
        if self._colmap_exe:
            self._colmap_lbl.config(text=f"COLMAP: {self._colmap_exe}", foreground="#1a7a1a")
            self._colmap_dl_btn.pack_forget()
        else:
            self._colmap_lbl.config(text="COLMAP not found (needed for Step 1).",
                                    foreground="#b00000")
            self._colmap_dl_btn.pack(side="left", padx=(10, 0))

    def _install_wsl(self):
        """Run `wsl --install` elevated (UAC). Needs a reboot to finish."""
        if sys.platform != "win32":
            return messagebox.showerror("Unsupported", "WSL is Windows-only.")
        try:
            import ctypes
            params = ('-NoProfile -ExecutionPolicy Bypass -Command '
                      '"wsl --install; Read-Host \'WSL install finished - press Enter to close\'"')
            rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", "powershell.exe",
                                                     params, None, 1)
            if rc <= 32:
                return messagebox.showerror(
                    "Elevation declined",
                    "Couldn't start the elevated installer (UAC declined). Run 'wsl --install' "
                    "in an admin PowerShell yourself.")
            messagebox.showinfo(
                "Installing WSL",
                "WSL is installing in the elevated window. When it finishes:\n"
                "1. Reboot Windows.\n"
                "2. Open Ubuntu once and set a username/password.\n"
                "3. Click 'Install GLUEMAP (WSL)' again.")
        except Exception as e:
            messagebox.showerror("WSL install failed", str(e))

    def _gluemap_install(self):
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        if not _wsl_ready():
            if messagebox.askokcancel(
                    "Install WSL",
                    "WSL isn't installed. I can run 'wsl --install' now - it needs admin (a UAC "
                    "prompt) and a reboot to finish.\n\nAfter rebooting, open Ubuntu once to set a "
                    "username, then click Install GLUEMAP again. (GLUEMAP also needs an NVIDIA "
                    "CUDA GPU.)\n\nInstall WSL now?"):
                self._install_wsl()
            return
        if not messagebox.askokcancel(
                "Install GLUEMAP",
                "This builds GLUEMAP inside WSL and downloads a CUDA PyTorch build plus several "
                "GB of model weights. It takes a while and needs an NVIDIA GPU.\n\nProceed?"):
            return
        self._cancel.clear()
        self._clear_log()
        self.all_btn.config(state="disabled")
        self.sfm_btn.config(state="disabled")
        self.refine_btn.config(state="disabled")
        self._gluemap_btn.config(state="disabled", text="Installing GLUEMAP…")
        self.status_text.config(text="installing GLUEMAP…")
        self._worker = threading.Thread(target=self._gluemap_install_worker, daemon=True)
        self._worker.start()

    def _gluemap_install_worker(self):
        try:
            script = _resource("scripts/install_gluemap.sh")
            if not os.path.isfile(script):
                self.q.put(("log", f"installer not found: {script}\n"))
                return self.q.put(("done", 1))
            wsl_script = _winpath_to_wsl(script)
            # strip CR (Windows checkout) and run under a WSL login shell
            cmd = ["wsl", "bash", "-lc", 'tr -d "\\r" < "$1" | bash -ls', "_", wsl_script]
            self.q.put(("log", f"$ wsl bash {wsl_script}\n\n"))
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, creationflags=flags)
            self._proc = proc
            for line in proc.stdout:
                self.q.put(("log", line))
                if self._cancel.is_set():
                    _kill_tree(proc)
                    break
            proc.wait()
            self._proc = None
            self.q.put(("done", proc.returncode))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("done", 1))

    def _colmap_download(self):
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        self._cancel.clear()
        self._colmap_dl_btn.config(state="disabled", text="Downloading…")
        self.all_btn.config(state="disabled")
        self.sfm_btn.config(state="disabled")
        self.refine_btn.config(state="disabled")
        self._clear_log()
        self._worker = threading.Thread(target=self._colmap_dl_worker, daemon=True)
        self._worker.start()

    def _colmap_dl_worker(self):
        try:
            with contextlib.redirect_stdout(_QueueWriter(self.q)):
                print("Looking up the latest COLMAP release…")
                url, fname = _latest_colmap_windows_url()
                print(f"Downloading {fname}\n  {url}")
                os.makedirs(_COLMAP_LOCAL_DIR, exist_ok=True)
                zip_path = os.path.join(_COLMAP_LOCAL_DIR, fname)
                req = urllib.request.Request(url, headers={"User-Agent": "lidar-align-gui"})
                with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp, \
                        open(zip_path, "wb") as out:
                    total = int(resp.headers.get("Content-Length", 0))
                    done = last = 0
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total and done * 100 // total != last:
                            last = done * 100 // total
                            self.q.put(("log", f"\r  {last}%  "))
                print("\nExtracting…")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(_COLMAP_LOCAL_DIR)
                os.remove(zip_path)
                found = None
                for d, _, files in os.walk(_COLMAP_LOCAL_DIR):
                    if "colmap.exe" in files:
                        found = os.path.join(d, "colmap.exe"); break
                if found:
                    print(f"COLMAP installed -> {found}")
                    self.q.put(("colmap_installed", found))
                else:
                    print("ERROR: colmap.exe not found after extraction.")
                    self.q.put(("done", 1))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("done", 1))

    # ── Step 1: SfM ─────────────────────────────────────────────────────────────
    def sfm_run(self):
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        photos = self.var["photos"].get().strip()
        if not photos:
            return messagebox.showerror("Missing input", "Photos folder is required for Step 1.")
        photos = self._abs(photos)
        if not os.path.isdir(photos):
            return messagebox.showerror("Not found", f"Photos folder not found:\n{photos}")

        engine = self.var["sfm_engine"].get()
        p = self._paths()

        if engine == "GLUEMAP":
            mode, gluemap_exe = _find_gluemap()
            if not gluemap_exe:
                return messagebox.showerror(
                    "GLUEMAP not found",
                    "gluemap-demo was not found on PATH or inside WSL.\n\n"
                    "GLUEMAP needs Linux + a CUDA GPU. On Windows, install it inside WSL2 "
                    "(with the NVIDIA CUDA driver); the app will run it as 'wsl gluemap-demo'.")
            cfg_raw = self.var["sfm_gluemap_config"].get().strip()
            # Validate the optional config early so we don't launch with a bad path.
            if cfg_raw and not os.path.isfile(self._abs(cfg_raw)):
                return messagebox.showerror(
                    "Bad config", f"GLUEMAP config file not found:\n{cfg_raw}")
            args = dict(
                engine="GLUEMAP", mode=mode, gluemap=gluemap_exe,
                images=photos, sparse=p["sparse"], project=p["project"],
                config=self._abs(cfg_raw) if cfg_raw else "",
            )
            self._start(f"running SfM (GLUEMAP via {mode})…", self._sfm_worker, (args,))
            return

        self._update_colmap_banner()
        if not self._colmap_exe:
            return messagebox.showerror(
                "COLMAP not found",
                "Step 1 needs COLMAP. Click 'Download COLMAP' or install it and add to PATH.")

        try:    # Spinbox text is freely editable; don't pass a bad token to COLMAP
            max_feats = str(int(self.var["sfm_max_feats"].get().strip() or "4096"))
            overlap = str(int(self.var["sfm_overlap"].get().strip() or "10"))
        except ValueError:
            return messagebox.showerror(
                "Invalid input", "Max keypoints and Match overlap must be whole numbers.")
        vocab_raw = self.var["sfm_vocab"].get().strip()
        args = dict(
            engine="COLMAP/GLOMAP",
            images=photos, db=p["db"], sparse=p["sparse"],
            camera=_CAMERA_MODEL[self.var["camera"].get()],
            max_feats=max_feats, overlap=overlap,
            quality=self.var["sfm_quality"].get(),
            vocab=self._abs(vocab_raw) if vocab_raw else "",
            colmap=self._colmap_exe,
        )
        self._start("running SfM (COLMAP + GLOMAP)…", self._sfm_worker, (args,))

    def _sfm_worker(self, a):
        def run(cmd, label):
            self.q.put(("log", f"\n== {label} ==\n"))
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, creationflags=flags)
            self._proc = proc
            for line in proc.stdout:
                self.q.put(("log", line))
                if self._cancel.is_set():
                    _kill_tree(proc)        # kills colmap + children, releases the db lock
                    proc.wait()
                    self._proc = None
                    return False
            proc.wait()
            self._proc = None
            if proc.returncode != 0:
                self.q.put(("log", f"\nERROR: {label} exited {proc.returncode}\n"))
                return False
            return True

        try:
            if a.get("engine") == "GLUEMAP":
                os.makedirs(a["project"], exist_ok=True)   # gluemap won't create --write_path
                if a.get("mode") == "wsl":
                    # run inside WSL via a login shell (so the ~/.local/bin wrapper is on PATH);
                    # translate Windows paths to /mnt/... . gluemap writes to the same on-disk
                    # folder, so Step 2 reads the model back on the Windows side.
                    inner = ('gluemap-demo --images_path "$1" --write_path "$2" '
                             '--intrinsics_mode SHARED')
                    sh_args = ["_", _winpath_to_wsl(a["images"]), _winpath_to_wsl(a["project"])]
                    if a.get("config"):
                        inner += ' --config "$3"'
                        sh_args.append(_winpath_to_wsl(a["config"]))
                    cmd = ["wsl", "bash", "-lc", inner] + sh_args
                else:
                    cmd = [a["gluemap"], "--images_path", a["images"],
                           "--write_path", a["project"], "--intrinsics_mode", "SHARED"]
                    if a.get("config"):
                        cmd += ["--config", a["config"]]
                if not run(cmd, f"gluemap ({a.get('mode')})"):
                    self.q.put(("log", "\nGLUEMAP run failed - see above for details.\n"))
                    return self.q.put(("done", 1))
                model = self._find_model_dir(a["project"])   # GLUEMAP layout isn't fixed
                if not self._has_model(model):
                    self.q.put(("log", "\nGLUEMAP exited 0 but no COLMAP model (points3D.bin/"
                                       ".txt) was found under the project folder.\n"))
                    return self.q.put(("done", 1))
                self.q.put(("log", f"\nSfM complete -> {model}\n"))
                return self.q.put(("done", 0))

            os.makedirs(a["sparse"], exist_ok=True)

            # COLMAP 4.0 renamed some option namespaces (e.g. SiftExtraction.use_gpu ->
            # FeatureExtraction.use_gpu). Resolve names against the installed binary's help.
            fe = _colmap_help(a["colmap"], "feature_extractor")
            ext_gpu = _resolve(fe, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
            feat = [a["colmap"], "feature_extractor",
                    "--database_path", a["db"], "--image_path", a["images"],
                    "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", a["camera"],
                    "--SiftExtraction.max_num_features", a["max_feats"]]
            if a["quality"] == "high":
                # affine-shape + DSP improve matches on blurry frames but COLMAP runs them
                # CPU-only; set use_gpu 0 so the (slow) CPU path is explicit, not a surprise.
                feat += ["--SiftExtraction.estimate_affine_shape", "1",
                         "--SiftExtraction.domain_size_pooling", "1",
                         f"--{ext_gpu}", "0"]
                self.q.put(("log", "[feature quality HIGH: CPU SIFT - slow on many photos]\n"))
            else:
                feat += [f"--{ext_gpu}", "1"]
            if not run(feat, "feature_extractor"):
                return self.q.put(("done", 1))

            fm = _colmap_help(a["colmap"], "sequential_matcher")
            mat_gpu = _resolve(fm, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
            overlap_opt = _resolve(fm, "SequentialMatching.overlap", "SequentialPairing.overlap")
            quad_opt = _present(fm, "SequentialMatching.quadratic_overlap",
                                "SequentialPairing.quadratic_overlap")
            loop_opt = _present(fm, "SequentialMatching.loop_detection",
                                "SequentialPairing.loop_detection")
            vocab_opt = _present(fm, "SequentialMatching.vocab_tree_path",
                                 "SequentialPairing.vocab_tree_path")
            # loop closure needs a real vocab-tree file and a supported option
            use_vocab = bool(a["vocab"]) and os.path.isfile(a["vocab"]) and bool(vocab_opt)
            match = [a["colmap"], "sequential_matcher", "--database_path", a["db"],
                     f"--{overlap_opt}", a["overlap"], f"--{mat_gpu}", "1"]
            if quad_opt:
                match += [f"--{quad_opt}", "1"]
            if loop_opt:
                match += [f"--{loop_opt}", "1" if use_vocab else "0"]
            if use_vocab:
                match += [f"--{vocab_opt}", a["vocab"]]
            if not run(match, "sequential_matcher"):
                return self.q.put(("done", 1))

            # Estimate focal priors from the view graph. Video frames usually have no EXIF
            # focal, so COLMAP's default guess is off and global_mapper rejects many pairs with
            # "relative relation errors". view_graph_calibrator fixes the focals first (the
            # GLOMAP default). Non-fatal: older COLMAP may lack it.
            if not run([a["colmap"], "view_graph_calibrator", "--database_path", a["db"]],
                       "view_graph_calibrator (focal priors)"):
                if self._cancel.is_set():
                    return self.q.put(("done", 1))
                self.q.put(("log", "view_graph_calibrator unavailable/failed - continuing; if "
                                   "global_mapper rejects many pairs, focal priors are the cause.\n"))

            mapper = [a["colmap"], "global_mapper", "--database_path", a["db"],
                      "--image_path", a["images"], "--output_path", a["sparse"]]
            if not run(mapper, "global_mapper (GLOMAP)"):
                return self.q.put(("done", 1))

            self.q.put(("log", f"\nSfM complete -> {os.path.join(a['sparse'], '0')}\n"))
            self.q.put(("done", 0))
        except FileNotFoundError:
            exe = a.get("colmap") or a.get("gluemap") or "the SfM tool"
            self.q.put(("log", f"\nERROR: could not launch '{exe}'.\n"))
            self.q.put(("done", 1))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("done", 1))

    # ── Step 2: refine ──────────────────────────────────────────────────────────
    def _refine_kwargs(self):
        v = self.var
        p = self._paths()
        lidar = v["lidar"].get().strip()
        if not lidar:
            raise ValueError("LiDAR cloud file is required.")
        photos = v["photos"].get().strip()
        voxel = v["voxel"].get().strip()
        axis = v["xmp_axis_flip"].get().strip()
        # an explicit XMP folder always wins; otherwise default to next-to-photos
        xmp_dir = v["xmp_dir"].get().strip()
        if xmp_dir:
            xmp_out = self._abs(xmp_dir)
        elif bool(v["xmp_next"].get()) and photos:
            xmp_out = self._abs(photos)
        else:
            xmp_out = None
        kw = dict(
            sparse_in=p["sparse_in"], lidar=self._abs(lidar), sparse_out=p["sparse_out"],
            prealign=bool(v["prealign"].get()),
            prealign_method=v["prealign_method"].get(),
            prealign_voxel=float(v["prealign_voxel"].get() or 0.5),
            voxel=(float(voxel) if voxel else None),
            crop_margin=float(v["crop_margin"].get() or 2.0),
            k_plane=int(float(v["k_plane"].get() or 16)),
            w_lidar=float(v["w_lidar"].get()),
            huber=float(v["huber"].get()),
            outer_iters=int(float(v["outer_iters"].get() or 8)),
            inner_iters=int(float(v["inner_iters"].get() or 50)),
            max_assoc_dist=float(v["max_assoc_dist"].get()),
            planarity_min=float(v["planarity_min"].get()),
            max_lidar_residuals=int(float(v["max_lidar_residuals"].get() or 30000)),
            anneal=bool(v["anneal"].get()),
            fix_intrinsics=bool(v["fix_intrinsics"].get()),
            qa_out=(p["qa"] if bool(v["write_qa"].get()) else None),
            xmp_out=xmp_out,
            xmp_pose_prior=v["xmp_pose_prior"].get(),
            xmp_axis_flip=(axis or None),
        )
        return kw, (self._abs(photos) if photos else None)

    def run_all(self):
        """Build the model, then automatically align to the cloud - one unattended run."""
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        try:
            kw, images_dir = self._refine_kwargs()       # validate align inputs up front
        except (ValueError, tk.TclError) as e:
            return messagebox.showerror("Invalid input", str(e))
        if not os.path.isfile(kw["lidar"]):
            return messagebox.showerror("Not found", f"LiDAR file not found:\n{kw['lidar']}")
        # arm the chain, then run Build; _on_done starts Align automatically if Build succeeds
        self._chain = (kw, images_dir, self._paths()["project"])
        self.sfm_run()
        if not (self._worker and self._worker.is_alive()):
            self._chain = None       # Build didn't start (bad input) -> don't chain

    def refine_run(self):
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        try:
            kw, images_dir = self._refine_kwargs()
        except (ValueError, tk.TclError) as e:
            return messagebox.showerror("Invalid input", str(e))
        if not os.path.isdir(kw["sparse_in"]):
            return messagebox.showerror(
                "Not found",
                f"No camera model at:\n{kw['sparse_in']}\n\nRun Step 1 first, or set "
                f"'Existing model' in Advanced.")
        if not os.path.isfile(kw["lidar"]):
            return messagebox.showerror("Not found", f"LiDAR file not found:\n{kw['lidar']}")
        self._start("running alignment (Stop cancels after the current round)…",
                    self._refine_worker, (kw, images_dir))

    def _refine_worker(self, kwargs, images_dir):
        kwargs["cancel_cb"] = self._cancel.is_set
        try:
            from lidar_align.refine import refine
            with contextlib.redirect_stdout(_QueueWriter(self.q)):
                refine(**kwargs)
                if images_dir and kwargs.get("xmp_out"):
                    _validate_images_dir(images_dir, kwargs["sparse_out"])
            self.q.put(("done", 0))
        except KeyboardInterrupt:                          # cooperative Stop during cloud load
            self.q.put(("log", "\n[cancelled]\n"))
            self.q.put(("done", 1))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("done", 1))

    # ── Merge / dedup multi-station cloud ────────────────────────────────────────
    def dedup_run(self):
        if self._worker and self._worker.is_alive():
            return messagebox.showwarning("Busy", "A job is already running.")
        lidar = self.var["lidar"].get().strip()
        if not lidar:
            return messagebox.showerror("Invalid input", "Set the Reference cloud first.")
        src = self._abs(lidar)
        if not os.path.isfile(src):
            return messagebox.showerror("Not found", f"Cloud file not found:\n{src}")
        voxel = self.var["voxel"].get().strip()
        try:
            vx = float(voxel) if voxel else 0.002          # default native ~2 mm spacing
        except ValueError:
            return messagebox.showerror("Invalid input", f"Downsample (m) is not a number: {voxel!r}")
        self._start(f"merging scan stations in {os.path.basename(src)} at {vx:g} m "
                    f"(removes station overlap, keeps detail; Stop is safe)…",
                    self._dedup_worker, (src, vx))

    def _dedup_worker(self, src, vx):
        try:
            from lidar_align.lidar_index import dedup_to_laz
            with contextlib.redirect_stdout(_QueueWriter(self.q)):
                dst, _ = dedup_to_laz(src, voxel=vx, cancel_cb=self._cancel.is_set)
            self.q.put(("dedup_done", dst))
            self.q.put(("done", 0))
        except KeyboardInterrupt:
            self.q.put(("log", "\n[cancelled - no file written]\n"))
            self.q.put(("done", 1))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("done", 1))

    # ── throbber animation ──────────────────────────────────────────────────────────
    def _throb_tick(self):
        """Spin the throbber and show elapsed time. Only when output has actually gone silent
        for a while does it heartbeat the log, so the window still looks alive on quiet steps
        without spamming while a step is printing."""
        if self._worker and self._worker.is_alive():
            now = time.time()
            if self._job_start is None:
                self._job_start = now
                self._last_beat = now
                self._last_output = now
            self._throb_state = (self._throb_state + 1) % 4
            el = int(now - self._job_start)
            self.status_throb.config(text="⠋⠙⠹⠸"[self._throb_state])
            state = "stopping" if self._cancel.is_set() else "running"
            self.status_text.config(text=f"{state}  {el // 60:d}:{el % 60:02d}")
            quiet = now - self._last_output
            if quiet >= 60 and now - self._last_beat >= 60:   # only when genuinely silent
                self._last_beat = now
                self._append(f"[still running — {el // 60}m{el % 60:02d}s elapsed, "
                             f"no output for {int(quiet)}s]\n")
        else:
            self._job_start = None
            self.status_throb.config(text="")
        self.root.after(100, self._throb_tick)

    # ── job lifecycle ────────────────────────────────────────────────────────────
    def _start(self, msg, target, args):
        self._cancel.clear()
        self._clear_log()
        self._append(msg + "\n\n")
        self.all_btn.config(state="disabled")
        self.sfm_btn.config(state="disabled")
        self.refine_btn.config(state="disabled")
        self.dedup_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_text.config(text="running…")
        self._worker = threading.Thread(target=target, args=args, daemon=True)
        self._worker.start()

    def stop(self):
        if self._worker and self._worker.is_alive():
            self._cancel.set()
            _kill_tree(self._proc)   # SfM runs as a subprocess: kill it now (don't wait on a blocked read)
            # The align/merge run in-process and can't be hard-killed mid-solve; they stop at the
            # next safe point (between scans while loading, between rounds while solving). The
            # throbber keeps spinning and the status reads "stopping" so it doesn't look frozen.
            self._append("\n[stopping — SfM stops now; align/merge finish the current step first]\n")
            self.stop_btn.config(state="disabled")

    def _drain(self):
        logs = []                                          # batch a tick's log lines -> one insert
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._last_output = time.time()
                    logs.append(payload)
                    continue
                if logs:                                   # flush buffered logs before a state event
                    self._append("".join(logs)); logs = []
                if kind == "dedup_done":
                    self.var["lidar"].set(payload)         # repoint the align at the merged copy
                    self._append(f"\nReference cloud set to merged copy:\n{payload}\n")
                elif kind == "colmap_installed":
                    self._colmap_exe = payload
                    self._update_colmap_banner()
                    self._colmap_dl_btn.config(state="normal", text="⬇ Download COLMAP")
                    self.all_btn.config(state="normal")
                    self.sfm_btn.config(state="normal")
                    self.refine_btn.config(state="normal")
                    self.dedup_btn.config(state="normal")
                    self._append("\nCOLMAP ready.\n")
                    self._worker = None
                else:
                    self._on_done(payload)
        except queue.Empty:
            pass
        if logs:                                           # flush the tail of this tick's logs
            self._append("".join(logs))
        self.root.after(100, self._drain)

    def _on_done(self, code):
        # Build + Align chain: after Build succeeds, auto-start Align (unattended full run).
        if code == 0 and self._chain is not None:
            kw, images_dir, project = self._chain
            self._chain = None
            model = self._find_model_dir(project)
            if self._has_model(model):
                kw["sparse_in"] = model
                self._append("\n=== Build done — starting Align to cloud ===\n")
                self._cancel.clear()
                self.all_btn.config(state="disabled")
                self.sfm_btn.config(state="disabled")
                self.refine_btn.config(state="disabled")
                self.dedup_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
                self.status_text.config(text="running…")
                self._worker = threading.Thread(target=self._refine_worker,
                                                args=(kw, images_dir), daemon=True)
                self._worker.start()
                return
            self._append("\nBuild finished but no camera model was found - skipping align.\n")
        self._chain = None
        tag = "cancelled" if self._cancel.is_set() else ("done" if code == 0 else f"error ({code})")
        self._append(f"\n─── finished ({tag}) ───\n")
        self.status_text.config(text=tag)
        self.status_throb.config(text="")
        self.all_btn.config(state="normal")
        self.sfm_btn.config(state="normal")
        self.refine_btn.config(state="normal")
        self.dedup_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._colmap_dl_btn.config(state="normal", text="⬇ Download COLMAP")
        self._gluemap_btn.config(state="normal", text="⬇ Install GLUEMAP (WSL)")
        self._worker = None

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Keep only the last _LOG_MAX_LINES so insert/scroll stay cheap no matter how long the
        # run logs for. The full COLMAP output still streams; only the on-screen tail is bounded.
        end_line = int(self.log.index("end-1c").split(".")[0])
        if end_line > _LOG_MAX_LINES:
            self.log.delete("1.0", f"{end_line - _LOG_MAX_LINES + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def on_close(self):
        self._cancel.set()
        _kill_tree(self._proc)   # don't leave colmap.exe orphaned when the window closes
        self._save_settings()
        self.root.destroy()


# ── frozen-bundle self-test ───────────────────────────────────────────────────
def _selftest():
    import tempfile
    import numpy as np
    import pycolmap
    import laspy
    from pycolmap import Rotation3d, Sim3d
    from lidar_align.refine import refine

    logf = os.path.join(tempfile.gettempdir(), "lidar_align_selftest.log")

    def _mark(msg):
        try:
            open(logf, "w").write(msg)
        except OSError:
            pass

    work = tempfile.mkdtemp(prefix="lidar_selftest_")
    try:
        o = pycolmap.SyntheticDatasetOptions()
        o.num_rigs = 1; o.num_cameras_per_rig = 1
        o.num_frames_per_rig = 6; o.num_points3D = 120; o.track_length = 6
        rec = pycolmap.synthesize_dataset(o)
        P = np.array([rec.points3D[p].xyz for p in rec.points3D], float)
        rng = np.random.default_rng(0); step = 0.01
        gu, gv = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
        grid = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)
        patches = []
        for Xi in P:
            n = rng.standard_normal(3); n /= np.linalg.norm(n)
            a = np.cross(n, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
            patches.append(Xi + (grid[:, 0:1] * a + grid[:, 1:2] * b) * step)
        lid = np.vstack(patches)
        h = laspy.LasHeader(point_format=3, version="1.2")
        h.scales = [1e-4] * 3; h.offsets = lid.min(0)
        las = laspy.LasData(h); las.x, las.y, las.z = lid[:, 0], lid[:, 1], lid[:, 2]
        las.write(os.path.join(work, "l.las"))
        ang = 0.01
        rec.transform(Sim3d(1.01, Rotation3d(np.array([0, 0, np.sin(ang / 2), np.cos(ang / 2)])),
                            np.array([0.03, -0.02, 0.03])))
        si = os.path.join(work, "si"); os.makedirs(si); rec.write(si)
        refine(sparse_in=si, lidar=os.path.join(work, "l.las"),
               sparse_out=os.path.join(work, "so"), prealign=False,
               outer_iters=4, inner_iters=20)
        assert os.path.exists(os.path.join(work, "so", "points3D.bin")), "no refined model"
        print("SELFTEST OK"); _mark("OK"); return 0
    except Exception:
        tb = traceback.format_exc(); traceback.print_exc(); _mark("FAIL\n" + tb); return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
