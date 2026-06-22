"""Interactive manual alignment for the case auto-align can't crack: repetitive structure (stairs,
balusters, repeated arches) + partial overlap, where FPFH locks a different wrong scale every run.

You drive it visually - bump the MODEL's scale (and nudge its position/rotation) until the orange
model sits on the gray scan, then accept. That writes the chosen similarity (scale + rotation +
translation) to a JSON the app reads as the pre-align; the LiDAR refine then polishes from there.

Runs as its OWN process (Open3D's window owns the event loop, which can't share Tk's). Uses the
legacy VisualizerWithKeyCallback - a small, stable API that bundles reliably under PyInstaller -
rather than the newer gui module. Controls are printed to the window title and the console.

CLI (the frozen app dispatches this):
    lidar-align.exe --align-tool <scan.las|.e57|.laz> <model.ply> <out.json>
"""
from __future__ import annotations
import json
import sys
import numpy as np


def _euler(yaw, pitch, roll):
    """Z(yaw) Y(pitch) X(roll) rotation, degrees -> 3x3."""
    cz, sz = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cy, sy = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cx, sx = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0, 0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0, 1.0, 0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def _extent_scale(model, scan):
    ms = np.percentile(model, 98, 0) - np.percentile(model, 2, 0)
    ss = np.percentile(scan, 98, 0) - np.percentile(scan, 2, 0)
    return float(np.median(np.maximum(ss, 1e-9) / np.maximum(ms, 1e-9)))


def manual_align(scan_pts, model_pts, out_json, init_scale=None):
    """Open the interactive window. On accept, write {scale, R, t} (SfM-model -> scan Sim3) to
    out_json and return it; on cancel return None. The model is shown as s*R*(X-cm)+cs+t so the
    initial view already centres the model on the scan; the equivalent global Sim3 written out is
    scale=s, rotation=R, translation = cs + t - s*R*cm."""
    import open3d as o3d
    scan = np.ascontiguousarray(scan_pts, np.float64)
    model0 = np.ascontiguousarray(model_pts, np.float64)
    cm = model0.mean(0)
    cs = scan.mean(0)
    diag = float(np.linalg.norm(np.percentile(scan, 98, 0) - np.percentile(scan, 2, 0))) or 1.0
    st = {"s": float(init_scale or _extent_scale(model0, scan)),
          "t": np.zeros(3), "e": np.zeros(3), "ok": False}

    sc = o3d.geometry.PointCloud()
    sc.points = o3d.utility.Vector3dVector(scan)
    sc.paint_uniform_color([0.55, 0.55, 0.55])
    mo = o3d.geometry.PointCloud()
    mo.paint_uniform_color([1.0, 0.5, 0.0])

    def render():
        r = _euler(*st["e"])
        mo.points = o3d.utility.Vector3dVector(st["s"] * ((model0 - cm) @ r.T) + cs + st["t"])

    render()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Match ORANGE model to GRAY scan:  +/- scale  WASD/RF move  QE/ZX rotate"
                      "  ENTER accept  ESC cancel", 1360, 860)
    vis.add_geometry(sc)
    vis.add_geometry(mo)

    def _bump(key, fn):
        def cb(v):
            fn()
            render()
            v.update_geometry(mo)
            return False
        vis.register_key_callback(ord(key), cb)

    mv = diag * 0.01                                   # 1% of scan size per move/rotate keypress
    _bump("=", lambda: st.update(s=st["s"] * 1.02));  _bump("+", lambda: st.update(s=st["s"] * 1.02))
    _bump("-", lambda: st.update(s=st["s"] / 1.02));  _bump("_", lambda: st.update(s=st["s"] / 1.02))
    _bump("D", lambda: st["t"].__setitem__(0, st["t"][0] + mv))
    _bump("A", lambda: st["t"].__setitem__(0, st["t"][0] - mv))
    _bump("W", lambda: st["t"].__setitem__(1, st["t"][1] + mv))
    _bump("S", lambda: st["t"].__setitem__(1, st["t"][1] - mv))
    _bump("R", lambda: st["t"].__setitem__(2, st["t"][2] + mv))
    _bump("F", lambda: st["t"].__setitem__(2, st["t"][2] - mv))
    _bump("Q", lambda: st["e"].__setitem__(0, st["e"][0] + 1.0))   # yaw
    _bump("E", lambda: st["e"].__setitem__(0, st["e"][0] - 1.0))
    _bump("Z", lambda: st["e"].__setitem__(1, st["e"][1] + 1.0))   # pitch
    _bump("X", lambda: st["e"].__setitem__(1, st["e"][1] - 1.0))

    def accept(v):
        st["ok"] = True
        v.close()
        return False
    vis.register_key_callback(257, accept)             # ENTER (GLFW key code)
    vis.register_key_callback(256, lambda v: (v.close(), False)[1])   # ESC -> cancel

    vis.run()
    vis.destroy_window()
    if not st["ok"]:
        return None
    r = _euler(*st["e"])
    sim = {"scale": float(st["s"]), "R": r.tolist(), "t": (cs + st["t"] - st["s"] * (r @ cm)).tolist()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sim, f, indent=2)
    return sim


def cli_main(argv):
    """`--align-tool <scan> <model.ply> <out.json>`: load a downsampled scan + the model PLY, run
    the window, write the Sim3 JSON. Returns 0 on accept, 1 on cancel/error."""
    try:
        i = argv.index("--align-tool")
        scan_path, model_ply, out_json = argv[i + 1], argv[i + 2], argv[i + 3]
    except (ValueError, IndexError):
        print("usage: --align-tool <scan> <model.ply> <out.json>")
        return 2
    import open3d as o3d
    from .lidar_index import _load_points
    scan = _load_points(scan_path, voxel=0.3, max_points=500_000, log=print)
    model = np.asarray(o3d.io.read_point_cloud(model_ply).points, np.float64)
    if len(scan) < 100 or len(model) < 100:
        print(f"too few points (scan {len(scan)}, model {len(model)})")
        return 1
    res = manual_align(scan, model, out_json)
    return 0 if res is not None else 1


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv))
