"""Interactive manual alignment for the case auto-align can't crack: repetitive structure (stairs,
balusters, repeated arches) + partial overlap, where FPFH locks a different wrong scale every run.

You drive it visually - drag the MODEL's scale / rotation / position until the orange model sits on
the gray scan, then accept. That writes the chosen similarity (scale + rotation + translation) to a
JSON the app reads as the pre-align; the LiDAR refine then polishes from there.

Two front-ends:
  - manual_align_gui:  Open3D's GUI (SceneWidget + real slider bars) - the nice one.
  - manual_align_keys: the legacy VisualizerWithKeyCallback (+/- and WASD) - rock-stable, the
    automatic fallback if the GUI window can't initialise (some headless/driver setups).
`manual_align` tries the GUI first and falls back to keys, so it always opens something usable.

Runs as its OWN process (Open3D's window owns the event loop, which can't share Tk's). Because the
frozen app is windowed (no console), everything is logged to stdout (the launcher captures it) AND
to %TEMP%/lidar_align_view.log.

CLI (the frozen app dispatches this):
    lidar-align.exe --align-tool <scan.las|.e57|.laz> <model.ply> <out.json>
"""
from __future__ import annotations
import os
import sys
import json
import tempfile
import datetime
import numpy as np

_LOGF = None


def _log(msg):
    global _LOGF
    line = f"[view {datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        if _LOGF is None:
            _LOGF = open(os.path.join(tempfile.gettempdir(), "lidar_align_view.log"), "w",
                         encoding="utf-8")
        _LOGF.write(line + "\n")
        _LOGF.flush()
    except Exception:
        pass


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


def _sim3_from_state(st, cm, cs):
    """The global Sim3 (SfM-model -> scan) for the current state. The model is shown as
    s*R*(X-cm)+cs+t, so the equivalent X' = s*R*X + (cs + t - s*R*cm)."""
    r = _euler(*st["e"])
    s = st["s"]
    t = cs + st["t"] - s * (r @ cm)
    return s, r, t


def _write_sim3(st, cm, cs, out_json):
    s, r, t = _sim3_from_state(st, cm, cs)
    sim = {"scale": float(s), "R": r.tolist(), "t": t.tolist()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sim, f, indent=2)
    _log(f"WROTE {out_json}: scale={s:.4f}  t={np.round(t, 2).tolist()}")
    return sim


# ── GUI front-end (real sliders) ──────────────────────────────────────────────
def manual_align_gui(scan, model0, out_json):
    import open3d as o3d
    from open3d.visualization import gui, rendering
    cm = model0.mean(0)
    cs = scan.mean(0)
    diag = float(np.linalg.norm(np.percentile(scan, 98, 0) - np.percentile(scan, 2, 0))) or 1.0
    s0 = _extent_scale(model0, scan)
    st = {"s": s0, "t": np.zeros(3), "e": np.zeros(3), "ok": False}

    app = gui.Application.instance
    app.initialize()
    w = app.create_window("Match the ORANGE model to the GRAY scan - drag the sliders", 1500, 920)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([1, 1, 1, 1])
    sc = o3d.geometry.PointCloud(); sc.points = o3d.utility.Vector3dVector(scan)
    sc.paint_uniform_color([0.55, 0.55, 0.55])
    mo = o3d.geometry.PointCloud(); mo.points = o3d.utility.Vector3dVector(model0)
    mo.paint_uniform_color([1.0, 0.5, 0.0])
    mg = rendering.MaterialRecord(); mg.shader = "defaultUnlit"; mg.point_size = 2.0
    mm = rendering.MaterialRecord(); mm.shader = "defaultUnlit"; mm.point_size = 3.0
    scene.scene.add_geometry("scan", sc, mg)
    scene.scene.add_geometry("model", mo, mm)
    bounds = sc.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bounds, bounds.get_center())
    w.add_child(scene)

    def update():
        s, r, t = _sim3_from_state(st, cm, cs)
        M = np.eye(4); M[:3, :3] = s * r; M[:3, 3] = t
        scene.scene.set_geometry_transform("model", M)

    em = w.theme.font_size
    panel = gui.Vert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))
    panel.add_child(gui.Label("Drag the sliders to match the orange MODEL onto the gray SCAN,"))
    panel.add_child(gui.Label("then click ACCEPT.  (Mouse-drag in the view = orbit / look around.)"))

    import math
    sliders = {}

    def do_reset():
        sliders["scale"](math.log10(s0))   # each set_value resets the bar, label, state, and view
        for nm in ("Yaw", "Pitch", "Roll", "Move X", "Move Y", "Move Z"):
            sliders[nm](0.0)
        _log("reset")

    def do_accept():
        st["ok"] = True; _log("ACCEPT"); app.quit()

    # buttons sit at the TOP of the panel so Accept/Cancel are always on screen, even if the window
    # is taller than the display and the lower sliders get clipped
    row = gui.Horiz(0.4 * em)
    b_ok = gui.Button("Accept"); b_ok.set_on_clicked(do_accept)
    b_rs = gui.Button("Reset"); b_rs.set_on_clicked(do_reset)
    b_cx = gui.Button("Cancel"); b_cx.set_on_clicked(lambda: (_log("CANCEL"), app.quit()))
    row.add_child(b_ok); row.add_child(b_rs); row.add_child(b_cx)
    panel.add_child(row)
    panel.add_child(gui.Label(""))

    def _slider(label, lo, hi, val, on_change):
        panel.add_child(gui.Label(label))
        sl = gui.Slider(gui.Slider.DOUBLE)
        sl.set_limits(lo, hi)
        out = gui.Label("")
        def cb(v):
            out.text = on_change(v)
        sl.set_on_value_changed(cb)
        def set_value(v):           # a programmatic double_value set does NOT fire cb, so do it here
            sl.double_value = v
            cb(v)                   # -> refreshes the value label + applies the transform
        set_value(val)
        panel.add_child(sl)
        panel.add_child(out)
        return set_value            # callers store this to drive the slider from code (e.g. Reset)

    sliders["scale"] = _slider("Scale", math.log10(s0) - 1.3, math.log10(s0) + 1.3, math.log10(s0),
                               lambda v: (st.__setitem__("s", 10.0 ** v), update(),
                                          f"scale = {st['s']:.4f}")[-1])
    for idx, nm in ((0, "Yaw"), (1, "Pitch"), (2, "Roll")):
        def mk_rot(i, name):
            def f(v):
                st["e"][i] = v; update(); return f"{name} = {v:.0f}°"
            return f
        sliders[nm] = _slider(nm + " (deg)", -180, 180, 0, mk_rot(idx, nm))
    for idx, nm in ((0, "Move X"), (1, "Move Y"), (2, "Move Z")):
        def mk_mov(i, name):
            def f(v):
                st["t"][i] = v; update(); return f"{name} = {v:+.2f} m"
            return f
        sliders[nm] = _slider(nm, -diag, diag, 0, mk_mov(idx, nm))

    w.add_child(panel)

    def on_layout(ctx):
        r = w.content_rect
        pw = int(20 * em)
        scene.frame = gui.Rect(r.x, r.y, max(r.width - pw, 1), r.height)
        panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)
    w.set_on_layout(on_layout)

    update()
    _log(f"GUI window open ({len(scan):,} scan / {len(model0):,} model pts; initial scale {s0:.3f})")
    app.run()
    return _write_sim3(st, cm, cs, out_json) if st["ok"] else None


# ── legacy key-driven front-end (fallback) ────────────────────────────────────
_LEGEND = ("CONTROLS:  +/- scale   W/S move Y   A/D move X   R/F move Z   "
           "Q/E yaw   Z/X pitch   C/V roll   0 reset   drag-mouse look   ENTER accept   ESC cancel")


def manual_align_keys(scan, model0, out_json):
    import open3d as o3d
    cm = model0.mean(0)
    cs = scan.mean(0)
    diag = float(np.linalg.norm(np.percentile(scan, 98, 0) - np.percentile(scan, 2, 0))) or 1.0
    s0 = _extent_scale(model0, scan)
    st = {"s": s0, "t": np.zeros(3), "e": np.zeros(3), "ok": False}
    _log(_LEGEND)

    sc = o3d.geometry.PointCloud(); sc.points = o3d.utility.Vector3dVector(scan)
    sc.paint_uniform_color([0.55, 0.55, 0.55])
    mo = o3d.geometry.PointCloud(); mo.paint_uniform_color([1.0, 0.5, 0.0])

    def render():
        r = _euler(*st["e"])
        mo.points = o3d.utility.Vector3dVector(st["s"] * ((model0 - cm) @ r.T) + cs + st["t"])

    render()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Match ORANGE model to GRAY scan  -  +/- scale  WASD/RF move  QE/ZX/CV rotate "
                      " 0 reset  ENTER ok  ESC cancel", 1360, 860)
    vis.add_geometry(sc); vis.add_geometry(mo)

    def _bump(key, fn):
        def cb(v):
            fn(); render(); v.update_geometry(mo); return False
        vis.register_key_callback(ord(key), cb)

    mv = diag * 0.01
    _bump("=", lambda: st.update(s=st["s"] * 1.02)); _bump("+", lambda: st.update(s=st["s"] * 1.02))
    _bump("-", lambda: st.update(s=st["s"] / 1.02)); _bump("_", lambda: st.update(s=st["s"] / 1.02))
    _bump("D", lambda: st["t"].__setitem__(0, st["t"][0] + mv))
    _bump("A", lambda: st["t"].__setitem__(0, st["t"][0] - mv))
    _bump("W", lambda: st["t"].__setitem__(1, st["t"][1] + mv))
    _bump("S", lambda: st["t"].__setitem__(1, st["t"][1] - mv))
    _bump("R", lambda: st["t"].__setitem__(2, st["t"][2] + mv))
    _bump("F", lambda: st["t"].__setitem__(2, st["t"][2] - mv))
    _bump("Q", lambda: st["e"].__setitem__(0, st["e"][0] + 1.0))
    _bump("E", lambda: st["e"].__setitem__(0, st["e"][0] - 1.0))
    _bump("Z", lambda: st["e"].__setitem__(1, st["e"][1] + 1.0))
    _bump("X", lambda: st["e"].__setitem__(1, st["e"][1] - 1.0))
    _bump("C", lambda: st["e"].__setitem__(2, st["e"][2] + 1.0))
    _bump("V", lambda: st["e"].__setitem__(2, st["e"][2] - 1.0))
    _bump("0", lambda: (st.update(s=s0), st["t"].fill(0.0), st["e"].fill(0.0)))

    def accept(v):
        st["ok"] = True; _log("ACCEPT"); v.close(); return False
    vis.register_key_callback(257, accept)
    vis.register_key_callback(256, lambda v: (_log("CANCEL"), v.close(), False)[2])

    vis.run()
    vis.destroy_window()
    return _write_sim3(st, cm, cs, out_json) if st["ok"] else None


def manual_align(scan_pts, model_pts, out_json):
    """Open the GUI slider window; fall back to the key-driven viewer if it can't initialise."""
    scan = np.ascontiguousarray(scan_pts, np.float64)
    model0 = np.ascontiguousarray(model_pts, np.float64)
    try:
        return manual_align_gui(scan, model0, out_json)
    except Exception as e:
        _log(f"GUI front-end failed ({e}); falling back to the key-driven viewer")
        return manual_align_keys(scan, model0, out_json)


def view_cloud(ply_path):
    """Just show a point cloud (Open3D's simplest, most robust viewer) so you can eyeball whether the
    SfM came out recognizable BEFORE sinking hours into aligning it. Returns 0 on success."""
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(ply_path))
    n = len(pc.points)
    if n == 0:
        _log(f"empty / unreadable cloud: {ply_path}")
        return 1
    _log(f"previewing {n:,} points - is the structure recognizable, or is it noise? "
         f"close the window when done.")
    o3d.visualization.draw_geometries(
        [pc], window_name="SfM model preview - recognizable structure, or noise?",
        width=1360, height=860)
    return 0


def cli_main(argv):
    if "--view-model" in argv:
        i = argv.index("--view-model")
        try:
            return view_cloud(argv[i + 1])
        except Exception:
            import traceback
            _log("ERROR:\n" + traceback.format_exc())
            return 1
    try:
        i = argv.index("--align-tool")
        scan_path, model_ply, out_json = argv[i + 1], argv[i + 2], argv[i + 3]
    except (ValueError, IndexError):
        print("usage: --align-tool <scan> <model.ply> <out.json>")
        return 2
    _log(f"--align-tool: scan={scan_path}  model={model_ply}  out={out_json}")
    try:
        import open3d as o3d
        from .lidar_index import _load_points
        _log("loading scan (downsample 0.3 m, cap 500k)…")
        scan = _load_points(scan_path, voxel=0.3, max_points=500_000, log=_log)
        _log("loading model PLY…")
        model = np.asarray(o3d.io.read_point_cloud(model_ply).points, np.float64)
        if len(scan) < 100 or len(model) < 100:
            _log(f"too few points (scan {len(scan)}, model {len(model)}) - aborting")
            return 1
        res = manual_align(scan, model, out_json)
        return 0 if res is not None else 1
    except Exception:
        import traceback
        _log("ERROR:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv))
