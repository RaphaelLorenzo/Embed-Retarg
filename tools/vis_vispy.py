#!/usr/bin/env python3
"""
Interactive vispy skeleton comparison viewer.

Usage:
    python vis_vispy.py path/to/output.npz [--fps 30]

Controls:
    Space        Play / Pause
    Right / Left Step forward / backward
    R            Reset to frame 0
    Q / Esc      Quit
"""
import argparse
import numpy as np
from vispy import app, scene


# -- Skeleton topology ---------------------------------------------------------

ER_PAIRS = [
    [0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
    [7,10],[8,11],[9,12],[12,13],[12,14],[12,15],
    [13,16],[14,17],[16,18],[17,19],[18,20],[19,21],
]
ER_LEFT  = {(0,1),(1,4),(4,7),(7,10),(12,13),(13,16),(16,18),(18,20)}
ER_RIGHT = {(0,2),(2,5),(5,8),(8,11),(12,14),(14,17),(17,19),(19,21)}

H36M_PAIRS = [
    [0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],
    [8,9],[8,11],[8,14],[9,10],[11,12],[12,13],[14,15],[15,16],
]
H36M_LEFT  = {(8,11),(11,12),(12,13),(0,4),(4,5),(5,6)}
H36M_RIGHT = {(8,14),(14,15),(15,16),(0,1),(1,2),(2,3)}

G1_PAIRS = [
    [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],
    [0,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],
    [0,16],[16,17],[17,18],[18,19],[19,20],
    [0,8],[18,21],
    [18,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],
    [18,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],
]
G1_LEFT = {
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
    (18,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),
}
G1_RIGHT = {
    (0,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),
    (18,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),
}

COL_L = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)   # red   – left
COL_M = np.array([0.2, 0.8, 0.2, 1.0], dtype=np.float32)   # green – center
COL_R = np.array([0.3, 0.5, 1.0, 1.0], dtype=np.float32)   # blue  – right


def build_segments(joints, pairs, left_set, right_set):
    """Flatten skeleton into segment positions + per-vertex RGBA colors."""
    n = len(pairs)
    pos = np.empty((n * 2, 3), dtype=np.float32)
    col = np.empty((n * 2, 4), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        pos[2 * i]     = joints[a]
        pos[2 * i + 1] = joints[b]
        tp = (a, b)
        c = COL_L if tp in left_set else (COL_R if tp in right_set else COL_M)
        col[2 * i] = col[2 * i + 1] = c
    return pos, col


class SkeletonViewer:
    def __init__(self, npz_path, fps=30):
        data = np.load(npz_path)
        orig = data['orig_pos']      # (T, 22, 3)  axes: right / front / up
        out  = data['output_pos']    # (T, 17, 3)  axes: right / down  / front

        self.T = T = min(orig.shape[0], out.shape[0])
        self.fps = fps
        self.frame = 0
        self.playing = True

        # vispy TurntableCamera: x-right, y-forward, z-up (default up='+z')
        #
        # orig_pos (right, front, up) already matches (x, y, z) – no transform
        self.orig = orig[:T].astype(np.float32)
        # output_pos (right, down, front) → (right, front, -down) = (x, y, z-up)
        if False:
            self.out = out[:T, :, [0, 2, 1]].astype(np.float32).copy()
            self.out[:, :, 2] *= -1
        else:
            self.out = out[:T].astype(np.float32)

        # -- Canvas & grid layout ----------------------------------------------
        self.canvas = scene.SceneCanvas(
            keys='interactive', title='Skeleton Comparison',
            size=(1400, 700), show=True, bgcolor='#1e1e1e')

        grid = self.canvas.central_widget.add_grid(margin=10)

        lbl_l = grid.add_widget(
            scene.Label('Original 3D  (22 joints)', color='white'),
            row=0, col=0)
        lbl_r = grid.add_widget(
            scene.Label('Output 3D  (17 joints)', color='white'),
            row=0, col=1)
        lbl_l.height_max = lbl_r.height_max = 30

        self.view_l = grid.add_view(row=1, col=0, camera='turntable',
                                    border_color='#444')
        self.view_r = grid.add_view(row=1, col=1, camera='turntable',
                                    border_color='#444')

        self.frame_label = grid.add_widget(
            scene.Label(self._status_text(0), color='#888'),
            row=2, col=0, col_span=2)
        self.frame_label.height_max = 25

        # -- Per-view camera setup ---------------------------------------------
        for view, d in [(self.view_l, self.orig), (self.view_r, self.out)]:
            flat = d.reshape(-1, 3)
            lo, hi = flat.min(0), flat.max(0)
            center = (lo + hi) / 2
            span = (hi - lo).max()
            cam = view.camera
            cam.center = center.tolist()
            cam.distance = float(span * 2.0)
            cam.elevation = 20
            cam.azimuth = 135
            cam.fov = 45

        # -- 3D axis indicators ------------------------------------------------
        for view in (self.view_l, self.view_r):
            scene.visuals.XYZAxis(parent=view.scene)

        # -- Skeleton visuals --------------------------------------------------
        lp, lc = build_segments(self.orig[0], ER_PAIRS, ER_LEFT, ER_RIGHT)
        self.line_l = scene.visuals.Line(
            pos=lp, color=lc, connect='segments', width=3,
            parent=self.view_l.scene)
        self.mark_l = scene.visuals.Markers(parent=self.view_l.scene)
        self.mark_l.set_data(self.orig[0], face_color='white',
                             edge_color=COL_M, size=8, edge_width=2)

        if self.out.shape[1] == 17:
            rp, rc = build_segments(self.out[0], H36M_PAIRS, H36M_LEFT, H36M_RIGHT)
        elif self.out.shape[1] == 38:
            rp, rc = build_segments(self.out[0], G1_PAIRS, G1_LEFT, G1_RIGHT)
        else:
            raise ValueError(f"Unexpected number of joints: {self.out.shape[1]}")
        self.line_r = scene.visuals.Line(
            pos=rp, color=rc, connect='segments', width=3,
            parent=self.view_r.scene)
        self.mark_r = scene.visuals.Markers(parent=self.view_r.scene)
        self.mark_r.set_data(self.out[0], face_color='white',
                             edge_color=COL_M, size=8, edge_width=2)

        # -- Timer & events ----------------------------------------------------
        self.timer = app.Timer(interval=1.0 / fps, connect=self.on_timer,
                               start=True)
        self.canvas.events.key_press.connect(self.on_key_press)

    def _status_text(self, f):
        state = "Playing" if self.playing else "Paused"
        return (f'Frame {f} / {self.T}   ({state})   '
                '[Space] play/pause   [\u2190\u2192] step   [R] reset')

    def set_frame(self, f):
        
        orig_root_pos = self.orig[self.frame, 0]
        out_root_pos = self.out[self.frame, 0]
        
        print(f"orig_root_pos: ({orig_root_pos[0]:.2f}, {orig_root_pos[1]:.2f}, {orig_root_pos[2]:.2f})")
        print(f"out_root_pos: ({out_root_pos[0]:.2f}, {out_root_pos[1]:.2f}, {out_root_pos[2]:.2f})")
        
        self.frame = f % self.T

        lp, lc = build_segments(self.orig[self.frame],
                                ER_PAIRS, ER_LEFT, ER_RIGHT)
        self.line_l.set_data(pos=lp, color=lc)
        self.mark_l.set_data(self.orig[self.frame], face_color='white',
                             edge_color=COL_M, size=8, edge_width=2)

        if self.out.shape[1] == 17:
            rp, rc = build_segments(self.out[self.frame],
                                    H36M_PAIRS, H36M_LEFT, H36M_RIGHT)
        elif self.out.shape[1] == 38:
            rp, rc = build_segments(self.out[self.frame], G1_PAIRS, G1_LEFT, G1_RIGHT)
        else:
            raise ValueError(f"Unexpected number of joints: {self.out.shape[1]}")
        
        self.line_r.set_data(pos=rp, color=rc)
        self.mark_r.set_data(self.out[self.frame], face_color='white',
                             edge_color=COL_M, size=8, edge_width=2)

        self.frame_label.text = self._status_text(self.frame)
        self.canvas.update()

    def on_timer(self, event):
        if self.playing:
            self.set_frame(self.frame + 1)

    def on_key_press(self, event):
        if event.key == 'Space':
            self.playing = not self.playing
            self.frame_label.text = self._status_text(self.frame)
        elif event.key == 'Right':
            self.playing = False
            self.set_frame(self.frame + 1)
        elif event.key == 'Left':
            self.playing = False
            self.set_frame(self.frame - 1)
        elif event.key == 'R':
            self.set_frame(0)
        elif event.key in ('Q', 'Escape'):
            self.canvas.close()
            app.quit()

    def run(self):
        app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('npz', help='Path to .npz saved by infer_wild.py')
    parser.add_argument('--fps', type=int, default=30,
                        help='Playback FPS (default: 30)')
    args = parser.parse_args()

    viewer = SkeletonViewer(args.npz, fps=args.fps)
    viewer.run()
