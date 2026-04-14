
## RENDERING FUNCTIONS ##
from tools.conversion_tools import g12h36m
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy

def quat_to_ypr(q):
    """Quaternion (x, y, z, w) → (yaw, pitch, roll) in degrees."""
    yaw, pitch, roll = R_scipy.from_quat(q).as_euler('ZYX', degrees=True)
    return yaw, pitch, roll

def render_comparison(orig_pos, input_pos, output_pos, target_pos, save_path, fps=30,
                      g1_pos=None, orig_quat=None, g1_quat=None):
    """
    Render side-by-side comparison video (3 or 4 panels).

    Panels:
      1. 3D original skeleton  (22 joints, axes: right / front / up)
      2. 2D network input      (17 joints, axes: x-right / y-down)
      3. 3D network output     (17 joints, axes: right / down / front)
      4. 3D G1 robot skeleton  (38 joints, axes: right / front / up)  [optional]
    """
    T = min(orig_pos.shape[0], input_pos.shape[0], output_pos.shape[0], target_pos.shape[0])
    if g1_pos is not None:
        T = min(T, g1_pos.shape[0])

    # --- skeleton topologies --------------------------------------------------
    er_pairs = [
        [0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
        [7,10],[8,11],[9,12],[12,13],[12,14],[12,15],
        [13,16],[14,17],[16,18],[17,19],[18,20],[19,21],
    ]
    er_left  = {(0,1),(1,4),(4,7),(7,10),(12,13),(13,16),(16,18),(18,20)}
    er_right = {(0,2),(2,5),(5,8),(8,11),(12,14),(14,17),(17,19),(19,21)}

    h36m_pairs = [
        [0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],
        [8,9],[8,11],[8,14],[9,10],[11,12],[12,13],[14,15],[15,16],
    ]
    h36m_left  = {(8,11),(11,12),(12,13),(0,4),(4,5),(5,6)}
    h36m_right = {(8,14),(14,15),(15,16),(0,1),(1,2),(2,3)}

    # G1 robot 38 joints
    # 0:pelvis  1-7:left leg  8:pelvis_contour  9-15:right leg
    # 16-17:waist  18:torso  19:head  20:head_mocap  21:imu_in_torso
    # 22-29:left arm  30-37:right arm
    g1_pairs = [
        [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],
        [0,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],
        [0,16],[16,17],[17,18],[18,19],[19,20],
        [0,8],[18,21],
        [18,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],
        [18,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],
    ]
    g1_left = {
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
        (18,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),
    }
    g1_right = {
        (0,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),
        (18,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),
    }

    color_l = "#FF3333"   # red   – left
    color_m = "#33CC33"   # green – center
    color_r = "#4D80FF"   # blue  – right

    def limb_color(pair, left_set, right_set):
        tp = tuple(pair)
        if tp in left_set:  return color_l
        if tp in right_set: return color_r
        return color_m

    # --- precompute stable axis limits ----------------------------------------
    def cube_limits(data):
        flat = data.reshape(-1, 3)
        lo, hi = flat.min(0), flat.max(0)
        center = (lo + hi) / 2
        hr = max((hi - lo).max() / 2, 0.01) * 1.2
        return center, hr

    orig_c, orig_r = cube_limits(orig_pos[:T])

    if "2026" in save_path:
        # new output format
        out_vis = output_pos[:T]
    else:
        # old output format
        out_vis = np.stack([-output_pos[:T,:,0],
                            -output_pos[:T,:,2],
                            -output_pos[:T,:,1]], axis=-1)
    out_c, out_r = cube_limits(out_vis)

    predtype = None
    if out_vis.shape[1] == 17:
        predtype = "h36m"
    elif out_vis.shape[1] == 38:
        predtype = "g1"
    else:
        raise ValueError(f"Unexpected number of joints: {out_vis.shape[1]}")

    if g1_pos is not None:
        n_panels = 6 if predtype == "h36m" else 5
    else:
        n_panels = 3

    inp2d = input_pos[:T, :, :2]
    inp_all = input_pos[:T]
    inp_flat = inp2d.reshape(-1, 2)
    inp_lo, inp_hi = inp_flat.min(0), inp_flat.max(0)
    inp_c = (inp_lo + inp_hi) / 2
    inp_r = max((inp_hi - inp_lo).max() / 2, 0.01) * 1.2

    if g1_pos is not None:
        g1_c, g1_r = cube_limits(g1_pos[:T])
    
    g1rr_c, g1rr_r = cube_limits(target_pos[:T])

    g1_as_h36m = None
    g1h_c, g1h_r = None, None
    if g1_pos is not None and predtype == "h36m":
        g1_as_h36m = g12h36m(target_pos[:T])
        g1h_c, g1h_r = cube_limits(g1_as_h36m)

    # --- frame loop -----------------------------------------------------------
    skip_frames = 3

    videowriter = imageio.get_writer(save_path, fps=int(fps/skip_frames))

    for f in tqdm(range(0, T, skip_frames), desc="Rendering"):
        fig = plt.figure(figsize=(6 * n_panels, 6))

        # ---- panel 1: original 3D (right, front, up) ------------------------
        ax1 = fig.add_subplot(1, n_panels, 1, projection='3d')
        j = orig_pos[f]
        for p in er_pairs:
            ax1.plot(j[p, 0], j[p, 1], j[p, 2],
                     color=limb_color(p, er_left, er_right),
                     lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax1.set_xlim(orig_c[0]-orig_r, orig_c[0]+orig_r)
        ax1.set_ylim(orig_c[1]-orig_r, orig_c[1]+orig_r)
        ax1.set_zlim(orig_c[2]-orig_r, orig_c[2]+orig_r)
        ax1.view_init(elev=15., azim=-70)
        r0 = orig_pos[f, 0]
        t1 = f'Original 3D (22j)\nroot=({r0[0]:.2f}, {r0[1]:.2f}, {r0[2]:.2f})'
        if orig_quat is not None:
            y0, p0, rl0 = quat_to_ypr(orig_quat[f, 0])
            t1 += f'\nypr=({y0:.1f}\u00b0, {p0:.1f}\u00b0, {rl0:.1f}\u00b0)'
        ax1.set_title(t1, fontsize=11)

        # ---- panel 2: input 2D (x-right, y-down) ----------------------------
        ax2 = fig.add_subplot(1, n_panels, 2)
        j2 = inp2d[f]
        for p in h36m_pairs:
            ax2.plot(j2[p, 0], j2[p, 1],
                     color=limb_color(p, h36m_left, h36m_right),
                     lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax2.set_xlim(inp_c[0]-inp_r, inp_c[0]+inp_r)
        ax2.set_ylim(inp_c[1]+inp_r, inp_c[1]-inp_r)   # flip y so "down" is down
        ax2.set_aspect('equal')
        r1 = inp_all[f, 0]
        ax2.set_title(f'Input 2D (17j)\nroot=({r1[0]:.2f}, {r1[1]:.2f}, conf={r1[2]:.2f})',
                      fontsize=12)

        # ---- panel 3: output 3D (transformed to -x, -z, -y) -----------------
        ax3 = fig.add_subplot(1, n_panels, 3, projection='3d')
        j3 = out_vis[f]
        if predtype == "h36m":
            for p in h36m_pairs:
                ax3.plot(j3[p, 0], j3[p, 1], j3[p, 2],
                        color=limb_color(p, h36m_left, h36m_right),
                        lw=2, marker='o', mfc='w', ms=3, mew=1)
        elif predtype == "g1":
            for p in g1_pairs:
                ax3.plot(j3[p, 0], j3[p, 1], j3[p, 2],
                        color=limb_color(p, g1_left, g1_right),
                        lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax3.set_xlim(out_c[0]-out_r, out_c[0]+out_r)
        ax3.set_ylim(out_c[1]-out_r, out_c[1]+out_r)
        ax3.set_zlim(out_c[2]-out_r, out_c[2]+out_r)
        ax3.view_init(elev=15., azim=-70)
        r2 = output_pos[f, 0]
        njoints = out_vis.shape[1]
        ax3.set_title(f'Output 3D ({njoints}j)\nroot=({r2[0]:.2f}, {r2[1]:.2f}, {r2[2]:.2f})',
                      fontsize=12)

        if g1_pos is not None:
            # ---- panel 4 (optional): G1 root-aligned → H36M (17j) -----------
            if predtype == "h36m":
                ax4 = fig.add_subplot(1, n_panels, 4, projection='3d')
                jh = g1_as_h36m[f]
                for p in h36m_pairs:
                    ax4.plot(jh[p, 0], jh[p, 1], jh[p, 2],
                                color=limb_color(p, h36m_left, h36m_right),
                                lw=2, marker='o', mfc='w', ms=3, mew=1)
                ax4.set_xlim(g1h_c[0]-g1h_r, g1h_c[0]+g1h_r)
                ax4.set_ylim(g1h_c[1]-g1h_r, g1h_c[1]+g1h_r)
                ax4.set_zlim(g1h_c[2]-g1h_r, g1h_c[2]+g1h_r)
                ax4.view_init(elev=15., azim=-70)
                ax4.set_title('Target 3D (G1→H36M) (17j)\nroot-aligned', fontsize=11)

            # ---- panel 5: G1 root-relative 3D -------------------------------
            ax5_idx = 5 if predtype == "h36m" else 4
            ax5 = fig.add_subplot(1, n_panels, ax5_idx, projection='3d')
            jrr = target_pos[f]
            for p in g1_pairs:
                ax5.plot(jrr[p, 0], jrr[p, 1], jrr[p, 2],
                            color=limb_color(p, g1_left, g1_right),
                            lw=2, marker='o', mfc='w', ms=2, mew=1)
            ax5.set_xlim(g1rr_c[0]-g1rr_r, g1rr_c[0]+g1rr_r)
            ax5.set_ylim(g1rr_c[1]-g1rr_r, g1rr_c[1]+g1rr_r)
            ax5.set_zlim(g1rr_c[2]-g1rr_r, g1rr_c[2]+g1rr_r)
            ax5.view_init(elev=15., azim=-70)
            if predtype == "g1":
                ax5.set_title('Target 3D (38j)\nroot-aligned', fontsize=11)
            elif predtype == "h36m":
                ax5.set_title('Target 3D (before conversion) (38j)\nroot-aligned', fontsize=11)

            # ---- panel 6: G1 robot 3D (right, front, up) --------------------
            ax6_idx = 6 if predtype == "h36m" else 5
            ax6 = fig.add_subplot(1, n_panels, ax6_idx, projection='3d')
            jg = g1_pos[f]
            for p in g1_pairs:
                ax6.plot(jg[p, 0], jg[p, 1], jg[p, 2],
                            color=limb_color(p, g1_left, g1_right),
                            lw=2, marker='o', mfc='w', ms=2, mew=1)
            ax6.set_xlim(g1_c[0]-g1_r, g1_c[0]+g1_r)
            ax6.set_ylim(g1_c[1]-g1_r, g1_c[1]+g1_r)
            ax6.set_zlim(g1_c[2]-g1_r, g1_c[2]+g1_r)
            ax6.view_init(elev=15., azim=-70)
            rg = g1_pos[f, 0]
            t6 = f'G1 Robot 3D (38j)\nroot=({rg[0]:.2f}, {rg[1]:.2f}, {rg[2]:.2f})'
            if g1_quat is not None:
                yg, pg, rlg = quat_to_ypr(g1_quat[f, 0])
                t6 += f'\nypr=({yg:.1f}\u00b0, {pg:.1f}\u00b0, {rlg:.1f}\u00b0)'
            ax6.set_title(t6, fontsize=11)

        # fig.tight_layout()
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()
        videowriter.append_data(frame)
        plt.close(fig)

    videowriter.close()
    print(f"Saved comparison video → {save_path}")
