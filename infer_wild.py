import os
import io
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
# from lib.data.dataset_wild import WildDetDataset
from lib.data.dataset_embedretarg import EmbedRetargDataset
from tools.conversion_tools import g12h36m

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoints/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-d', '--data_path', type=str, help='data path', default='/home/raphael/Projects/github/accad_subset_random_shapes')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path', default='results/')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--subset', type=str, default='special_walking', help='subset to use')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

args.num_new_joints = 38 if args.remap_joints_head else 0
model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 1,
          'persistent_workers': True,
          'drop_last': False
}

os.makedirs(opts.out_path, exist_ok=True)

dataset = EmbedRetargDataset(opts.data_path, 
                            max_len=opts.clip_len,
                            stride=args.data_stride,
                            root_rel_target=args.rootrel,
                            scale_by=args.scale_by,
                            scale_range=args.scale_range,
                            subset=opts.subset)


test_loader = DataLoader(dataset, **testloader_params)

results_all = []
input_all = []
files_all = []
target_pos_all = []
seq_idx_all = []
with torch.no_grad():
    for batch_input, target_pos, file, seq_idx in tqdm(test_loader):
        input_all.append(batch_input)
        target_pos_all.append(target_pos)
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
            
        # if args.no_conf:
        #     batch_input = batch_input[:, :, :, :2]
        # if args.flip:    
        #     batch_input_flip = flip_data(batch_input)
        #     predicted_3d_pos_1 = model_pos(batch_input)
        #     predicted_3d_pos_flip = model_pos(batch_input_flip)
        #     predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
        #     predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        # else:
        
        predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        # if args.gt_2d:
        #     predicted_3d_pos[...,:2] = batch_input[...,:2]
        
        results_all.append(predicted_3d_pos) #.cpu().numpy())
        files_all.append(file[0])
        seq_idx_all.append(seq_idx)






## RENDERING FUNCTIONS ##

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
    n_panels = 6 if g1_pos is not None else 3

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

    if "2026" in opts.evaluate:
        # new output format
        out_vis = output_pos[:T]
    else:
        # old output format
        out_vis = np.stack([-output_pos[:T,:,0],
                            -output_pos[:T,:,2],
                            -output_pos[:T,:,1]], axis=-1)
    out_c, out_r = cube_limits(out_vis)

    inp2d = input_pos[:T, :, :2]
    inp_all = input_pos[:T]
    inp_flat = inp2d.reshape(-1, 2)
    inp_lo, inp_hi = inp_flat.min(0), inp_flat.max(0)
    inp_c = (inp_lo + inp_hi) / 2
    inp_r = max((inp_hi - inp_lo).max() / 2, 0.01) * 1.2

    if g1_pos is not None:
        g1_c, g1_r = cube_limits(g1_pos[:T])
    
    g1rr_c, g1rr_r = cube_limits(target_pos[:T])

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
        if out_vis.shape[1] == 17:
            for p in h36m_pairs:
                ax3.plot(j3[p, 0], j3[p, 1], j3[p, 2],
                        color=limb_color(p, h36m_left, h36m_right),
                        lw=2, marker='o', mfc='w', ms=3, mew=1)
        elif out_vis.shape[1] == 38:
            for p in g1_pairs:
                ax3.plot(j3[p, 0], j3[p, 1], j3[p, 2],
                        color=limb_color(p, g1_left, g1_right),
                        lw=2, marker='o', mfc='w', ms=3, mew=1)
        else:
            raise ValueError(f"Unexpected number of joints: {out_vis.shape[1]}")
        ax3.set_xlim(out_c[0]-out_r, out_c[0]+out_r)
        ax3.set_ylim(out_c[1]-out_r, out_c[1]+out_r)
        ax3.set_zlim(out_c[2]-out_r, out_c[2]+out_r)
        ax3.view_init(elev=15., azim=-70)
        r2 = output_pos[f, 0]
        njoints = out_vis.shape[1]
        ax3.set_title(f'Output 3D ({njoints}j)\nroot=({r2[0]:.2f}, {r2[1]:.2f}, {r2[2]:.2f})',
                      fontsize=12)

        # ---- panel 4: G1 root-aligned → H36M (17j) --------------------------
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
        ax4.set_title('G1→H36M (17j)\nroot-aligned', fontsize=11)

        # ---- panel 5: G1 root-relative 3D -----------------------------------
        ax5 = fig.add_subplot(1, n_panels, 5, projection='3d')
        jrr = target_pos[f]
        for p in g1_pairs:
            ax5.plot(jrr[p, 0], jrr[p, 1], jrr[p, 2],
                        color=limb_color(p, g1_left, g1_right),
                        lw=2, marker='o', mfc='w', ms=2, mew=1)
        ax5.set_xlim(g1rr_c[0]-g1rr_r, g1rr_c[0]+g1rr_r)
        ax5.set_ylim(g1rr_c[1]-g1rr_r, g1rr_c[1]+g1rr_r)
        ax5.set_zlim(g1rr_c[2]-g1rr_r, g1rr_c[2]+g1rr_r)
        ax5.view_init(elev=15., azim=-70)
        t5 = 'G1 Root-Aligned (38j)\nroot=(0, 0, 0) ypr=(0\u00b0, 0\u00b0, 0\u00b0)'
        ax5.set_title(t5, fontsize=11)
                
        # ---- panel 6: G1 robot 3D (right, front, up) ------------------------
        ax6 = fig.add_subplot(1, n_panels, 6, projection='3d')
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

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()
        videowriter.append_data(frame)
        plt.close(fig)

    videowriter.close()
    print(f"Saved comparison video → {save_path}")


for file, inp, result, target_pos, seq_idx in zip(files_all, input_all, results_all, target_pos_all, seq_idx_all):
    print(file)
    outfilename = file.replace(opts.data_path, '')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    seq_idx_str = f'{seq_idx[0]:03d}'
    outfilename = outfilename.replace('.npz', f'{timestamp}_{seq_idx_str}.mp4')
    outfilename = outfilename.replace('/', '_')

    # data from the inference loop
    input_pos = inp[0].cpu().numpy()                           # (T, 17, 3)  x-right / y-down / conf
    output_pos = result[0].cpu().numpy()                       # (T, 17, 3)  right / down / front
    target_pos = target_pos[0].cpu().numpy()                   # (T, 38, 3)  right / front / up

    # data from the original file
    orig_data = np.load(file)
    start = seq_idx * 81
    end = start + opts.clip_len
    orig_pos = orig_data['body_pos_w'][start:end]          # (T, 22, 3)  right / front / up
    orig_quat = orig_data['body_quat_w'][start:end]      # (T, 22, 4)  x, y, z, w
    orig_quat = orig_quat[:, :, [1, 2, 3, 0]] # (T, 22, 4)  x, y, z, w
    
    # data from the G1 robot file
    g1_path = os.path.join(os.path.dirname(file), 'motion_shape_g1.npz')
    assert os.path.exists(g1_path), f"G1 data not found at {g1_path}"
    g1_data = np.load(g1_path)
    g1_pos = g1_data['body_pos_w'][start:end]          # (T, 38, 3)  right / front / up
    g1_quat = g1_data['body_quat_w'][start:end] if g1_pos is not None else None # (T, 38, 4)  w, x, y, z
    g1_quat = g1_quat[:, :, [1, 2, 3, 0]] # (T, 38, 4)  x, y, z, w
    print(g1_data["body_link_names"])

    # render the comparison video
    fps_in = 30
    save_path = os.path.join(opts.out_path, outfilename)
    render_comparison(orig_pos, input_pos, output_pos, target_pos, save_path,
                      fps=fps_in, g1_pos=g1_pos,
                      orig_quat=orig_quat, g1_quat=g1_quat)

    # save the data to a npz file
    output_npz = {
        'orig_pos': orig_pos,
        'input_pos': input_pos,
        'output_pos': output_pos,
    }
    if g1_pos is not None:
        output_npz['g1_pos'] = g1_pos
    np.savez(os.path.join(opts.out_path, outfilename.replace('.mp4', '.npz')), **output_npz)