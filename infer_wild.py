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
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.data.dataset_wild import EmbedRetargDataset

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
    parser.add_argument('--wild_dataset', "-wd", action='store_true', help='use wild dataset')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

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

fps_in = 30
if opts.wild_dataset:
    if opts.pixel:
        vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(opts.data_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.data_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)
else:
    wild_dataset = EmbedRetargDataset(opts.data_path, scale_range=[1,1])


test_loader = DataLoader(wild_dataset, **testloader_params)

results_all = []
input_all = []
files_all = []
with torch.no_grad():
    for batch_input, file in tqdm(test_loader):
        input_all.append(batch_input)
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if args.flip:    
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        if args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
        results_all.append(predicted_3d_pos.cpu().numpy())
        files_all.append(file[0])


def render_comparison(orig_pos, input_pos, output_pos, save_path, fps=30):
    """
    Render side-by-side comparison video:
      Left:   3D original skeleton  (22 joints, axes: right / front / up)
      Middle: 2D network input      (17 joints, axes: x-right / y-down)
      Right:  3D network output     (17 joints, axes: right / down / front)
    """
    T = min(orig_pos.shape[0], input_pos.shape[0], output_pos.shape[0])

    # --- skeleton topology ---------------------------------------------------
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

    color_l, color_m, color_r = "#02315E", "#00457E", "#2F70AF"

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

    out_vis = np.stack([-output_pos[:T,:,0],
                        -output_pos[:T,:,2],
                        -output_pos[:T,:,1]], axis=-1)
    out_c, out_r = cube_limits(out_vis)

    inp2d = input_pos[:T, :, :2]
    inp_flat = inp2d.reshape(-1, 2)
    inp_lo, inp_hi = inp_flat.min(0), inp_flat.max(0)
    inp_c = (inp_lo + inp_hi) / 2
    inp_r = max((inp_hi - inp_lo).max() / 2, 0.01) * 1.2

    # --- frame loop -----------------------------------------------------------
    skip_frames = 3

    videowriter = imageio.get_writer(save_path, fps=int(fps/skip_frames))

    for f in tqdm(range(0, T, skip_frames), desc="Rendering"):
        fig = plt.figure(figsize=(18, 6))

        # ---- left: original 3D (right, front, up) ---------------------------
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        j = orig_pos[f]
        for p in er_pairs:
            ax1.plot(j[p, 0], j[p, 1], j[p, 2],
                     color=limb_color(p, er_left, er_right),
                     lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax1.set_xlim(orig_c[0]-orig_r, orig_c[0]+orig_r)
        ax1.set_ylim(orig_c[1]-orig_r, orig_c[1]+orig_r)
        ax1.set_zlim(orig_c[2]-orig_r, orig_c[2]+orig_r)
        ax1.view_init(elev=15., azim=-70)
        ax1.set_title('Original 3D (22j)', fontsize=13)

        # ---- middle: input 2D (x-right, y-down) -----------------------------
        ax2 = fig.add_subplot(1, 3, 2)
        j2 = inp2d[f]
        for p in h36m_pairs:
            ax2.plot(j2[p, 0], j2[p, 1],
                     color=limb_color(p, h36m_left, h36m_right),
                     lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax2.set_xlim(inp_c[0]-inp_r, inp_c[0]+inp_r)
        ax2.set_ylim(inp_c[1]+inp_r, inp_c[1]-inp_r)   # flip y so "down" is down
        ax2.set_aspect('equal')
        ax2.set_title('Input 2D (17j)', fontsize=13)

        # ---- right: output 3D (transformed to -x, -z, -y) -------------------
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        j3 = out_vis[f]
        for p in h36m_pairs:
            ax3.plot(j3[p, 0], j3[p, 1], j3[p, 2],
                     color=limb_color(p, h36m_left, h36m_right),
                     lw=2, marker='o', mfc='w', ms=3, mew=1)
        ax3.set_xlim(out_c[0]-out_r, out_c[0]+out_r)
        ax3.set_ylim(out_c[1]-out_r, out_c[1]+out_r)
        ax3.set_zlim(out_c[2]-out_r, out_c[2]+out_r)
        ax3.view_init(elev=12., azim=80)
        ax3.set_title('Output 3D (17j)', fontsize=13)

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()
        videowriter.append_data(frame)
        plt.close(fig)

    videowriter.close()
    print(f"Saved comparison video → {save_path}")


for file, inp, result in zip(files_all, input_all, results_all):
    print(file)
    outfilename = file.replace(opts.data_path, '')
    outfilename = outfilename.replace('.npz', '.mp4')
    outfilename = outfilename.replace('/', '_')

    orig_data = np.load(file)
    orig_pos = orig_data['body_pos_w']          # (T, 22, 3)  right / front / up

    input_pos = inp[0].numpy()                   # (T, 17, 3)  x-right / y-down / conf
    output_pos = result[0]                       # (T, 17, 3)  right / down / front

    T = min(orig_pos.shape[0], input_pos.shape[0], output_pos.shape[0])
    orig_pos   = orig_pos[:T]
    input_pos  = input_pos[:T]
    output_pos = output_pos[:T]

    save_path = os.path.join(opts.out_path, outfilename)
    render_comparison(orig_pos, input_pos, output_pos, save_path, fps=fps_in)
    
    output_npz = {
        'orig_pos': orig_pos,
        'input_pos': input_pos,
        'output_pos': output_pos,
    }
    np.savez(os.path.join(opts.out_path, outfilename.replace('.mp4', '.npz')), **output_npz)